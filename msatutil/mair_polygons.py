import argparse
from pathlib import Path
from typing import Callable, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from affine import Affine
from netCDF4 import Dataset
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from msatutil.mair_targets import PONumber, get_target_dict


def simplify_to_exterior_safe(
    geometry: Union[Polygon, MultiPolygon],
    max_points: int = 500,
    initial_tolerance: float = 1e-5,
    scale_factor: float = 1.5,
) -> Polygon:
    """
    Converts an input Polygon or MultiPolygon into a Polygon that has less than max_points vertices
    For a MultiPolygon, only keep the exterior of the largest geometry

    Inputs:
        geometry (Union[Polygon, MultiPolygon]): a shapely polygon
        max_points (int): the output polygon will have less than this many points
        initial_tolerance (float): starting tolerance to draw the exterior polygon
        scale_factor (float): tolerance*scale_factor at each iteration until polygon has < max_points
    Outputs:
        simplified (Polygon): polygon with less than max_points
    """
    merged = unary_union(geometry)

    # Check type
    if merged.is_empty:
        raise ValueError("Resulting geometry is empty.")
    if isinstance(merged, Polygon):
        exterior_poly = Polygon(merged.exterior.coords)
    elif isinstance(merged, MultiPolygon):
        # Merge all parts into one exterior polygon
        # Take the exterior of the largest polygon
        largest = max(merged.geoms, key=lambda g: g.area)
        exterior_poly = Polygon(largest.exterior.coords)
    else:
        # If it collapsed to a LineString or Point, we cannot form a polygon
        raise ValueError(f"Resulting geometry is of type {type(merged)}; cannot extract polygon.")

    # Simplify
    tolerance = initial_tolerance
    simplified = exterior_poly.simplify(tolerance, preserve_topology=True)
    while len(simplified.exterior.coords) > max_points:
        tolerance *= scale_factor
        simplified = exterior_poly.simplify(tolerance, preserve_topology=True)

    return simplified


def merge_to_single_exterior(geometry) -> Polygon:
    """
    Merge any Polygon or MultiPolygon into a single polygon
    using only the outer boundary, discarding holes.

    Inputs:
        geometry: a shapely geometry
    Outputs:
        merged (Polygon): a single shapely polygon
    """
    merged = unary_union(geometry)

    # Handle MultiPolygon
    if isinstance(merged, MultiPolygon):
        # Merge into one polygon using the exterior of the largest polygon
        largest = max(merged.geoms, key=lambda g: g.area)
        merged = Polygon(largest.exterior.coords)
    elif isinstance(merged, Polygon):
        merged = Polygon(merged.exterior.coords)
    else:
        raise ValueError(f"Cannot extract polygon from {type(merged)}")

    return merged


def derive_mair_polygon(l3_mosaic_file: str, simplify_npoints: Optional[int] = None) -> Polygon:
    """
    Derive an exterior polygon from a MethaneAIR L3 mosaic file

    Inputs:
        l3_mosaic_file (str): full path to a L3 mosaic file
        simplify_npoints (Optional[int]): generate polygons with less than this many many
    Outputs:
        merged (Polygon): polygon that represents the given L3 file
    """

    with Dataset(l3_mosaic_file) as l3:
        xch4 = l3["xch4"][:].filled(np.nan)
        lon = l3["lon"][:]
        lat = l3["lat"][:]

    # Grid resolution
    dlon = lon[1] - lon[0]  # step in longitude
    dlat = lat[1] - lat[0]  # step in latitude (often negative)

    # Affine transform
    transform = Affine.translation(lon[0] - dlon / 2, lat[0] - dlat / 2) * Affine.scale(dlon, dlat)

    mask = ~np.isnan(xch4)

    # shapes() yields geometry + value
    geoms = []
    for geom, v in features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        geoms.append(shape(geom))

    # merge polygons into one (holes preserved)
    polygon = unary_union(geoms)

    merged = merge_to_single_exterior(polygon)

    if simplify_npoints is not None:
        merged = simplify_to_exterior_safe(merged, simplify_npoints)

    return merged


def mair_polygons(
    l3_mosaic_list: str,
    output_file: str = "mair_polygons.geojson",
    simplify_npoints: Optional[int] = None,
    use_mount: bool = False,
    min_rotated_rectangle: bool = False,
    update_existing: bool = False,
):
    """
    Generate a GeoJSON file with polygons for the mapping areas of the MethaneAIR flights
    listed in the l3_mosaic_list file

    Inputs:
        l3_mosaic_list (str): file listing paths to MethaneAIR L3 mosaic files
        output_file (str): output geojson file
        simplify_npoints (Optional[int]): generate polygons with less than this many many
        use_mount (bool): if True, convert gs:// paths to local paths
        min_rotated_rectangle (bool): if True, save the minimum rotated rectangles that matches the flight polygons
        updated_existing (bool): if True, update an existing geojson file and only compute polygons for new flights and PIDs
    """
    update_existing = update_existing and Path(output_file).exists()

    if update_existing:
        gdf = gpd.read_file(output_file)
        n_replaced = 0

    td = get_target_dict(l3_mosaic_list)

    gd = {
        "campaign": [],
        "flight": [],
        "area": [],
        "file": [],
        "pid": [],
        "geometry": [],
        "Name": [],
    }
    for c in td:
        print(c)
        for f in td[c]:
            print("\t", f)
            for p in td[c][f]:
                if update_existing and p in gdf["pid"].values:
                    print(f"Skipping {c} {f} {p}")
                    continue
                if update_existing and f in gdf["flight"].values:
                    old_p = gdf.loc[gdf["flight"] == f, "pid"].values[0]
                    if PONumber(p) > PONumber(old_p):
                        gdf = gdf[gdf["pid"] != old_p]
                    print(f"Replacing {c} {f} {old_p} with {p}")
                    n_replaced += 1
                areas = td[c][f][p].keys()
                has_map = any("priority-map" in i for i in areas)
                has_target = any("priority-target" in i for i in areas)
                area_key = "priority-map" if has_map else "priority-target"
                if not (has_map or has_target):
                    area_key = list(td[c][f][p].keys())[0]
                for a in td[c][f][p]:
                    if area_key in a:
                        if use_mount:
                            td[c][f][p][a] = td[c][f][p][a].replace("gs://", "/mnt/gcs/")
                        gd["campaign"] += [c]
                        gd["flight"] += [f]
                        gd["area"] += [a]
                        gd["pid"] += [p]
                        gd["file"] += [td[c][f][p][a]]
                        gd["Name"] += [f"{c}_{f}_{p}_{a}"]
                        gd["geometry"] += [
                            derive_mair_polygon(td[c][f][p][a], simplify_npoints=simplify_npoints)
                        ]
    if not update_existing:
        gdf = gpd.GeoDataFrame(gd, crs="EPSG:4326")
        if min_rotated_rectangle:
            gdf["geometry"] = gdf["geometry"].apply(lambda x: x.minimum_rotated_rectangle)
    else:
        n_new = len(gd["campaign"])
        print(f"Replaced {n_replaced} existing flights with higher PIDs")
        print(f"Added {n_new-n_replaced} new flights")
        if n_new > 0:
            gdf2 = gpd.GeoDataFrame(gd, crs="EPSG:4326")
            if min_rotated_rectangle:
                gdf2["geometry"] = gdf2["geometry"].apply(lambda x: x.minimum_rotated_rectangle)
            gdf = gpd.GeoDataFrame(pd.concat([gdf, gdf2], ignore_index=True), crs=gdf.crs)

    gdf.to_file(output_file, driver="GeoJSON")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a geojson file with polygons corresponding to the MAIR priority maps"
    )
    parser.add_argument("l3_file_list", help="full path to file listing MAIR L3 mosaic gs files")
    parser.add_argument(
        "-n",
        "--simplify-npoints",
        default=None,
        type=int,
        help="generate polygons with that many vertices",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="mair_polygons.geojson",
        help="full path to output geojson file",
    )
    parser.add_argument(
        "--use-mount",
        action="store_true",
        help="if given use local mount for gcs paths",
    )
    parser.add_argument(
        "-r",
        "--min-rotated-rectangle",
        action="store_true",
        help="if given, make tight-fitting rectangles for the polygons",
    )
    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="if given, update existing file by only regenerating new PIDs",
    )
    args = parser.parse_args()

    mair_polygons(
        args.l3_file_list,
        args.output_file,
        args.simplify_npoints,
        args.use_mount,
        args.min_rotated_rectangle,
        args.update_existing,
    )


if __name__ == "__main__":
    main()
