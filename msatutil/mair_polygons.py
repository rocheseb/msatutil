import argparse
from typing import Callable, Optional, Union

import numpy as np
from affine import Affine
from netCDF4 import Dataset
from rasterio import features
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union
import geopandas as gpd

from msatutil.mair_targets import get_target_dict


def simplify_to_exterior_safe(
    geometry, max_points: int = 500, initial_tolerance: float = 1e-5, scale_factor: float = 1.5
) -> Polygon:
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
    Derive an exterior polygon from a MAIR L3 mosaic file
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
):

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
                areas = td[c][f][p].keys()
                has_map = any(["priority-map" in i for i in areas])
                area_key = "priority-map" if has_map else "priority-target"
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
    gdf = gpd.GeoDataFrame(gd, crs="EPSG:4326")

    if min_rotated_rectangle:
        gdf["geometry"] = gdf["geometry"].apply(lambda x: x.minimum_rotated_rectangle)

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
    args = parser.parse_args()

    mair_polygons(
        args.l3_file_list,
        args.output_file,
        args.simplify_npoints,
        args.use_mount,
        args.min_rotated_rectangle,
    )


if __name__ == "__main__":
    main()
