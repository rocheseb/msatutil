import argparse
import os
import traceback
import warnings

import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, shape

from msatutil.mair_ls import mair_ls_serial
from msatutil.msat_dset import msat_dset

# Use pyogrio for reading/writing large shapefiles
engine = "pyogrio"


# Ignore UserWarning raised by google.auth._default
warnings.filterwarnings(
    "ignore",
    message="Your application has authenticated using end user credentials from Google Cloud SDK without a quota project.",
    category=UserWarning,
    module="google.auth._default",
)

warnings.filterwarnings(
    "ignore",
    message="Geometry is in a geographic CRS. Results from 'buffer' are likely incorrect. Use 'GeoSeries.to_crs()' to re-project geometries to a projected CRS before this operation.",
    category=UserWarning,
)


def simplifyWithCatch(multipolygon, simplify):
    """
    Simplifies a Polygon MultiPolygon geometry with a specified tolerance, attempting a larger tolerance upon failure.

    This function attempts to simplify a shapely MultiPolygon geometry using a given tolerance value.
    If the simplification process fails, it retries with a larger tolerance before giving up.

    Parameters
    ----------
    multipolygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The MultiPolygon geometry to be simplified.
    simplify : float
        The initial tolerance for simplification. This value determines how much the geometry is simplified;
        larger values result in more significant simplification.

    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The simplified MultiPolygon geometry. If the simplification fails with the initial and secondary larger tolerance,
        the original geometry is returned without simplification.

    Examples
    --------
    >>> from shapely.geometry import MultiPolygon
    >>> multipolygon = MultiPolygon([Polygon([(0, 0), (1, 1), (1, 0)])])
    >>> simplified_multipolygon = simplifyWithCatch(multipolygon, 0.01)
    >>> print(simplified_multipolygon)
    MULTIPOLYGON (((0 0, 1 1, 1 0, 0 0)))
    """

    # Simplify the geometry, but Sometimes the simplify operation can return an error
    try:
        multipolygon = multipolygon.simplify(simplify, preserve_topology=False)
    except:
        # Try again with a larger tolerance, then give up if it fails
        try:
            multipolygon = multipolygon.simplify(simplify * 3, preserve_topology=False)
        except:
            pass
        pass
    return multipolygon


def level2Netcdf2Geom(ds, decimal_rounding=2, simplify=None):
    """
    Extracts latitude and longitude from an xarray dataset with MSAT level2 structure and creates a shapely geometry.

    This function processes an xarray dataset containing methane observations and geolocation information
    to generate a geopandas GeoDataFrame. Each valid methane observation (XCH4) is represented as a point
    in the GeoDataFrame. Because L2 products are not gridded, data is first binned to grid cells, which
    are converted to geometries. The points are then buffered by the resolution given by the rounding
    amount to create an outline, which is then dissolved to form a single geometry representing the area
    of valid data.

    Modified from Nick Balasus: https://github.com/nicholasbalasus/write_blended_files/blob/main/resources/aws_tutorial_1.ipynb

    Parameters
    ----------
    ds : xarray.Dataset
        An xarray Dataset containing 'xch4_co2proxy' data variable within the 'product' group and
        'latitude' and 'longitude' within the 'geolocation' group.
    decimal_rounding : int, optional
        The number of decimal places to round the buffer resolution for creating point geometries, by default 2.
    simplify : float, optional
        Polygon simplification in map units (probably lat/lon) after decimal rounding. Suggest setting to a value > 10**(-{decimal_rounding}) (default: None)


    Returns
    -------
    shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        A single geometry object representing the area of valid data.

    Examples
    --------
    >>> ds = xr.open_dataset("path_to_netcdf_file.nc")
    >>> outline_geometry = level2Netcdf2gdf(ds, decimal_rounding=2)
    >>> type(outline_geometry)
    <class 'shapely.geometry.polygon.Polygon'> # This could also be shapely.geometry.multipolygon.MultiPolygon depending on the data
    """
    product_str = "product"
    try:
        ds[product_str]
    except:
        product_str = "product_co2proxy"  # old L2 version group name
    finally:
        data_variable = ds[product_str]["xch4_co2proxy"][:]
    valid_data_mask = ~np.isnan(data_variable).mask
    res = 10 ** (-decimal_rounding)

    # Make a DataFrame for this file
    gdf = gpd.GeoDataFrame(
        {"lat": ds["geolocation"]["latitude"][:][valid_data_mask]},
        geometry=gpd.points_from_xy(
            ds["geolocation"]["longitude"][:][valid_data_mask],
            ds["geolocation"]["latitude"][:][valid_data_mask],
        ),
        # crs="EPSG:4326"
    )
    outline = gpd.GeoDataFrame(geometry=gdf.buffer(res)).dissolve()
    if len(outline) == 0:  # empty bc no data at all
        return None
    else:
        multipolygon = outline["geometry"].values[0]
        if simplify is not None:
            multipolygon = simplifyWithCatch(multipolygon, simplify)
        return multipolygon


def validDataArea2Geom(ds, simplify=None):
    """
    This function converts valid data areas from a msatutil.msat_dset.msat_dset (Netcdf4 Dataset) into a MultiPolygon Shapely geometry.

    Geometries are derived from the rioxarray affine transformation to the gridded data. It takes about an hour to run for 1,000 mosaics.
    See https://github.com/methanesat-org/sandbox-viz-app/blob/main/src/make_geotiff_L3.py for a similar script that converts netcdf to geotiff.

    Parameters:
    - ds: Netcdf4 Dataset object containing the data
    - simplify: Polygon simplification in map units (probably lat/lon) (default: None)

    Returns:
    - multipolygon: A Polygon or MultiPolygon object representing the valid data areas

    Notes:
    - The function assumes a specific geospatial resolution of 1/3 or 1arcseconds and returns an assertion error if this is not true.
    - The function assumes geographic coordinates in WGS84 (EPSG:4326).
    - The function assumes that the dataset has a 'xch4' variable that contains the valid data areas.
    - It creates a MultiPolygon object from valid data areas based on the 'xch4' variable in the dataset.
    - The function uses a transformation defined by the geospatial information of the dataset.
    - There were some errors using a simplify_tol of 0.001 degrees. Using 0.01 instead is slightly too coarse, but saves disk space and makes rendering quick.

    Examples:
    validDataArea2Gdf(ds, simplify=None)
    """

    # Define the transform and metadata for the temporary raster
    if ds.geospatial_lat_resolution == ds.geospatial_lon_resolution == "1/3 arcseconds":
        res = 1 / 60 / 60 / 3
    elif ds.geospatial_lat_resolution == ds.geospatial_lon_resolution == "1 arcsecond":
        res = 1 / 60 / 60
    else:
        raise ValueError(
            "Geospatial resolutions of latitude and longitude do not match or value needs to be specified"
        )
    transform = from_origin(float(ds.geospatial_lon_min), float(ds.geospatial_lat_min), res, -res)
    data_variable = ds.variables["xch4"][:]
    valid_data_mask = ~np.isnan(data_variable)

    # Convert to geometry
    shapes_gen = shapes((~valid_data_mask.mask).astype("uint8"), transform=transform)
    polygons = []
    for poly_shape, value in shapes_gen:
        if value == 1:  # Valid data value
            polygons.append(shape(poly_shape))

    multipolygon = MultiPolygon(polygons)

    if simplify is not None:
        multipolygon = simplifyWithCatch(multipolygon, simplify)
    return multipolygon


def save_geojson(
    catalogue_pth: str,
    working_dir: str,
    load_from_chkpt: bool = True,
    simplify_tol: float = None,
    save_frequency: int = 50,
    out_path: str = None,
    l2_data=False,
    decimal_rounding=2,
    latest=False,
    resolution="30m",
    type="mosaic",
    uri=None,
    **kwargs,
) -> list[str]:
    """
    save_geojson Calls validDataArea2Gdf and writes out as an ESRI shapefile and a geojson (if polygon simplification is used).

    Requires user to be authenticated to GCS in order to load cloud paths. kwargs are passed to mair_ls.

    Parameters
    ----------
    catalogue_pth : str
        Local or cloud path to a csv file with a column 'uri' listing cloud paths to L2 or L3 .nc files. Other attributes are copied over to final output. If using a cloud path, requires gcsfs to be installed.
    working_dir : str
        Output directory to save final spatial file
    load_from_chkpt : bool, optional
        Whether to load from an existing (potentially partly complete) output, based on name, or from `catalogue_pth`, by default True
    simplify_tol : float, optional
        If given, uses polygon simplification of `simplify_tol` map units to reduce output file size and rendering times.
        For L2 data, suggest setting to a value > 10**(-{decimal_rounding}). by default None
    save_frequency : int, optional
        Enables intermediate saving every `save_frequency` files. Set to a high number to disable. B default 50
    out_path : str, optional
        Output path (extensions will be replaced), by default uses basename of `catalogue_pth`
    l2_data : bool, optional (False)
        Whether filepaths are for L2 data.
    decimal_rounding : int, optional (2)
        Number of lat/long decimal places to use for binning L2 data. Default is 2 (ignored if l2_data is False).
    latest : bool, optional (False)
        If true, passes `latest=True` arg to mair_ls.py in order to only process the most recent versions
    resolution : str, optional ('30m')
        Ignored unless latest == True (arg for mair_ls). File resolution to filter for (30m or 10m).
    type : str, optional ('mosaic')
        Ignored unless latest == True (arg for mair_ls). File data type (mosaic or regrid)
    uri : str, optional
        Pattern to look for in the google storage path
    Returns
    -------
    list[str]
        A list of file paths for the written files. The list contains paths to the ESRI shapefile and geojson file
        if polygon simplification was applied.

    Examples
    --------
    >>> save_geojson('path/to/catalogue.csv', 'path/to/output/directory')
    ['path/to/output/directory/outputfile.shp', 'path/to/output/directory/outputfile.geojson']
    ```

    """
    # Try-except block allows function to return output path names during testing, even if there is a keyboard interrupt.
    try:
        # Load
        if simplify_tol is not None:
            tol_str = f"_tol{simplify_tol}"
        else:
            tol_str = ""
        if out_path is None:
            catalogue_shp_out_basename = os.path.basename(catalogue_pth).replace(".csv", "")
        else:
            catalogue_shp_out_basename = os.path.splitext(out_path)[0]
        catalogue_shp_out_pth = os.path.join(
            working_dir, f"{catalogue_shp_out_basename}{tol_str}.shp"
        )
        catalogue_geojson_out_pth = catalogue_shp_out_pth.replace(".shp", ".geojson")
        if load_from_chkpt and os.path.exists(os.path.expanduser(catalogue_shp_out_pth)):
            # Assumes if loading from .shp, data is already filtered for latest, if desired
            df = gpd.read_file(catalogue_shp_out_pth, engine=engine)
        else:
            print("Computing latest version for each flight product...\n")
            df = mair_ls_serial(
                catalogue_pth, latest, resolution=resolution, type=type, uri=uri, **kwargs
            )
            print(df)
            print("\n")
        # Loop
        nrows = len(df)
        for index, row in df.iterrows():
            gs_pth = row["uri"]
            if "geometry" in df.columns:  # loaded from checkpoint
                if pd.notnull(df.at[index, "geometry"]):
                    print(f"Geometry exists for {gs_pth.split('/mosaic/')[-1]}")
                    continue  # Skip if geometry is already present

            print(f"[{index} / {nrows}] {gs_pth}")

            fs = fsspec.filesystem("gcs")
            if fs.exists(gs_pth):
                ds = msat_dset(gs_pth)
            else:
                print(f"GCS object no longer exists: {gs_pth}")
                continue
            if l2_data:
                geom = level2Netcdf2Geom(
                    ds, decimal_rounding=decimal_rounding, simplify=simplify_tol
                )
            else:
                geom = validDataArea2Geom(ds, simplify=simplify_tol)
            df.at[index, "geometry"] = geom

            # Save intermittently
            if (index % save_frequency == 0) or (index == len(df) - 1):
                if (index % save_frequency == 0) and (index != len(df) - 1):
                    print("\t> Saving checkpoint.")
                if index == len(df) - 1:
                    print("\t> Saving final.")
                gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

                # ESRI shapefile can't handle datetimeformat
                if not l2_data:
                    try:
                        for col in [
                            "flight_date",
                            "production_timestamp",
                            "time_start",
                            "time_end",
                        ]:
                            gdf[col] = gdf[col].astype(str)
                    except:
                        for col in ["flight_dat", "producti_2", "time_start", "time_end"]:
                            gdf[col] = gdf[col].astype(str)

                # Save as shapefile and geojson to disk
                gdf.to_file(catalogue_shp_out_pth, engine=engine)
                if (simplify_tol is not None) or l2_data:
                    gdf.to_file(catalogue_geojson_out_pth)
                else:
                    catalogue_geojson_out_pth = ""
        print("Finished creating mask shapefile.")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        traceback_details = traceback.format_exc()
        print(f"An exception occurred: {traceback_details}")
    finally:
        return catalogue_shp_out_pth, catalogue_geojson_out_pth


def main():
    parser = argparse.ArgumentParser(
        description="Calls validDataArea2Gdf and writes out as an ESRI shapefile (if no polygon simplification), or a geojson otherwise. Requires user to be authenticated to GCS in order to load cloud paths.",
        epilog="""
        Examples
        --------
        python msatutil/make_spatial_index.py -c gs://path/to/L3.csv -w /tmp -s 0.003 -a -t mosaic
        python msatutil/make_spatial_index.py -c gs://path/to/L3.csv -w /tmp -s 0.003 -a -t regrid -e 30m -u priority-target
        
        Dataset paths:
        --------
        L2pp: gs://path/to/L2_pp.csv
        L3 segments/mosaics: gs://path/to/L3.csv

        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Required arguments with shortcuts
    parser.add_argument(
        "-c",
        "--catalogue_pth",
        type=str,
        required=True,
        help="Local or cloud path to a csv file listing cloud paths to L2 or L3 .nc files. Requires gcsfs for cloud paths.",
    )
    parser.add_argument(
        "-w",
        "--working_dir",
        type=str,
        required=True,
        help="Output directory to save the final spatial file.",
    )

    # Optional arguments with shortcuts
    parser.add_argument(
        "-l",
        "--no-load_from_chkpt",
        action="store_true",
        help="Don't load from an existing output or from the catalogue path. Default: False (loads from checkpoint).",
    )
    parser.add_argument(
        "-s",
        "--simplify_tol",
        type=float,
        default=None,
        help="Polygon simplification tolerance in map units to reduce file size and rendering times. For L2 data, suggest setting to a value > 10**(-{decimal_rounding}). Default: None.",
    )
    parser.add_argument(
        "-f",
        "--save_frequency",
        type=int,
        default=50,
        help="Intermediate saving frequency every 'n' files. High number disables it. Default: 50.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default=None,
        help="Output path for the file, with extensions replaced. Default: Basename of catalogue_pth.",
    )
    parser.add_argument(
        "-d", "--l2_data", action="store_true", help="Whether filepaths are for L2 data."
    )
    parser.add_argument(
        "-r",
        "--decimal_rounding",
        type=int,
        default=2,
        help="Number of lat/long decimal places to use for binning L2 data. (ignored if l2_data is False)",
    )
    parser.add_argument(
        "-a",
        "--latest",
        action="store_true",
        help="Only process the most recent product versions based on mair_ls.py",
    )
    parser.add_argument(
        "-e",
        "--resolution",
        type=str,
        default="30m",
        help="Ignored unless latest == True (arg for mair_ls). File resolution to filter for (30m or 10m).",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="mosaic",
        help="Ignored unless latest == True (arg for mair_ls). File data type (mosaic or regrid)",
    )
    parser.add_argument(
        "-u", "--uri", type=str, default=None, help="Pattern to look for in the google storage path"
    )
    args = parser.parse_args()

    # Call the function with the parsed arguments
    save_geojson(
        catalogue_pth=args.catalogue_pth,
        working_dir=args.working_dir,
        load_from_chkpt=not args.no_load_from_chkpt,
        simplify_tol=args.simplify_tol,
        save_frequency=args.save_frequency,
        out_path=args.out_path,
        l2_data=args.l2_data,
        decimal_rounding=args.decimal_rounding,
        latest=args.latest,
        resolution=args.resolution,
        type=args.type,
        uri=args.uri,
    )


if __name__ == "__main__":
    main()
