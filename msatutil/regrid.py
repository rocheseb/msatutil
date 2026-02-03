from datashader import Canvas
import xarray as xr
import numpy as np
from pyproj import Transformer


def get_pixel_dims(
    bbox: tuple[float, float, float, float], width_resolution: float, height_resolution: float
):
    """
    Convert a desired pixel resolution in meters to a number of pixels for the Geoviews raster.
    The raster will then have the given resolution but with vertical pixels.
    So here width_resolution and height_resolution are really longitude_resolution and
    latitude_resolution (in meters).

    Inputs:
        bbox (tuple[float, float, float, float]): [minlon,minlat,maxlon,maxlat]
        width_resolution (float): desired pixel width (meters)
        height_resolution (float): desired pixel height (meters)

    Outputs:
        width_pixels (int): number of pixels for the plot width
        height_pixels (int): number of pixels for the plot height
    """

    # Define the CRS for transformation
    wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    # Transform bounding box to meters
    min_x, min_y = wgs84_to_mercator.transform(bbox[0], bbox[1])
    max_x, max_y = wgs84_to_mercator.transform(bbox[2], bbox[3])

    # Calculate the number of pixels required for desired resolution
    width_pixels = int((max_x - min_x) / width_resolution)
    height_pixels = int((max_y - min_y) / height_resolution)

    return width_pixels, height_pixels


def make_canvas_and_grid(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    pixel_resolution: tuple[float, float],  # (dx_m, dy_m)
) -> tuple[Canvas, np.ndarray, np.ndarray]:
    """
    Create a Datashader Canvas and corresponding lon/lat grid.

    Inputs:
        lon_min (float): minimum longitude of the output grid
        lon_max (float): maximum longitude of the output grid
        lat_min (float): minimum latitude of the output grid
        lat_max (float): maximum latitude of the output grid
        pixel_resolution (tuple[float,float]): desired pixel (width,height) in meters

    Outputs:
        canvas (ds.Canvas): Datashader Canvas configured for the given domain and resolution
        lon_grid (np.ndarray): 2D array of longitudes for the output grid
        lat_grid (np.ndarray): 2D array of latitudes for the output grid
    """
    width_pixels, height_pixels = get_pixel_dims(
        [lon_min, lat_min, lon_max, lat_max],
        pixel_resolution[0],
        pixel_resolution[1],
    )

    canvas = Canvas(
        plot_width=width_pixels,
        plot_height=height_pixels,
        x_range=(lon_min, lon_max),
        y_range=(lat_min, lat_max),
    )

    lon = np.linspace(lon_min, lon_max, width_pixels)
    lat = np.linspace(lat_min, lat_max, height_pixels)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    return canvas, lon_grid, lat_grid


def regrid_to_canvas(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    canvas: Canvas,
) -> np.ndarray:
    """
    Regrid the input field onto an existing Datashader Canvas.

    Inputs:
        x (np.ndarray): 1D or 2D array of longitudes (shape (N,) or (M,N))
        y (np.ndarray): 1D or 2D array of latitudes (shape (N,) or (M,N))
        z (np.ndarray): 2D array of the data to regrid (shape (M,N))
        canvas (ds.Canvas): Datashader Canvas defining the target lon/lat grid

    Outputs:
        z_grid (np.ndarray): 2D array of z regridded onto the canvas grid
    """
    da = xr.DataArray(
        z,
        name="Z",
        dims=["y", "x"],
        coords={
            "Longitude": (["y", "x"], x),
            "Latitude": (["y", "x"], y),
        },
    )
    quadmesh = canvas.quadmesh(da, x="Longitude", y="Latitude")
    return quadmesh.values


class Regridder:
    def __init__(self, lon_min, lon_max, lat_min, lat_max, pixel_resolution):
        self.canvas, self.lon_grid, self.lat_grid = make_canvas_and_grid(
            lon_min, lon_max, lat_min, lat_max, pixel_resolution
        )

    def __call__(self, x, y, z):
        z_grid = regrid_to_canvas(x, y, z, self.canvas)
        return self.lon_grid, self.lat_grid, z_grid
