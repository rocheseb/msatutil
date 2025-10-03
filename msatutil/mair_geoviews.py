import warnings

warnings.simplefilter("ignore")
import argparse
import inspect
import os
import subprocess
from typing import List, Optional, Tuple, Union

import bokeh
import geoviews as gv
import holoviews as hv
import numpy as np
import panel as pn
from bokeh.embed import file_html
from bokeh.models import (
    ColorBar,
    Column,
    CrosshairTool,
    CustomJS,
    Div,
    NumericInput,
    Row,
    Select,
    Slider,
    Span,
)
from bokeh.resources import CDN
from geoviews.element import WMTS
from geoviews.tile_sources import EsriImagery
from holoviews.operation.datashader import rasterize
from holoviews.plotting import list_cmaps
from holoviews.plotting.util import process_cmap
from pyproj import Transformer

from msatutil.mair_ls import create_parser as create_ls_parser
from msatutil.mair_ls import mair_ls
from msatutil.msat_dset import gs_list, msat_dset
from msatutil.msat_interface import get_msat

hv.extension("bokeh")


CONTEXT_L1_VARIABLES = [
    "Geolocation/SurfaceAltitude",
    "Geolocation/ViewingZenithAngle",
]

CONTEXT_L2_VARIABLES = [
    "Level1/SurfaceAltitude",
    "Level1/ViewingZenithAngle",
    "OptProp_Band1/RefWvl_BRDF_KernelAmplitude_isotr",
]

CONTEXT_L2PP_VARIABLES = [
    "co2proxy_fit_diagnostics/retrieved_albedo_1606nm",
    "o2dp_fit_diagnostics/bias_corrected_delta_pressure",
    "apriori_data/surface_pressure",
    "product_co2proxy/main_quality_flag",
]

CONTEXT_L3_VARIABLES = [
    "apriori_data/albedo_ch4band",
    "geolocation/terrain_height",
    "o2dp_fit_diagnostics/bias_corrected_delta_pressure",
    "apriori_data/zonal_wind",
    "apriori_data/meridional_wind",
    "num_samples",
]

CONTEXT_VARIABLES_DICT = {
    "l1": CONTEXT_L1_VARIABLES,
    "l2": CONTEXT_L2_VARIABLES,
    "l2pp": CONTEXT_L2PP_VARIABLES,
    "l3": CONTEXT_L3_VARIABLES,
}


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


def regrid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    pixel_resolution: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Regrid the input data to roughly the given resolution (in meters)

    Inputs:
        x: 1D or 2D array of longitudes (shape (N,) or (M,N))
        y: 1D or 2D array of latitudes (shape (M,) or (M,N))
        z: 2D array of the data to plot (shape (M,N))
    Outputs:
        tuple[np.ndarray, np.ndarray, np.ndarray]: output lon, lat, and regridded field
    """
    quad = gv.project(gv.QuadMesh((x, y, z)))
    width_pixels, height_pixels = get_pixel_dims(
        [np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y)],
        pixel_resolution[0],
        pixel_resolution[1],
    )
    raster = rasterize(quad, width=width_pixels, height=height_pixels, precompute=True)
    data = raster[()].data

    return data["Longitude"].values, data["Latitude"].values, data["z"].values


def set_clim(z: np.ndarray, n_std: int = 3) -> tuple[float, float]:
    """
    Define color limits as median +/- n_std standard deviations.
    Estimate std from IQR to eliminate outliers.

    Inputs:
        z (np.ndarray): input data
    Outputs:
        clim (tuple[float,float]): output limits
    """
    z = np.ma.MaskedArray(z).filled(np.nan)
    med_z = np.nanmedian(z)
    q25, q75 = np.nanpercentile(z, [25, 75])
    std_z = 0.74 * (q75 - q25)
    clim = (med_z - n_std * std_z, med_z + n_std * std_z)

    return clim


def show_map(
    x,
    y,
    z,
    width: int = 550,
    height: int = 450,
    cmap: str = "viridis",
    clim: Optional[Union[Tuple[float, float], bool]] = None,
    alpha: int = 1,
    title: str = "",
    background_tile_list=[EsriImagery],
    single_panel: bool = False,
    pixel_ratio: int = 1,
    active_tools: list[str] = ["pan", "wheel_zoom"],
    tools: list[str] = ["fullscreen"],
    pixel_resolution: Optional[tuple[float, float]] = None,
    clipping_colors: dict = {"NaN": (0, 0, 0, 0), "min": None, "max": None},
    **opts,
):
    """
    Make a geoviews map of z overlayed on background_tile
    This doesn't preserve pixel shapes as the image is re-rasterized after each zoom/pan

    Inputs:
        x: 1D or 2D array of longitudes (shape (N,) or (M,N))
        y: 1D or 2D array of latitudes (shape (M,) or (M,N))
        z: 2D array of the data to plot (shape (M,N))
        width (int): plot width in pixels
        height (int): plot height in pixels
        cmap (str): named colormap
        clim (Optional[Union[Tuple[float, float],bool]]): z-limits for the colorbar, give False to use dynamic colorbar
        alpha (float): between 0 and 1, sets the opacity of the plotted z field
        title (str): plot title
        background_tile_list: the geoviews tile the plot will be made over and that will be in the linked 2nd panel
                         if None only makes one panel with no background but with the save tool active
        single_panel (bool): if True, do not add the linked panel with only esri imagery
        pixel_ratio (int): the initial map (and the static maps) will have width x height pixels, this multiplies the number of pixels
        active_tools (list[str]): Active map tools for mouse use (default: ['pan', 'wheel_zoom'])
        tools (list[str]): Additional map tools (default: [], suggested: ['hover']). An apparent bug makes hover active if specified
        pixel_resolution (Optional[tuple[float,float]]): desired pixel (width,height) in meters
        clipping_colors (dict): dictionary for the holoviews colorbar options, (0,0,0,0) is transparent.
            min and max are for below and above the minimum and maximum colorbar limits.
            can use named colors or rgba 4-tuple.
            however there is no nice way to indicate the clipping colors with bokeh, they won't show on the colorbar.

    Outputs:
        geoviews figure
    """

    quad = gv.project(gv.QuadMesh((x, y, z)))

    if clim is None:
        clim = set_clim(z)

    if pixel_resolution is not None:
        width_pixels, height_pixels = get_pixel_dims(
            [np.nanmin(x), np.nanmin(y), np.nanmax(x), np.nanmax(y)],
            pixel_resolution[0],
            pixel_resolution[1],
        )
        pixel_ratio = 1
    else:
        width_pixels = width
        height_pixels = height

    scalebar_args = {
        "scalebar": True,
        "scalebar_range": "x",
        "scalebar_location": "bottom_left",
        "scalebar_opts": {
            "background_fill_alpha": 0.6,
            "border_line_color": None,
            "label_text_font_size": "20px",
            "label_text_color": "black",
            "label_location": "left",
            "length_sizing": "exact",
            "bar_length": 0.2,
            "bar_line_color": "black",
            "bar_line_width": 5,
            "margin": 0,
            "padding": 5,
        },
    }

    raster = rasterize(
        quad, width=width_pixels, height=height_pixels, pixel_ratio=pixel_ratio
    ).opts(
        width=width,
        height=height,
        cmap=cmap,
        colorbar=True,
        title=title,
        alpha=alpha,
        active_tools=active_tools,
        tools=tools,
        clipping_colors=clipping_colors,
        **scalebar_args,
    )

    if clim is not False:
        raster = raster.opts(clim=clim)

    if (background_tile_list is not None) and (not single_panel):
        # Make a dummy quadmesh that will have alpha=0 in the second panel so we can see the EsriImagery under the data
        # I do this so it will add a colorbar on the second plot so we don't need to think about resizing it
        # just use a small subset of data so it doesn't trigger much computations
        if x.ndim == 1:
            xdum = x[:10]
            ydum = y[:10]
        else:
            xdum = x[:10, :10]
            ydum = y[:10, :10]

        dummy = gv.project(gv.QuadMesh((xdum, ydum, z[:10, :10]))).opts(
            width=width,
            height=height,
            cmap=cmap,
            colorbar=True,
            alpha=0,
            active_tools=active_tools,
            tools=tools,
            title=f"Tile source: {background_tile_list[0].name}",
            **scalebar_args,
        )
        if clim is not False:
            dummy.opts(clim=clim)

    if background_tile_list is None:
        plot = raster
    elif single_panel:
        plot = background_tile_list[-1] * raster
    else:
        plot = (background_tile_list[-1] * raster) + (background_tile_list[0] * dummy)

    return plot


def save_static_plot_with_widgets(
    out_file: str,
    plot,
    alpha: float = 1.0,
    cmap: str = "viridis",
    browser_tab_title: str = "MethaneSAT",
    layout_title: str = "",
    layout_details: str = "",
    linked_colorbars: bool = False,
    exclude_links: list[str] = [],
    high_color: Optional[str] = None,
    low_color: Optional[str] = None,
) -> None:
    """
    Save the output of show_map to a html file

    transform into a bokeh object
    add numeric inputs for the first colorbar limits
    add an alpha slider
    add a colormap selector
    save as an html file

    Inputs:
        out_file (str): full path to output html file
        plot: holoviews layout object
        alpha (float): initial heatmap alpha
        cmap (str): initial colormap
        browser_tab_title (str): the title of the tab in the browser
        layout_title (str): text for a title Div above the layout
        layout_details (str): more info to include below the title
        linked_colorbars (bool): if True, all colorbars will be updated by the colorbar limits inputs
        exclude_links (list[str]): when linked_colorbars is True, this lists plot title to exclude
        high_color (Optional[str]): if given, set as the color above the max of the colorbar
        low_color (Optional[str]): if given, set as the color below the min of the colorbar
    """

    # Convert the holoviews layout object to a bokeh object
    bokeh_plot = hv.render(plot, backend="bokeh")
    bokeh_plot.sizing_mode = "scale_both"

    if type(bokeh_plot) is bokeh.models.plots.GridPlot:
        # Layout of multiple figures
        colorbar_list = [
            i[0].select_one(ColorBar)
            for i in bokeh_plot.children
            if "Tile source" not in i[0].title.text
            and not any([j.lower() in i[0].title.text.lower() for j in exclude_links])
        ]
        glyph_list = [
            fig[0].renderers[1].glyph
            for fig in bokeh_plot.children
            if type(fig[0].renderers[1].glyph) is bokeh.models.glyphs.Image
        ]
        imagery_panel = np.where(
            [fig[0].title.text.startswith("Tile source") for fig in bokeh_plot.children]
        )[0]
        if imagery_panel.size > 0:
            # when there is a panel with only imagery
            # remove the colorbar from that panel
            # set fixed borders to all panels so they keep the same dimensions
            for fig in bokeh_plot.children:
                fig[0].min_border_right = 100
            bokeh_plot.children[imagery_panel[0]][0].right[0].visible = False
        # add linked crosshairs
        width = Span(dimension="width", line_dash="dashed")
        height = Span(dimension="height", line_dash="dashed")
        for fig in bokeh_plot.children:
            fig[0].add_tools(CrosshairTool(overlay=[width, height]))
        # make each subplot toolbar visible
        for plot in bokeh_plot.children:
            plot[0].sizing_mode = "scale_both"
            plot[0].toolbar_location = "right"
            plot[0].toolbar.visible = True
            plot[0].toolbar.logo = None
            for tool in plot[0].tools:
                if isinstance(tool, CrosshairTool):
                    tool.visible = False
    elif type(bokeh_plot) is bokeh.plotting._figure.figure:
        # Single figure
        colorbar_list = [bokeh_plot.select_one(ColorBar)]
        glyph_list = [bokeh_plot.renderers[1].glyph]

    if linked_colorbars:
        for colorbar in colorbar_list:
            if high_color:
                colorbar.color_mapper.high_color = high_color
            if low_color:
                colorbar.color_mapper.low_color = low_color
    else:
        colorbar_list = [colorbar_list[0]]
        if high_color:
            colorbar_list[0].color_mapper.high_color = high_color
        if low_color:
            colorbar_list[0].color_mapper.low_color = low_color

    # Numeric inputs for the first colorbar limits
    callback = CustomJS(
        args={"color_mappers": [i.color_mapper for i in colorbar_list]},
        code="""
            for (let i = 0; i < color_mappers.length; i++) {
                color_mappers[i].low = cb_obj.value;
            }
        """,
    )
    first_colorbar_low = NumericInput(
        value=colorbar_list[0].color_mapper.low, width=100, title="1st colorbar Low", mode="float"
    )
    first_colorbar_low.js_on_change("value", callback)

    callback = CustomJS(
        args={"color_mappers": [i.color_mapper for i in colorbar_list]},
        code="""
            for (let i = 0; i < color_mappers.length; i++) {
                color_mappers[i].high = cb_obj.value;
            }
        """,
    )
    first_colorbar_high = NumericInput(
        value=colorbar_list[0].color_mapper.high, width=100, title="1st colorbar High", mode="float"
    )
    first_colorbar_high.js_on_change("value", callback)

    # Alpha slider
    callback = CustomJS(
        args={"glyphs": glyph_list},
        code="for (var i=0;i<glyphs.length;i++){glyphs[i].global_alpha = cb_obj.value;};",
    )
    alpha_slider = Slider(start=0, end=1.0, step=0.1, value=alpha, title="Heatmap alpha", width=200)
    alpha_slider.js_on_change("value", callback)

    # Color palette select
    uniform_sequential_palettes = list_cmaps(category="Uniform Sequential")
    sequential_palettes = list_cmaps(category="Sequential")
    palette_dict = {k.lower() + " (u)": process_cmap(k) for k in uniform_sequential_palettes}
    for k in set(sequential_palettes).difference(uniform_sequential_palettes):
        palette_dict[k.lower()] = process_cmap(k)
    for k in ["turbo", "jet", "rainbow"]:
        palette_dict[k.lower()] = process_cmap(k)
        k_r = f"{k}_r"
        palette_dict[k_r.lower()] = process_cmap(k_r)

    palette_select = Select(
        options=sorted(list(palette_dict.keys())),
        value=cmap.lower(),
        title="Colormap",
    )
    callback = CustomJS(
        args={"glyphs": glyph_list, "palette_dict": palette_dict},
        code="""for (var i=0;i<glyphs.length;i++){
        glyphs[i].color_mapper.palette = palette_dict[cb_obj.value];
        };""",
    )
    palette_select.js_on_change("value", callback)

    maps_widgets = Row(palette_select, alpha_slider, first_colorbar_low, first_colorbar_high)

    maps_with_widgets = Column(bokeh_plot, maps_widgets)
    maps_with_widgets.sizing_mode = "scale_both"

    flexbox_elements = []
    if layout_title:
        text_settings = "font-family: 'Arial, sans-serif'; font-size: 24px; font-weight: bold;"
        flexbox_elements.append(
            Div(text=f'<p style="{text_settings}">{layout_title}</p>', width=1200, height=30)
        )
    if layout_details:
        text_settings = "font-family: 'Arial, sans-serif'; font-size: 14px;"
        flexbox_elements.append(
            Div(text=f'<p style="{text_settings}">{layout_details}</p>', width=1200, height=20)
        )
    flexbox_elements.append(maps_with_widgets)
    bokeh_layout = Column(*flexbox_elements)
    bokeh_layout.sizing_mode = "scale_both"

    with open(out_file, "w") as out:
        out.write(file_html(bokeh_layout, CDN, browser_tab_title, suppress_callback_warning=True))

    print(out_file)


def set_outfile(in_path: str, out_path: str) -> str:
    """
    Determine the output html file path

    Inputs:
    in_path (str): full path to the input netcdf data file, can be a gs:// path. OR full path to input directory
    out_path (Optional[str]): full path to the output .html file or to a directory where it will be saved.
                              If None, save output html file in the current working directory
    Outputs:
    out_file (str): full path to the output html file
    """
    if in_path.endswith(".nc"):
        default_html_filename = os.path.basename(in_path).replace(".nc", ".html")
    else:
        default_html_filename = os.path.basename(in_path) + ".html"
    if out_path is None:
        out_file = os.path.join(os.getcwd(), default_html_filename)
    elif os.path.splitext(out_path)[1] == "":
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, default_html_filename)
    else:
        if os.path.dirname(out_path) and not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        out_file = out_path

    return out_file


def read_variables(
    in_path: str,
    variables: list[str],
    lon_var: str = "lon",
    lat_var: str = "lat",
    num_samples_threshold: Optional[float] = None,
    option: Optional[str] = None,
    option_axis_dim: str = "spectral_channel",
    apply_flag: Optional[str] = None,
    date_range=None,
    srchstr="*.nc",
):
    var_list = []
    if in_path.endswith(".nc"):
        with msat_dset(in_path) as nc:
            lon = nc[lon_var][:].filled(np.nan)
            lat = nc[lat_var][:].filled(np.nan)
            if num_samples_threshold is not None:
                num_samples = nc["num_samples"][:].filled(np.nan)
                num_samples[num_samples < num_samples_threshold] = np.nan
                nan_num_samples = np.isnan(num_samples)
            for i, var in enumerate(variables):
                v = nc[var][:].astype(float).filled(np.nan)
                if num_samples_threshold is not None:
                    v[nan_num_samples] = np.nan
                if i == 0 and option is not None:
                    v = getattr(np, option)(v, axis=nc[var].dimensions.index(option_axis_dim))
                if i == 0 and apply_flag:
                    assert (
                        nc[apply_flag][:].shape == v.shape
                    ), f"Flag variable {apply_flag} has shape {nc[apply_flag][:].shape}, but first variable {var} has shape {v.shape}"
                    flags = nc[apply_flag][:].astype(float).filled(np.nan)
                    v[flags != 0] = np.nan
                var_list += [v]
            title_list = [
                f"{var} ({nc[var].units})" if hasattr(nc[var], "units") else var
                for var in variables
            ]
    else:
        with get_msat(in_path, date_range=date_range, srchstr=srchstr) as msat_data:
            # make the valid cross track check on the variable to plot
            # if it is as 3D variable make it on the longitude variable
            # sometimes longitude has 1 extra valid cross track on each side
            valid_check_var = variables[0] if not option else lon_var
            msat_data.valid_xtrack = msat_data.get_valid_xtrack(valid_check_var)
            lon = msat_data.pmesh_prep(lon_var, use_valid_xtrack=True).compute()
            lat = msat_data.pmesh_prep(lat_var, use_valid_xtrack=True).compute()
            # read the first variable alone as it can take in an operation
            var_list = [
                msat_data.pmesh_prep(
                    variables[0],
                    use_valid_xtrack=True,
                    option=option,
                    option_axis_dim=option_axis_dim,
                ).compute()
            ]
            if apply_flag:
                flags = msat_data.pmesh_prep(apply_flag, use_valid_xtrack=True).compute()
                var_list[0][flags != 0] = np.nan
            # read the rest of the variables if they exist
            if len(variables) > 1:
                var_list += [
                    msat_data.pmesh_prep(var, use_valid_xtrack=True).compute()
                    for var in variables[1:]
                ]
            nc = msat_data.dsets[msat_data.ids[0]]
            title_list = [
                f"{var} ({nc[var].units})" if hasattr(nc[var], "units") else var
                for var in variables
            ]

    return lon, lat, var_list, title_list


def set_background_tile_list(background_tile_name_list: Optional[list[str]] = None):
    """
    Convert list of background tile names into list of background tile objects

    Inputs:
        background_tile_name_list (list[str]): list of background tile names

    Outputs:
        background_tile_list (list): list of background tiles
    """
    if background_tile_name_list is None:
        background_tile_list = [EsriImagery]
    else:
        GoogleMapsImagery = WMTS(
            "https://mt1.google.com/vt/lyrs=s&x={X}&y={Y}&z={Z}", name="GoogleMapsImagery"
        )
        BingMapsImagery = WMTS(
            "http://ecn.t3.tiles.virtualearth.net/tiles/a{Q}.jpeg?g=1", name="BingMapsImagery"
        )
        tile_dict = {k.lower(): v for k, v in gv.tile_sources.__dict__["tile_sources"].items()}
        tile_dict["googlemapsimagery"] = GoogleMapsImagery
        tile_dict["bingmapsimagery"] = BingMapsImagery
        background_tile_list = ["" for i in background_tile_name_list]
        for i, background_tile_name in enumerate(background_tile_name_list):
            if background_tile_name.lower() not in tile_dict:
                background_tile_list[i] = tile_dict["esriimagery"]
            else:
                background_tile_list[i] = tile_dict[background_tile_name.lower()]

    return background_tile_list


def do_html_plot(
    in_path: str,
    variables: list[str],
    lon_var: str = "lon",
    lat_var: str = "lat",
    out_path: Optional[str] = None,
    title: str = "",
    layout_title: str = "",
    layout_details: str = "",
    browser_tab_title: str = "MethaneAIR map",
    cmap: str = "viridis",
    clim: Optional[Union[Tuple[float, float], bool]] = None,
    width: int = 850,
    height: int = 750,
    alpha: float = 1,
    panel_serve: bool = False,
    single_panel: bool = False,
    background_tile_name_list: List[str] = ["googlemapsimagery"],
    num_samples_threshold: Optional[float] = None,
    pixel_ratio: int = 1,
    add_standalone_imagery: bool = False,
    ncols: int = 3,
    option: Optional[str] = None,
    option_axis_dim: str = "spectral_channel",
    apply_flag: Optional[str] = None,
    pixel_resolution: Optional[tuple[float, float]] = None,
    clipping_colors: dict = {"NaN": (0, 0, 0, 0), "min": None, "max": None},
    custom_data: Optional[dict] = None,
    lat_offset: float = 0,
    lon_offset: float = 0,
    srchstr: str = "*.nc",
) -> None:
    """
    Save a html plot of var from in_path

    in_path (str): full path to the input netcdf data file, can be a gs:// path. OR full path to input directory
    variables (str): list of variable paths in the data file
    lon_var (str): longitude variable path in the data file
    lat_var (str): latitude variable path in the data file
    out_path (Optional[str]): full path to the output .html file or to a directory where it will be saved.
                              If None, save output html file in the current working directory
    title (str): title of the first plot (include field name and units here)
    layout_title (str): overall title that will appear in a Div above the plots
    layout_details (str): normal text between layout_title and the plots
    browser_tab_title (str): text to be displayed in the browser tab
    cmap (str): colormap name
    clim (Optional[Union[Tuple[float, float],bool]]): z-limits for the colorbar, give False to use dynamic colorbar
    width (int): plot width in pixels, the colorbar is ~100 wide so width should be height+100 to make squares
    height (int): plot height in pixels
    panel_serve (bool): if True, start an interactive session
    single_panel (bool): if True, do not add the linked panel with only esri imagery
    background_tile_name_list (Optional[List[str]]): name of the background tile from https://holoviews.org/reference/elements/bokeh/Tiles.html (case insensitive)
                                                    for the main (last value) and linked (first value) panels
    num_samples_threshold (Optional[float]): filter out data with num_samples<num_samples_threshold
    pixel_ratio (int): the initial map (and the static maps) will have width x height pixels, this multiplies the number of pixels
    add_standalone_imagery (bool): if True, add a panel with only the imagery
    ncols (int): number of columns the panels will be arranged in
    option (Optional[str]): numpy operation to apply on the first variable
    option_axis_dim (str): dimension name along which the option will be applied
    apply_flag (Optional[str]): if given, use this flag variable to nan out the data
    pixel_resolution (Optional[tuple[float, float]]): pixel (width,height) in meters
    clipping_colors (dict): dictionary for the holoviews colorbar options, (0,0,0,0) is transparent.
        min and max are for below and above the minimum and maximum colorbar limits.
        can use named colors or rgba 4-tuple.
        however there is no nice way to indicate the clipping colors with bokeh, they won't show on the colorbar.
    custom_data (Optional[dict]): can be given to plot custom fields instead of using in_path and variables
        must have this structure:
        {
            "lon":lon,
            "lat":lat,
            "variables": {
                "name1":var1,
                "name2":var2,
                ...
            }
            "clim": [(start1,end1),(start2,end2),...]
        }
    lat_offset (float): can be given to try shifting the image in latitudes
    lon_offset (float): can be given to try shifting the image in longitudes
    """
    out_file = set_outfile(in_path, out_path)

    if custom_data is not None:
        lon = custom_data["lon"]
        lat = custom_data["lat"]
        var_list = list(custom_data["variables"].values())
        variables = list(custom_data["variables"].keys())
        title_list = [i for i in variables]
        clim_list = [i for i in custom_data["clim"]]
    else:
        lon, lat, var_list, title_list = read_variables(
            in_path,
            variables,
            lon_var=lon_var,
            lat_var=lat_var,
            num_samples_threshold=num_samples_threshold,
            option=option,
            option_axis_dim=option_axis_dim,
            apply_flag=apply_flag,
            srchstr=srchstr,
        )
        clim_list = [None for i in var_list]
        clim_list[0] = clim

    lat += lat_offset
    lon += lon_offset

    if title:
        title_list[0] = title

    background_tile_list = set_background_tile_list(background_tile_name_list)

    # when making multiple subplots, don't add the Imagery panel
    single_panel = len(variables) > 1 or single_panel

    if len(variables) > 1 and (width == 850 or height == 750):
        print(
            "The default width and height (850x750) is too large for multiple panels, reducing to 550x450"
        )
        width = 550
        height = 450

    for i, var in enumerate(variables):
        if i > 0 and "delta_pressure" in var:
            clim_list[i] = (-20, 20)
        elif i > 0 and any([v in var for v in ["num_samples", "flag", "albedo"]]):
            clim_list[i] = (np.nanmin(var_list[i]), np.nanmax(var_list[i]))
        if "albedo_ch4band" in var:
            title_list[i] = "A Priori Surface Albedo at 1606 nm"

    plot_list = [
        show_map(
            lon,
            lat,
            var,
            title=title_list[i],
            cmap=cmap,
            clim=clim_list[i],
            width=width,
            height=height,
            alpha=alpha,
            single_panel=single_panel,
            background_tile_list=background_tile_list,
            pixel_ratio=pixel_ratio,
            pixel_resolution=pixel_resolution,
            clipping_colors=clipping_colors,
        )
        for i, var in enumerate(var_list)
    ]
    if add_standalone_imagery and len(variables) > 1:
        plot_list[0] = show_map(
            lon,
            lat,
            var_list[0],
            title=title_list[0],
            clim=clim_list[0],
            cmap=cmap,
            width=width,
            height=height,
            alpha=alpha,
            background_tile_list=background_tile_list,
            pixel_ratio=pixel_ratio,
            pixel_resolution=pixel_resolution,
        )

    if len(variables) > 1:
        plot = hv.Layout(plot_list).cols(ncols)
    else:
        plot = plot_list[0]

    save_static_plot_with_widgets(
        out_file,
        plot,
        alpha=alpha,
        cmap=cmap,
        layout_title=layout_title,
        layout_details=layout_details,
        browser_tab_title=browser_tab_title,
    )

    if panel_serve:
        pn.serve(pn.Column(plot))

    return plot


def L3_mosaics_to_html(
    l3_dir: str,
    out_dir: str,
    var: str = "xch4",
    lon_var: str = "lon",
    lat_var: str = "lat",
    overwrite: bool = False,
    html_index: bool = False,
    title_prefix: str = "",
    cmap: str = "viridis",
    clim: Optional[Union[Tuple[float, float], bool]] = None,
    width: int = 850,
    height: int = 750,
    alpha: float = 1,
    single_panel: bool = False,
    background_tile_name_list: Optional[List[str]] = None,
    num_samples_threshold: Optional[float] = None,
    pixel_ratio: int = 1,
    pixel_resolution: Optional[tuple[float, float]] = None,
) -> None:
    """
    l3_dir: full path to the L3 directory, assumes the following directory structure

    l3_dir
    -target_dir
    --resolution_dir
    ---mosaic_file

    out_dir (str): full path to the directory where the plots will be saved
    overwrite (bool): if True overwrite existing plots if they have the same name
    html_index (bool): if True, will generated index.html files in the output directory tree

    plot parameters:

    var (str): name of the variable to plot from the L3 files
    lon_var (str): name of the longitude variable
    lat_var (str): name of the latitude variable
    title_prefix (str): plot titles will be "title_prefix; target; resolution; XCH4 (pbb)"
    cmap (str): name of the colormap
    clim (Optional[Union[Tuple[float, float],bool]]): z-limits for the colorbar, give False to use dynamic colorbar
    width (int): plot width in pixels
    height (int): plot height in pixels
    single_panel (bool): if True, do not add the linked panel with only esri imagery
    background_tile_name_list (Optional[List[str]]): name of the background tile from https://holoviews.org/reference/elements/bokeh/Tiles.html (case insensitive)
                                                    for the main (last value) and linked (first value) panels
    num_samples_threshold (Optional[float]): filter out data with num_samples<num_samples_threshold
    pixel_ratio (int): the static maps will have width x height pixels, this multiplies the number of pixels
    pixel_resolution (Optional[tuple[float, float]]): pixel (width,height) in meters. Overrides pixel_ratio.
    """
    l3_on_gs = l3_dir.startswith("gs://")
    if l3_on_gs:
        target_list = gs_list(l3_dir, srchstr="*_L3_mosaic_*.nc")
        mosaic_file_dict = {}
        for mosaic_file_path in target_list:
            target = os.path.basename(os.path.dirname(os.path.dirname(mosaic_file_path)))
            resolution = os.path.basename(os.path.dirname(mosaic_file_path))
            if target not in mosaic_file_dict:
                mosaic_file_dict[target] = {}
            if resolution not in mosaic_file_dict[target]:
                mosaic_file_dict[target][resolution] = []
            mosaic_file_dict[target][resolution] += [mosaic_file_path]
        target_list = mosaic_file_dict.keys()
    else:
        target_list = os.listdir(l3_dir)

    for target in target_list:
        print(target)
        if l3_on_gs:
            resolution_list = mosaic_file_dict[target].keys()
        else:
            target_dir = os.path.join(l3_dir, target)
            resolution_list = mosaic_file_dict[target].keys()
        for resolution in resolution_list:
            if resolution == "10m":
                continue
            print(f"\t{resolution}")
            if l3_on_gs:
                mosaic_file_list = [
                    os.path.basename(i) for i in mosaic_file_dict[target][resolution]
                ]
            else:
                resolution_dir = os.path.join(target_dir, resolution)
                mosaic_file_list = os.listdir(resolution_dir)
            for file_id, mosaic_file in enumerate(mosaic_file_list):
                print(f"\t\t{mosaic_file}")

                plot_out_dir = os.path.join(out_dir, target, resolution)
                if not os.path.exists(plot_out_dir):
                    os.makedirs(plot_out_dir)
                plot_out_file = os.path.join(plot_out_dir, mosaic_file.replace(".nc", ".html"))

                if os.path.exists(plot_out_file) and not overwrite:
                    continue

                if l3_on_gs:
                    mosaic_file_path = mosaic_file_dict[target][resolution][file_id]
                else:
                    mosaic_file_path = os.path.join(resolution_dir, mosaic_file)

                title = (
                    f"{title_prefix}; {' '.join(target.split('_')[1:])}; {resolution}; XCH4 (ppb)"
                )

                _ = do_html_plot(
                    mosaic_file_path,
                    var,
                    lon_var=lon_var,
                    lat_var=lat_var,
                    out_path=plot_out_file,
                    title=title,
                    cmap=cmap,
                    clim=clim,
                    width=width,
                    height=height,
                    alpha=alpha,
                    single_panel=single_panel,
                    background_tile_name_list=background_tile_name_list,
                    num_samples_threshold=num_samples_threshold,
                    pixel_ratio=pixel_ratio,
                    pixel_resolution=pixel_resolution,
                )

    if html_index:
        print("Generating html index")
        generate_html_index(out_dir)


def generate_html_index(out_dir: str) -> None:
    """
    Uses tree to recursively generated index.html file under out_dir
    These link to all the *html files in out_dir

    out_dir (str): the directory to be recursively indexed
    """

    subprocess.run(
        ["sh", os.path.join(os.path.dirname(__file__), "generate_html_index.sh"), out_dir]
    )


def create_plot_parser(**kwargs):
    plot_parser = argparse.ArgumentParser(**kwargs)
    plot_parser.add_argument(
        "in_path",
        help="input path",
    )
    plot_parser.add_argument(
        "out_path",
        help="full path to the output directory",
    )
    plot_parser.add_argument(
        "--use-get-msat",
        action="store_true",
        help="if given and in_path is a directory, use msat_interface.get_msat to read the data",
    )
    plot_parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="if given, overwrite existing plots",
    )
    plot_parser.add_argument(
        "--title",
        default="",
        help="Will be added to the plot titles",
    )
    plot_parser.add_argument(
        "--layout-title",
        default="",
        help="title text to be displayed above all the plots",
    )
    plot_parser.add_argument(
        "--layout-details",
        default="",
        help="normal text to be displayed above all the plots (and under layout_title)",
    )
    plot_parser.add_argument(
        "--tab-title",
        default="MethaneAIR map",
        help="text in the browser tab",
    )
    plot_parser.add_argument(
        "-i",
        "--index",
        action="store_true",
        help="if given, will generate index.html files in the output directory tree",
    )
    plot_parser.add_argument(
        "-c",
        "--cmap",
        default="viridis",
        help="colormap name",
    )
    plot_parser.add_argument(
        "--clim-bounds",
        nargs=2,
        type=float,
        default=None,
        help="Set fixed limits for the colorbar",
    )
    plot_parser.add_argument(
        "--dynamic-clim",
        action="store_true",
        help="if given, using dynamic colorbar (readjusts to the data displayed)",
    )
    plot_parser.add_argument(
        "--width",
        type=int,
        default=850,
        help="width of the plots in pixels",
    )
    plot_parser.add_argument(
        "--height",
        type=int,
        default=750,
        help=" heigh of the plots in pixels",
    )
    plot_parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=1.0,
        help="alpha value (transparency) for the plot, between 0 (transparent) and 1 (opaque)",
    )
    plot_parser.add_argument(
        "-v",
        "--variable",
        nargs="+",
        default="xch4",
        help="name of the variable to plot from the L3 files",
    )
    plot_parser.add_argument("--lon-var", default="lon", help="netCDF variable path for longitude")
    plot_parser.add_argument("--lat-var", default="lat", help="netCDF variable path for latitude")
    plot_parser.add_argument(
        "--serve",
        action="store_true",
        help="if given, open the plot in an interactive session",
    )
    plot_parser.add_argument(
        "--single-panel",
        action="store_true",
        help="if given, do not add the linked panel with only ESRI imagery (e.g. with alpha<1)",
    )
    plot_parser.add_argument(
        "--background-tile",
        default=["EsriImagery"],
        nargs="*",
        help="background tile name from https://holoviews.org/reference/elements/bokeh/Tiles.html (case insensitive)."
        "Can take up to 2 values uses for the main (last value) and linked (first value) panels."
        "If only one value is given it is used for both panels.",
    )
    plot_parser.add_argument(
        "--filter-num-samples",
        default=None,
        type=float,
        help="if used with L3 files, filter out data with num_samples<filter_num_samples",
    )
    plot_parser.add_argument(
        "--pixel-ratio",
        default=1,
        type=int,
        help="Resolution of the static plots (or initial image with --serve) will be (width*height)*pixel_ratio",
    )
    plot_parser.add_argument(
        "--add-standalone-imagery",
        action="store_true",
        help="if given with multiple variables, add a panel with only the backrgound tile",
    )
    plot_parser.add_argument(
        "--ncols",
        type=int,
        default=3,
        help="Number of columns the panels will be arranged in when using multiple variables",
    )
    plot_parser.add_argument(
        "--add-context",
        default="",
        choices=["l1", "l2", "l2pp", "l3"],
        help="add some default variables to the --variables",
    )
    plot_parser.add_argument(
        "--option",
        type=str,
        default=None,
        help="one of numpy operations to apply on the first --variable (e.g. nanmean)",
    )
    plot_parser.add_argument(
        "--option-axis-dim",
        type=str,
        default="spectral_channel",
        help="dimension name along which --option will be applied",
    )
    plot_parser.add_argument(
        "--apply-flag",
        type=str,
        default=None,
        help="use this flag to filter out the main data",
    )
    plot_parser.add_argument(
        "--pixel-resolution",
        nargs=2,
        type=float,
        default=None,
        help="[pixel_width,pixel_height] in meters. Overrides --pixel-ratio.",
    )
    return plot_parser


def create_parser():
    description = """Generate interactive maps from L1B/L2/L3 MethaneSAT/AIR files

    The behavior changes based on the nature of the in_path argument.

    .directory: L3 specific, in_path is the path to a directory that has an assumed structure (see README)
    .netcdf file: in_path is the full path to a L1B/L2/L3 file
        the code will generate a static map for the given file
        --serve is only used by this case to popup the plot with a local webserver with dynamic regridding
    .csv file: in_path is the full path to a csv table that stores metadata on files
        the code will use mair_ls to get a list of files matching the filter arguments and
        will generate a static map for each file.
        the following filter arguments are only used if in_path is a csv file:
        --uri
        --aggregation
        --resolution
        --production-operation
        --flight-date
        --timestamp
        --time-start
        --time-end
        --production-environment
        --latest
        --molecule
        --show (when given, exit after calling mair_ls, without generating plots)
        the following arguments are ignored when in_path is a csv file:
        --title
        --serve

    NOTE: Be careful when using a csv file as in_path, first check which files you would get using mairls
    """
    ls_parser = create_ls_parser(add_help=False)

    plot_parser = create_plot_parser(
        parents=[ls_parser],
        formatter_class=argparse.RawTextHelpFormatter,
        description=description,
    )

    return plot_parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.dynamic_clim and args.clim_bounds is not None:
        raise Exception("Cannot give both --clim-bounds and --dynamic-clim")
    elif args.dynamic_clim:
        clim = False
    elif args.clim_bounds is not None:
        clim = tuple(args.clim_bounds)
    else:
        clim = args.clim_bounds

    if args.add_context:
        args.variable += [
            i for i in CONTEXT_VARIABLES_DICT[args.add_context] if i not in args.variable
        ]

    if args.in_path.endswith(".csv"):
        mair_ls_arglist = inspect.getfullargspec(mair_ls).args
        mair_ls_args = {k: v for k, v in vars(args).items() if k in mair_ls_arglist}
        df = mair_ls(**mair_ls_args)
        if args.show:
            print("Exiting after showing list of files because --show was given")
            return
        uri_list = df.uri.tolist()
        check = input(f"Will generate {len(uri_list)} maps, continue? (Y/N)\n")
        if check.lower() not in ["y", "yes"]:
            return
        if any(["xch4" in i.lower() for i in args.variable]):
            label = "XCH4 (ppb)"
        else:
            label = args.variable[0]
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
        for i, row in df.iterrows():
            if "level3_target_name" in row:
                title = f'{row["flight_name"].upper()} {row["production_operation"]}; {row["level3_target_name"]}; {label}'
                out_path = os.path.join(
                    args.out_path,
                    f'{row["flight_name"]}_{row["production_operation"]}_{row["aggregation"]}_{row["level3_resolution"]}_{row["level3_target_name"]}.html',
                )
            else:
                title = f'{row["flight_name"].upper()} {row["production_operation"]}; {label}'
                out_path = os.path.join(
                    args.out_path,
                    f'{row["flight_name"]}_{row["production_operation"]}_{row["aggregation"]}.html',
                )
            print(row["uri"])
            _ = do_html_plot(
                row["uri"],
                args.variable,
                lon_var=args.lon_var,
                lat_var=args.lat_var,
                out_path=out_path,
                title=title,
                cmap=args.cmap,
                clim=clim,
                width=args.width,
                height=args.height,
                alpha=args.alpha,
                panel_serve=False,
                single_panel=args.single_panel,
                background_tile_name_list=args.background_tile,
                num_samples_threshold=args.filter_num_samples,
                pixel_ratio=args.pixel_ratio,
                add_standalone_imagery=args.add_standalone_imagery,
                ncols=args.ncols,
                layout_title=args.layout_title,
                layout_details=args.layout_details,
                browser_tab_title=args.tab_title,
                option=args.option,
                option_axis_dim=args.option_axis_dim,
                apply_flag=args.apply_flag,
                pixel_resolution=args.pixel_resolution,
            )

    elif os.path.splitext(args.in_path)[1] != "" or args.use_get_msat:
        # If in_path point directly to a L3 mosaic file
        _ = do_html_plot(
            args.in_path,
            args.variable,
            lon_var=args.lon_var,
            lat_var=args.lat_var,
            out_path=args.out_path,
            title=args.title,
            cmap=args.cmap,
            clim=clim,
            width=args.width,
            height=args.height,
            alpha=args.alpha,
            panel_serve=args.serve,
            single_panel=args.single_panel,
            background_tile_name_list=args.background_tile,
            num_samples_threshold=args.filter_num_samples,
            pixel_ratio=args.pixel_ratio,
            add_standalone_imagery=args.add_standalone_imagery,
            ncols=args.ncols,
            layout_title=args.layout_title,
            layout_details=args.layout_details,
            browser_tab_title=args.tab_title,
            option=args.option,
            option_axis_dim=args.option_axis_dim,
            apply_flag=args.apply_flag,
            pixel_resolution=args.pixel_resolution,
        )
    else:
        # If in_path points to a directory structured as expected by L3_mosaics_to_html
        L3_mosaics_to_html(
            args.in_path,
            args.out_path,
            var=args.variable,
            lon_var=args.lon_var,
            lat_var=args.lat_var,
            overwrite=args.overwrite,
            html_index=args.index,
            title_prefix=args.title,
            cmap=args.cmap,
            clim=clim,
            width=args.width,
            height=args.height,
            alpha=args.alpha,
            single_panel=args.single_panel,
            background_tile_name_list=args.background_tile,
            num_samples_threshold=args.filter_num_samples,
            pixel_ratio=args.pixel_ratio,
            pixel_resolution=args.pixel_resolution,
        )


if __name__ == "__main__":
    main()
