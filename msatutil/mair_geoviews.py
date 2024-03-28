import warnings

warnings.simplefilter("ignore")
import os
import numpy as np
import argparse
import inspect

os.environ["BOKEH_PY_LOG_LEVEL"] = "error"
import bokeh
from bokeh.models import CustomJS, Slider, Column, Select, Row
from bokeh.resources import CDN
from bokeh.embed import file_html

import holoviews as hv
from holoviews.operation.datashader import rasterize
from holoviews.plotting import list_cmaps
from holoviews.plotting.util import process_cmap
import geoviews as gv
from geoviews.tile_sources import EsriImagery
import panel as pn

from msatutil.msat_dset import msat_dset, gs_list
from msatutil.mair_ls import mair_ls
from msatutil.mair_ls import create_parser as create_ls_parser

from typing import Optional, Tuple, Union, List

import subprocess

hv.extension("bokeh")


def show_map(
    x,
    y,
    z,
    width: int = 450,
    height: int = 450,
    cmap: str = "viridis",
    clim: Optional[Union[Tuple[float, float], bool]] = None,
    alpha: int = 1,
    title: str = "",
    background_tile_list=[EsriImagery],
    single_panel: bool = False,
    pixel_ratio: int = 1,
    active_tools=['pan', 'wheel_zoom'],
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

    Outputs:
        geoviews figure
    """

    quad = gv.project(gv.QuadMesh((x, y, z)))

    if clim is None:
        # define color limits as mean +/- 3 std
        mean_z = np.nanmean(z)
        std_z = np.nanstd(z, ddof=1)
        clim = (mean_z - 3 * std_z, mean_z + 3 * std_z)

    raster = rasterize(quad, width=width, height=height, pixel_ratio=pixel_ratio).opts(
        width=width,
        height=height,
        cmap=cmap,
        colorbar=True,
        title=title,
        alpha=alpha,
        active_tools=active_tools,
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


def save_static_plot_with_widgets(out_file: str, plot, alpha: float = 1.0, cmap: str = "viridis"):
    """
    Save the output of show_map to a html file

    transform into a bokeh object
    add an alpha slider and a colormap selector
    save as an html file
    """

    bokeh_plot = hv.render(plot, backend="bokeh")

    if type(bokeh_plot) is bokeh.models.plots.GridPlot:
        glyph = bokeh_plot.children[0][0].renderers[1].glyph
        # get rid of the dummy legend
        bokeh_plot.children[0][0].min_border_right = 100
        bokeh_plot.children[1][0].min_border_right = 100
        bokeh_plot.children[1][0].right[0].visible = False
    elif type(bokeh_plot) is bokeh.plotting._figure.figure:
        glyph = bokeh_plot.renderers[1].glyph

    # alpha slider
    callback = CustomJS(args={"plot": glyph}, code="plot.global_alpha = cb_obj.value;")
    alpha_slider = Slider(start=0, end=1.0, step=0.1, value=alpha, title="Heatmap alpha")
    alpha_slider.js_on_change("value", callback)

    # color palette select
    uniform_sequential_palettes = list_cmaps(category="Uniform Sequential")
    sequential_palettes = list_cmaps(category="Sequential")
    palette_dict = {k.lower() + " (u)": process_cmap(k) for k in uniform_sequential_palettes}
    for k in set(sequential_palettes).difference(uniform_sequential_palettes):
        palette_dict[k.lower()] = process_cmap(k)

    palette_select = Select(
        options=sorted(list(palette_dict.keys())),
        value=cmap.lower(),
        title="Colormap",
    )
    callback = CustomJS(
        args={"glyph": glyph, "palette_dict": palette_dict},
        code="glyph.color_mapper.palette = palette_dict[cb_obj.value];",
    )
    palette_select.js_on_change("value", callback)

    bokeh_layout = Column(bokeh_plot, Row(palette_select, alpha_slider))

    with open(out_file, "w") as out:
        out.write(file_html(bokeh_layout, CDN, "MethaneAIR map"))

    print(out_file)


def do_single_map(
    data_file: str,
    var: str,
    lon_var: str = "lon",
    lat_var: str = "lat",
    out_path: Optional[str] = None,
    title: str = "",
    cmap: str = "viridis",
    clim: Optional[Union[Tuple[float, float], bool]] = None,
    width: int = 850,
    height: int = 750,
    alpha: float = 1,
    panel_serve: bool = False,
    single_panel: bool = False,
    background_tile_name_list: Optional[List[str]] = None,
    num_samples_threshold: Optional[float] = None,
    pixel_ratio: int = 1,
) -> None:
    """
    Save a html plot of var from data_file

    data_file (str): full path to the input netcdf data file, can be a gs:// path
    var (str): variable name in the data file
    lon_var (str): name of the longitude variable
    lat_var (str): name of the latitude variable
    out_path (Optional[str]): full path to the output .html file or to a directory where it will be saved.
                              If None, save output html file in the current working directory
    title (str): title of the plot (include field name and units here)
    cmap (str): colormap name
    clim (Optional[Union[Tuple[float, float],bool]]): z-limits for the colorbar, give False to use dynamic colorbar
    width (int): plot width in pixels
    height (int): plot height in pixels
    panel_serve (bool): if True, start an interactive session
    single_panel (bool): if True, do not add the linked panel with only esri imagery
    background_tile_name_list (Optional[List[str]]): name of the background tile from https://holoviews.org/reference/elements/bokeh/Tiles.html (case insensitive)
                                                    for the main (last value) and linked (first value) panels
    num_samples_threshold (Optional[float]): filter out data with num_samples<num_samples_threshold
    pixel_ratio (int): the initial map (and the static maps) will have width x height pixels, this multiplies the number of pixels
    """
    default_html_filename = os.path.basename(data_file).replace(".nc", ".html")
    if out_path is None:
        out_file = os.path.join(os.getcwd(), default_html_filename)
    elif os.path.splitext(out_path)[1] == "":
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file = os.path.join(out_path, default_html_filename)
    else:
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        out_file = out_path

    with msat_dset(data_file) as nc:
        lon = nc[lon_var][:]
        lat = nc[lat_var][:]
        v = nc[var][:]
        if num_samples_threshold is not None:
            num_samples = nc["num_samples"][:]
            v[num_samples < num_samples_threshold] = np.nan

    if background_tile_name_list is not None:
        tile_dict = {k.lower(): v for k, v in gv.tile_sources.__dict__["tile_sources"].items()}
        background_tile_list = ["" for i in background_tile_name_list]
        for i, background_tile_name in enumerate(background_tile_name_list):
            if background_tile_name.lower() not in tile_dict:
                background_tile_list[i] = tile_dict["esriimagery"]
            else:
                background_tile_list[i] = tile_dict[background_tile_name.lower()]

    plot = show_map(
        lon,
        lat,
        v,
        title=title,
        cmap=cmap,
        clim=clim,
        width=width,
        height=height,
        alpha=alpha,
        single_panel=single_panel,
        background_tile_list=background_tile_list,
        pixel_ratio=pixel_ratio,
    )

    save_static_plot_with_widgets(out_file, plot, alpha=alpha, cmap=cmap)

    if panel_serve:
        pn.serve(pn.Column(plot))


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

                do_single_map(
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
        "-o",
        "--overwrite",
        action="store_true",
        help="if given, overwrite existing plots",
    )
    plot_parser.add_argument(
        "-t",
        "--title",
        default="",
        help="Will be added to the plot titles",
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
        if "xch4" in args.variable.lower():
            label = "XCH4 (ppb)"
        else:
            label = args.variable
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
            do_single_map(
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
            )

    elif os.path.splitext(args.in_path)[1] != "":
        # If in_path point directly to a L3 mosaic file
        do_single_map(
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
        )
    else:
        # If in_path point to a directory
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
        )


if __name__ == "__main__":
    main()
