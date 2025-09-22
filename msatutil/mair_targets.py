import argparse
import re
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Callable, Optional

import geopandas as gpd
import geoviews as gv
import holoviews as hv
import pandas as pd
from bokeh.embed import file_html
from bokeh.models import (
    BoxSelectTool,
    Button,
    Column,
    ColumnDataSource,
    CustomJS,
    DateRangeSlider,
    Div,
    GlyphRenderer,
    HoverTool,
    InlineStyleSheet,
    Row,
    Select,
    TabPanel,
    Tabs,
    TapTool,
    TextInput,
)
from bokeh.plotting import figure
from bokeh.resources import CDN
from geoviews.element import WMTS
import reverse_geocode

from msatutil.msat_targets import (
    GOOGLE_IMAGERY,
    extract_timestamp,
    gs_posixpath_to_str,
    plot_polygons,
    none_or_str,
)
import warnings

warnings.simplefilter("ignore")

gv.extension("bokeh")

BASE_MAPS = {
    "GOOGLE": GOOGLE_IMAGERY,
    "ESRI": gv.tile_sources.EsriImagery(),
}


class PONumber:
    """
    Object representing a MAIR/SAT processing ID (PID)
    adds a > operator to check if a PID is larger (newer) than another
    """

    def __init__(self, val: str):
        match = re.search(r"(?i)po[-_](\d+)([A-Za-z]?)", val)
        self.str = val
        self.value = int(match.group(1)) if match else None
        self.extra = ord(match.group(2).upper()) - ord("A") + 1 if match and match.group(2) else 0

    def __gt__(self, other):
        if isinstance(other, PONumber):
            if self.value == other.value:
                return self.extra > other.extra
            else:
                return self.value > other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, PONumber):
            return self.value == other.value and self.extra == other.extra
        return NotImplemented

    def __str__(self):
        return self.str

    def __repr__(self):
        return self.__str__()


def derive_L3_html_path(
    l3_mosaic_file_path: PosixPath, html_bucket: str, campaign: str, flight: str, area: str
) -> str:

    fname = l3_mosaic_file_path.name.replace(".nc", ".html")

    html_path = Path(f"gs://{html_bucket}") / campaign / flight / area / "30m" / fname

    return gs_posixpath_to_str(html_path).replace("gs://", "https://storage.cloud.google.com/")


def get_target_dict(file_list: str, func: Callable = gs_posixpath_to_str, **kwargs) -> dict:
    with open(file_list, "r") as fin:
        file_list = [Path(i.strip()) for i in fin.readlines()]

    is_L2 = "level2" in str(file_list[0])
    ids_L2 = {
        "campaign": 2,
        "flight": 6,
        "pid": 7,
    }
    ids_L3 = ids_L2.copy()
    ids_L3["area"] = 12
    ids_L3_b = {
        "campaign": 2,
        "flight": 4,
        "pid": 5,
        "area": 11,
    }
    ids_L3_c = ids_L3_b.copy()
    ids_L3_c["area"] = 13

    d = {}
    for i in file_list:
        if is_L2:
            ids = ids_L2
        elif len(i.parts[3]) == 4:
            ids = ids_L3
        elif "mosaic/t" in str(i):
            ids = ids_L3_c
        else:
            ids = ids_L3_b
        params = {k: i.parts[v] for k, v in ids.items()}

        pid = PONumber(params["pid"])

        if pid.value is None:
            print(f"Could not parse pid for {i}")
            continue

        c = params["campaign"]
        f = params["flight"]
        p = params["pid"]

        if c not in d:
            d[c] = {}
        if f not in d[c]:
            d[c][f] = {}

        # only keep the highest pid
        # if a processing exists, overwrite with the higher pid
        if d[c][f] != {}:
            old_pid = PONumber(list(d[c][f].keys())[0])
            if old_pid > pid:
                continue
            elif pid > old_pid:
                del d[c][f][str(old_pid)]

        if is_L2:
            d[c][f] = {p: func(i, **kwargs).replace("gs://", "https://storage.cloud.google.com/")}
        else:
            if p not in d[c][f]:
                d[c][f][p] = {}
            if "html_bucket" in kwargs:
                for k in ["campaign", "flight", "area"]:
                    kwargs[k] = params[k]
            d[c][f][p][params["area"]] = func(i, **kwargs)

    return d


def get_info(lat: float, lon: float, key: str):
    """
    Get some metadata corresponding to the given lat/lon

    Inputs:
        lat (float): latitude
        lon (float): longitude
        key (str): one of the reverse_geocode outputs
    Outputs:
        result (Optional[str]): the resulting info
    """
    try:
        result = reverse_geocode.get((lat, lon))[key]
    except Exception:
        result = None
    return result


def first_non_dict_value(d: dict):
    if not isinstance(d, dict):  # already a value
        return d
    for v in d.values():
        result = first_non_dict_value(v)
        if result is not None:
            return result
    return None  # nothing found


def make_mair_targets_map(
    polygon_file: str,
    out_file: str,
    file_list: str,
    title: str = "MethaneAIR flights",
    html_bucket: Optional[str] = None,
    imagery_source: str = "ESRI",
    write: bool = True,
):
    """
    Read the list of targets from the infile geojson file and make a html map

    Inputs:
        polygon_file (str): input geojson file with all the target polygons
        out_file (str): full path to the output html file
        file_list (str): full path to list of data bucket files
        title (str): map title
        html_bucket (Optional[str]): bucket on which the mosaic html maps are saved
        imagery_source (str): one of GOOGLE or ESRI
        write (bool): if True, write the html map file
    """
    gdf = gpd.read_file(polygon_file)

    gdf["country"] = gdf["geometry"].apply(
        lambda x: get_info(x.centroid.y, x.centroid.x, "country")
    )
    gdf["state"] = gdf["geometry"].apply(lambda x: get_info(x.centroid.y, x.centroid.x, "state"))

    gdf["default_color"] = "purple"

    gdf["fill_color"] = gdf["default_color"].copy()
    gdf["line_color"] = gdf["default_color"].copy()
    gdf["fill_alpha"] = 0.7
    gdf["default_alpha"] = gdf["fill_alpha"].copy()
    gdf["data_files"] = ""
    gdf["single_file"] = ""

    vdims = [
        "campaign",
        "flight",
        "country",
        "state",
        "fill_color",
        "default_color",
        "line_color",
        "fill_alpha",
        "default_alpha",
        "data_files",
    ]
    hover_tooltips = [
        ("Campaign", "@campaign"),
        ("Flight", "@flight"),
        ("Country", "@country"),
        ("State", "@state"),
    ]

    map_tools = ["hover", "fullscreen", "tap", "box_select"]
    td = get_target_dict(file_list)
    first_path = first_non_dict_value(td)
    is_L2 = "level2" in first_path
    do_html = "_L3_" in first_path and html_bucket is not None
    scatter_columns = ["data_files", "single_file", "campaign", "flight", "pid", "country", "state"]
    if do_html:
        gdf["html_files"] = ""
        vdims += ["html_files"]
        scatter_columns += ["html_files"]
        html_td = get_target_dict(file_list, derive_L3_html_path, html_bucket=html_bucket)
    for c in td:
        if c not in gdf["campaign"].values:
            continue
        for f in td[c]:
            if f not in gdf["flight"].values:
                continue
            p = list(td[c][f].keys())[0]

            gdf.loc[gdf["flight"] == f, "data_files"] = (
                td[c][f][p] if is_L2 else "\n".join([td[c][f][p][a] for a in td[c][f][p]])
            )
            gdf.loc[gdf["flight"] == f, "single_file"] = first_non_dict_value(td[c][f][p])
            if do_html:
                gdf.loc[gdf["flight"] == f, "html_files"] = "\n".join(
                    [html_td[c][f][p][a] for a in td[c][f][p]]
                )
    scatter_df = gdf[scatter_columns]
    scatter_df = scatter_df[~(scatter_df["single_file"]=="")]
    scatter_df["timestamps"] = scatter_df["single_file"].apply(
        lambda x: extract_timestamp(x, True) if is_L2 else extract_timestamp(Path(x).name)
    )
    scatter_df["timestrings"] = scatter_df["timestamps"].astype(str)
    scatter_df["counts"] = 1
    scatter_df.loc[pd.isna(scatter_df["timestamps"]), "counts"] = 0
    scatter_df = scatter_df.sort_values(by=["timestamps"]).reset_index()
    scatter_df["cumulcounts"] = scatter_df["counts"].cumsum()
    scatter_df["color"] = "#1f77b4"
    scatter_df["size"] = 4

    base_map = BASE_MAPS[imagery_source]
    bokeh_plot, poly_source, poly_renderer = plot_polygons(
        gdf, vdims, map_tools, base_map, hover_tooltips, title
    )

    inp = TextInput(value="", title="Highlight this flight:")

    # callback to highlight the polygon corresponding to the target in the input widget
    inp_callback_code = """
    if (typeof window.zoom_input === "undefined") {
        window.zoom_input = 1;
    }

    var data = poly_source.data;
    var country = data['country'];
    var state = data['state'];
    var selected_flight = cb_obj.value;
    var color = data['fill_color'];
    var default_color = data['default_color'];
    var line_color = data['line_color'];
    var xs = data['xs'];
    var ys = data['ys'];
    var campaign = data['campaign'];
    var flight = data['flight'];
    var data_files = data['data_files'];
    var alpha = Array.from(data['fill_alpha']);
    var default_alpha = Array.from(data['default_alpha']);
    if ('html_files' in data) var html_files = data['html_files'];

    var hid = -1;
    for (var i=0;i<flight.length;i++){
        if (flight[i]==selected_flight) {
            hid=i;
            break;
        }
    }

    if (hid>=0){
        function moveToEnd(arr) {
            arr.push(arr.splice(hid, 1)[0]);
        }
        moveToEnd(campaign);
        moveToEnd(flight);
        moveToEnd(data_files);
        moveToEnd(xs);
        moveToEnd(ys);
        moveToEnd(country)
        moveToEnd(state)
        moveToEnd(campaign)
        if ('html_files' in data) moveToEnd(html_files);
        moveToEnd(alpha)
        moveToEnd(default_alpha)
        moveToEnd(default_color)
        moveToEnd(line_color)
        data['fill_alpha'] = new Float32Array(alpha);
        data['default_alpha'] = new Float32Array(default_alpha);
        for (var i=0;i<color.length;i++){
            color[i] = default_color[i];
            data['fill_alpha'][i] = data['default_alpha'][i];
        }
        color[color.length-1] = 'red';
        data['fill_alpha'][alpha.length-1] = 1;
    } else {
        color[color.length-1] = default_color[default_color.length-1];
        data['fill_alpha'][alpha.length-1] = data['default_alpha'][default_alpha.length-1];
    }
    poly_renderer.glyph.fill_alpha = {field: 'fill_alpha'};
    poly_renderer.glyph.fill_color = {field: 'fill_color'};
    poly_source.change.emit();

    // if the input widget is entered manually, reset the plot
    if (window.zoom_input==1) {
        var x = poly_source.data["xs"][xs.length-1][0][0];
        var y = poly_source.data["ys"][ys.length-1][0][0];
        const mean = arr => arr.reduce((sum, val) => sum + val, 0) / arr.length;
        const x_offset = 3300000;
        const y_offset = 2400000;
        const mean_x = mean(x);
        const mean_y = mean(y);
        plot.x_range.setv({start: mean_x - x_offset, end:  mean_x + x_offset});
        plot.y_range.setv({start: mean_y - y_offset, end: mean_y + y_offset});
    } else {
        window.zoom_input = 1;
    }
    """

    inp_callback_args = {
        "poly_source": poly_source,
        "poly_renderer": poly_renderer,
        "plot": bokeh_plot,
    }

    poly_hover = bokeh_plot.select_one(HoverTool)
    poly_hover.callback = CustomJS(
        args={"inp": inp, "poly_source": poly_source},
        code="""
        const selected = cb_data["index"].indices;
        const index = selected[selected.length-1];
        const flight = poly_source.data["flight"][index];

        if (selected.length>0 && inp.value!=flight) {
            window.zoom_input = 0;
            inp.value = flight;
        }
        """,
    )

    taptool = bokeh_plot.select_one(TapTool)
    taptool_callback_args = {"poly_source": poly_source}
    file_type_select_options = ["Data"]
    if do_html:
        file_type_select_options += ["HTML Plots"]
        file_type_select = Select(
            options=file_type_select_options, value="Data", title="File type:"
        )
        taptool_callback_args["file_type_select"] = file_type_select
    # callback to copy the corresponding files when clicking on a polygon
    taptool.callback = CustomJS(
        args=taptool_callback_args,
        code="""
        const selected_indices = poly_source.selected.indices;
        let key;
        if (typeof file_type_select !== 'undefined'){
            if (file_type_select.value==='Data') {
                key = 'data_files';
            } else if (file_type_select.value==='HTML Plots') {
                key = 'html_files';
            }
        } else {
            key = 'data_files';
        }

        // Only create the popup div if it doesn't exist
        if (!document.getElementById("popup-div")) {
            const popup = document.createElement("div");
            popup.id = "popup-div";
            popup.style.position = "fixed";
            popup.style.top = "20%";
            popup.style.left = "40%";
            popup.style.transform = "translate(-50%, -50%)";  // shift by half its size
            popup.style.zIndex = "9999";
            popup.style.background = "white";
            popup.style.border = "1px solid black";
            popup.style.padding = "10px";
            popup.style.maxHeight = "200px";
            popup.style.overflow = "auto";
            popup.style.whiteSpace = "pre-wrap";
            popup.style.display = "none";
            document.body.appendChild(popup);
        }

        if (selected_indices.length > 0) {
            let all_data_files = [];
            
            // Loop over all selected polygons
            for (let i = 0; i < selected_indices.length; i++) {
                const idx = selected_indices[i];
                const data_files = poly_source.data[key][idx];
                all_data_files.push(data_files);
            }
            
            const combined_paths = all_data_files.join('\\n');

            // Copy to clipboard
            navigator.clipboard.writeText(combined_paths).then(function() {
                const popup = document.getElementById("popup-div");
                popup.innerHTML =
                  `<div style='text-align:right; font-weight:bold; cursor:pointer;' onclick='this.parentElement.style.display="none"'>×</div>` +
                  "<b>File paths copied:</b><br>" +
                  combined_paths.split('\\n').map(path => {
                    const reg = /\/\d{8}\//;
                    const parts = path.split(/[/\\\\]/);
                    const part6 = parts[6] ?? '';
                    const part5 = parts[5] ?? '';
                    const part7 = parts[7] ?? '';
                    const part8 = parts[8] ?? '';
                    let name;
                    if (path.includes("_L3_") && path.includes(".html")) {
                        name = part6 + ": " + parts[parts.length-3] + " HTML PLOT";
                    } else if (path.includes("_L3_") && path.includes(".nc")) {
                        if (reg.test(path)) {
                            name = part5 + ": " + parts[parts.length-3] + " DATA FILE";
                        } else {
                            name = part7 + ": " + parts[parts.length-3] + " DATA FILE";
                        }
                    } else if (path.includes(".pdf")) {
                        name = part8 + " QAQC report";
                    } else {
                        name = parts.pop();
                    }
                    return `<a href='${path}' target='_blank'>${name}</a>`;
                  }).join('<br>');
                popup.style.display = "block";
            }, function(err) {
                console.error('Failed to copy text: ', err);
            });
        }
        poly_source.selected.indices = [];
        poly_source.change.emit();
    """,
    )

    scatter_source = ColumnDataSource(scatter_df)
    fig = figure(
        title=f"{scatter_df['counts'].sum()} flights over {len(list(td.keys()))} campaigns",
        width=350,
        height=300,
        x_axis_type="datetime",
        tools="pan,wheel_zoom,box_zoom,reset,tap",
        active_drag="pan",
        active_scroll="wheel_zoom",
    )
    scatter = fig.scatter(
        "timestamps",
        "cumulcounts",
        source=scatter_source,
        color="color",
        size="size",
    )
    scatter_hover = HoverTool(
        tooltips=[
            ("Campaign", "@campaign"),
            ("Flight", "@flight"),
            ("Process ID", "@pid"),
            ("Country", "@country"),
            ("State", "@state"),
            ("UTC Time", "@timestrings"),
        ],
        renderers=[scatter],
    )
    fig.add_tools(scatter_hover)

    scatter_taptool = fig.select_one(TapTool)
    scatter_taptool_callback_args = {"scatter_source": scatter_source}
    if do_html:
        scatter_taptool_callback_args["file_type_select"] = file_type_select

    # callback to copy the corresponding file path when clicking on the scatter points
    scatter_taptool.callback = CustomJS(
        args=scatter_taptool_callback_args,
        code="""
        const selected = scatter_source.selected.indices;
        let key;
        if (typeof file_type_select !== 'undefined'){
            if (file_type_select.value==='Data') {
                key = 'data_files';
            } else if (file_type_select.value==='HTML Plots') {
                key = 'html_files';
            }
        } else {
            key = 'data_files';
        }

        // Only create the popup div if it doesn't exist
        if (!document.getElementById("popup-div")) {
            const popup = document.createElement("div");
            popup.id = "popup-div";
            popup.style.position = "fixed";
            popup.style.top = "20%";
            popup.style.left = "40%";
            popup.style.transform = "translate(-50%, -50%)";  // shift by half its size
            popup.style.zIndex = "9999";
            popup.style.background = "white";
            popup.style.border = "1px solid black";
            popup.style.padding = "10px";
            popup.style.maxHeight = "200px";
            popup.style.overflow = "auto";
            popup.style.whiteSpace = "pre-wrap";
            popup.style.display = "none";
            document.body.appendChild(popup);
        }

        const combined_paths = scatter_source.data[key][selected[selected.length-1]];

        // Copy to clipboard
        navigator.clipboard.writeText(combined_paths).then(function() {
            const popup = document.getElementById("popup-div");
            popup.innerHTML =
              `<div style='text-align:right; font-weight:bold; cursor:pointer;' onclick='this.parentElement.style.display="none"'>×</div>` +
              "<b>File paths copied:</b><br>" +
              combined_paths.split('\\n').map(path => {
                const reg = /\/\d{8}\//;
                const parts = path.split(/[/\\\\]/);
                const part6 = parts[6] ?? '';
                const part5 = parts[5] ?? '';
                const part7 = parts[7] ?? '';
                const part8 = parts[8] ?? '';
                let name;
                if (path.includes("_L3_") && path.includes(".html")) {
                    name = part6 + ": " + parts[parts.length-3] + " HTML PLOT";
                } else if (path.includes("_L3_") && path.includes(".nc")) {
                    if (reg.test(path)) {
                        name = part5 + ": " + parts[parts.length-3] + " DATA FILE";
                    } else {
                        name = part7 + ": " + parts[parts.length-3] + " DATA FILE";
                    }
                } else if (path.includes(".pdf")) {
                    name = part8 + " QAQC report";
                } else {
                    name = parts.pop();
                }
                return `<a href='${path}' target='_blank'>${name}</a>`;
              }).join('<br>');
            popup.style.display = "block";
        }, function(err) {
            console.error('Failed to copy text: ', err);
        });

        scatter_source.selected.indices = [];
        scatter_source.change.emit()

        """,
    )

    # CustomJS Callback to Highlight Polygons and scatter points on scatter Hover
    # this does it by changing the value of the input widget, which triggers the input callback
    # that will also highlight the corresponding scatter points
    scatter_hover.callback = CustomJS(
        args={"scatter_source": scatter_source, "inp": inp},
        code="""
        var hovered_indices = cb_data["index"].indices;
        const scatter_data = scatter_source.data;
        const hovered_index = hovered_indices[hovered_indices.length-1];
        const flight = scatter_source.data["flight"][hovered_index];

        if (hovered_indices.length>0 && inp.value!=flight){
            inp.value = flight;
        } 

        setTimeout(function() {
            // this is to only keep the last tooltip when hovering over clumped points
            // set the other tooltips display to none
            var tooltips = window.document.querySelectorAll(".bk-Tooltip")[0]?.shadowRoot?.children[7]?.children[0]?.children;
            
            if (tooltips) {
                for (let i = 0; i < tooltips.length - 1; i++) {
                    tooltips[i].style.display = 'none';
                }
            } else {
                console.warn("Tooltips not found");
            }
        }, 100);
        """,
    )

    inp_callback_args["scatter_source"] = scatter_source
    # callback to highlight scatter points corresponding to the target in the input widget
    inp_callback_code += """
    var scatter_colors = scatter_source.data["color"];
    var scatter_size = scatter_source.data["size"];
    var flight = scatter_source.data["flight"];

    for (let i = 0; i < flight.length; i++) {
        scatter_colors[i] = (flight[i] === selected_flight) ? "red" : "#1f77b4";
        scatter_size[i] = (flight[i] === selected_flight) ? 8 : 4;
    }

    scatter_source.change.emit(); 
    """

    start = scatter_df["timestamps"][0].date()
    end = scatter_df["timestamps"][len(scatter_df.index) - 1].date()
    date_slider = DateRangeSlider(
        value=(start, end),
        start=start,
        end=end,
        width=300,
    )
    date_slider_info_div = Div(
        text=f"{scatter_df['counts'].sum()} flights in selected date range",
        width=400,
    )
    # callback to update a text Div with the number of flights in the selected date range
    date_slider_callback = CustomJS(
        args={"info_div": date_slider_info_div, "scatter_source": scatter_source},
        code="""
        const timestamps = scatter_source.data["timestamps"];
        const start = cb_obj.value[0];
        const end = cb_obj.value[1] + 86400000; // make the last date inclusive
        const count = timestamps.filter(x => x >= start && x <= end).length;
        info_div.text = `${count} flights in selected date range`;
        """,
    )
    date_slider.js_on_change("value_throttled", date_slider_callback)
    date_slider_button = Button(label="Copy flight paths in selected date range", width=300)
    # callback to copy the files in the selected date range
    date_slider_button_callback_args = {
        "scatter_source": scatter_source,
        "date_slider": date_slider,
    }
    if do_html:
        date_slider_button_callback_args["file_type_select"] = file_type_select
    date_slider_button.js_on_click(
        CustomJS(
            args=date_slider_button_callback_args,
            code="""
        const timestamps = scatter_source.data["timestamps"];
        const flight = scatter_source.data["flight"];
        let key;
        if (typeof file_type_select !== 'undefined'){
            if (file_type_select.value==='Data') {
                key = 'data_files';
            } else if (file_type_select.value==='HTML Plots') {
                key = 'html_files';
            }
        } else {
            key = 'data_files';
        }

        const paths = scatter_source.data[key];
        const start = date_slider.value[0];
        const end = date_slider.value[1] + 86400000; // make the last date inclusive
        let selected_paths = [];
        let selected_flight = [];
        for (let i=0;i<timestamps.length;i++){
            if (timestamps[i] >= start && timestamps[i]<= end){
                selected_paths.push(paths[i]);
                selected_flight.push(flight[i]);
            }
        }

        // sort the selected paths by target id
        const sortedIndices = selected_flight.map((value, index) => index)
                    .sort((i, j) => selected_flight[i] - selected_flight[j]);
        const sorted_paths = sortedIndices.map(i => selected_paths[i]);

        const combined_paths = sorted_paths.join('\\n');
        
        // Copy to clipboard
        navigator.clipboard.writeText(combined_paths).then(function() {
            alert(sorted_paths.length+` File paths copied to clipboard`);
        }, function(err) {
            console.error('Failed to copy text: ', err);
        });
        """,
        )
    )

    box_select = bokeh_plot.select_one(BoxSelectTool)
    box_select_callback_args = {
        "box_select": box_select,
        "taptool": taptool,
        "scatter_source": scatter_source,
        "poly_source": poly_source,
    }
    if do_html:
        box_select_callback_args["file_type_select"] = file_type_select
    bokeh_plot.js_on_event(
        "selectiongeometry",
        CustomJS(
            args=box_select_callback_args,
            code="""
        if (poly_source.selected.indices.length>0){
            const selected = poly_source.selected.indices;
            const poly_flight = poly_source.data["flight"];
            const targets_in_box = selected.map(i => poly_flight[i]);
            let key;
            if (typeof file_type_select !== 'undefined'){
                if (file_type_select.value==='Data') {
                    key = 'data_files';
                } else if (file_type_select.value==='HTML Plots') {
                    key = 'html_files';
                }
            } else {
                key = 'data_files';
            }
            const paths = scatter_source.data[key];
            const scatter_flight = scatter_source.data["flight"];

            let selected_paths = [];
            let selected_flight = [];
            for (let i=0;i<paths.length;i++){
                if (targets_in_box.includes(scatter_flight[i])){
                    selected_paths.push(paths[i]);
                    selected_flight.push(scatter_flight[i]);
                }
            }

            // sort the selected paths by target id
            const sortedIndices = selected_flight.map((value, index) => index)
                        .sort((i, j) => selected_flight[i] - selected_flight[j]);
            const sorted_paths = sortedIndices.map(i => selected_paths[i]);

            const combined_paths = sorted_paths.join('\\n');
            
            // Copy to clipboard
            navigator.clipboard.writeText(combined_paths).then(function() {
                alert(sorted_paths.length+` File paths copied to clipboard for `+targets_in_box.length+` flights`);
            }, function(err) {
                console.error('Failed to copy text: ', err);
            });
            poly_source.selected.indices = [];
            poly_source.change.emit();
        }
    """,
        ),
    )

    inp_callback = CustomJS(args=inp_callback_args, code=inp_callback_code)
    inp.js_on_change("value", inp_callback)

    alpha_button = Button(label="Zero Polygon Alpha", button_type="warning")
    # callback to change the alpha of the polygons
    alpha_button_callback = CustomJS(
        args={
            "poly_source": poly_source,
            "alpha_button": alpha_button,
            "poly_renderer": poly_renderer,
        },
        code="""
        var data = poly_source.data;

        if (alpha_button.button_type==="warning") {
            alpha_button.button_type = "primary";
            alpha_button.label = "Raise Polygon Alpha";
            data['fill_alpha'] = Array(data["fill_alpha"].length).fill(0);
        } else {
            alpha_button.button_type = "warning";
            alpha_button.label = "Zero Polygon Alpha";
            data['fill_alpha'] = data['default_alpha'].slice();
        }
        poly_renderer.glyph.fill_alpha = {field: 'fill_alpha'};

        poly_source.change.emit();
    """,
    )
    alpha_button.js_on_click(alpha_button_callback)

    creation_time_div = Div(
        text=f"<font size=2 color='teal'><b>Last update</b></font>: {pd.Timestamp.strftime(pd.Timestamp.utcnow(),'%Y-%m-%d %H:%M UTC')}",
        width=300,
    )

    notes = Div(text="Note 'flight' here counts the number of priority maps.<br>e.g. MX064 has two priority maps so counts for 2 flights.")

    if do_html:
        layout = Row(
            bokeh_plot,
            Column(
                inp,
                notes,
                fig,
                date_slider,
                date_slider_info_div,
                date_slider_button,
                file_type_select,
                alpha_button,
                creation_time_div,
            ),
        )
    elif not is_L2:
        layout = Row(
            bokeh_plot,
            Column(
                inp,
                notes,
                fig,
                date_slider,
                date_slider_info_div,
                date_slider_button,
                file_type_select,
                alpha_button,
                creation_time_div,
            ),
        )        
    else:
        layout = Row(
            bokeh_plot,
            Column(
                inp,
                fig,
                date_slider,
                date_slider_info_div,
                date_slider_button,
                alpha_button,
                creation_time_div,
            ),
        )       
    layout.sizing_mode = "scale_both"

    if write:
        with open(out_file, "w") as out:
            out.write(file_html(layout, CDN, "MethaneAIR flights", suppress_callback_warning=True))

    return layout


def make_mair_targets_map_tabs(
    polygon_file: str,
    out_file: str,
    title: str = "MethaneAIR flights",
    tab_title: Optional[list[str]] = None,
    file_list: Optional[list[Optional[str]]] = None,
    html_bucket: Optional[list[Optional[str]]] = None,
):
    """
    Read the list of targets from the polygon_file geojson file

    Inputs:
        polygon_file (str): input geojson file with all the target polygons
        out_file (str): full path to the output html file
        title (str): map title
        tab_title (Optional[list[str]]): name of the tabs in the final layout
        file_list (Optional[str]): full path to list of data bucket files
    """

    tabs = [
        TabPanel(
            child=make_mair_targets_map(
                polygon_file=polygon_file,
                out_file=out_file,
                title=title,
                file_list=file_list[i],
                html_bucket=html_bucket[i],
                write=False,
            ),
            title=v,
        )
        for i, v in enumerate(tab_title)
    ]

    layout = Tabs(
        tabs=tabs,
        stylesheets=[
            InlineStyleSheet(
                css="""
                    div.bk-tab {
                        background-color: lightcyan;
                        font-weight: bold;
                        border-color: darkgray;
                        color: teal;
                    }
                    div.bk-tab.bk-active {
                        background-color: lightblue;
                        border-color: teal;
                        color: teal;
                        font-weight: bold;
                    }
                    div.bk-header {
                        border-bottom: 0px !important;
                    }
                    """
            )
        ],
    )

    layout.sizing_mode = "scale_both"

    with open(out_file, "w") as out:
        out.write(file_html(layout, CDN, "MethaneAIR flights", suppress_callback_warning=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_file", help="full path to the input geojson file")
    parser.add_argument("out_file", help="full path to the output html file")
    parser.add_argument("-t", "--title", default="MethaneAIR flights", help="Map title")
    parser.add_argument(
        "-f",
        "--file-list",
        nargs="+",
        required=True,
        type=str,
        help="full path to a list of bucket path, the flights file paths will be added to the flights hover tooltips",
    )
    parser.add_argument(
        "--tab-title",
        nargs="+",
        help="when generating multi-level maps under tabs, sets the tab titles",
    )
    parser.add_argument(
        "--html-bucket",
        default=[None],
        nargs="+",
        type=none_or_str,
        help="full path to the bucket where the flight html are stored",
    )
    args = parser.parse_args()

    if len(args.file_list) == 1:
        _ = make_mair_targets_map(
            polygon_file=args.in_file,
            out_file=args.out_file,
            title=args.title[0],
            file_list=args.file_list[0],
            html_bucket=args.html_bucket[0],
        )
    else:
        if (
            len(
                set(
                    [
                        len(i)
                        for i in [
                            args.tab_title,
                            args.file_list,
                            args.html_bucket,
                        ]
                    ]
                )
            )
            != 1
        ):
            raise Exception("List arguments are not all the same length")
        make_mair_targets_map_tabs(
            polygon_file=args.in_file,
            out_file=args.out_file,
            title=args.title,
            tab_title=args.tab_title,
            file_list=args.file_list,
            html_bucket=args.html_bucket,
        )


if __name__ == "__main__":
    main()
