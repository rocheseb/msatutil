import os
import argparse
import re
from datetime import datetime
import holoviews as hv
import geoviews as gv
from bokeh.models import (
    NumericInput,
    CustomJS,
    Row,
    Column,
    TapTool,
    Div,
    ColumnDataSource,
    HoverTool,
    Button,
    DateRangeSlider,
    Select,
)
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.plotting import figure
import pandas as pd
import geopandas as gpd
from typing import Optional
from pathlib import Path

gv.extension("bokeh")


def extract_timestamp(text: str) -> Optional[str]:
    """
    Get a YYYYMMDDTHHMMSS timestamp from a string

    Inputs:
        text (str): input text to match
    Outputs:
        (Optional[str]): the timestamp (or None)
    """
    match = re.search(r"(\d{8}T\d{6})", text)
    time_fmt = "%Y%m%dT%H%M%S"
    if match:
        return pd.to_datetime(datetime.strptime(match.group(1), time_fmt), format=time_fmt)
    return None


def derive_L2_qaqc_path(l2pp_file_path: str) -> str:
    """
    Return the qaqc path corresponding to a L2 post-processed file

    Input:
        - l2pp_file_path (str): path to the L2 post-processed file
    Ouputs:
        - qaqc_file_path (str): path to the L2 qaqc file
    """
    l2pp_file_path = Path(l2pp_file_path)
    qaqc_file_path = (
        l2pp_file_path.parent
        / "qaqc"
        / l2pp_file_path.name.replace("_L2_", "_L2_QAQC_Plots_").replace(".nc", ".html")
    )

    return str(qaqc_file_path)


def get_target_dict(file_list: str) -> dict:
    """
    Parse a list of MSAT bucket paths and store them in a dictionary by target/collect/processing_id

    Inputs:
        file_list (str): full path to input file listing MSAT bucket paths
    Outputs:
        d (dict): dictionary of targets by target/collect/processing_id
    """
    with open(file_list, "r") as fin:
        file_list = [Path(i) for i in fin.readlines()]
    d = {}
    for i in file_list:
        t = int(i.parts[3].strip("t"))
        c = i.parts[5].strip("c")
        p = int(i.parts[6].strip("p"))
        if t not in d:
            d[t] = {}
        if c not in d[t]:
            d[t][c] = {}
        # only keep the highest pid
        # if a processing exists, overwrite with the higher pid
        if d[t][c] != {}:
            old_p = list(d[t][c].keys())[0]
            if p < old_p:
                continue
        d[t][c][p] = str(i).rstrip()

    return d


def make_msat_targets_map(
    infile: str,
    outfile: str,
    title: str = "MethaneSAT targets",
    file_list: Optional[str] = None,
):
    """
    Read the list of targets from the infile geojson file

    Inputs:
            infile (str): input geojson file with all the target polygons
            outfile (str): full path to the output html file
            title (str): map title
            file_list (Optional[str]): full path to list of data bucket files
    """
    gdf = gpd.read_file(infile)

    gdf.loc[gdf["type"] == "Oil And Gas", "default_color"] = "purple"
    gdf.loc[gdf["type"] == "Agriculture", "default_color"] = "green"
    gdf.loc[gdf["type"] == "CalVal", "default_color"] = "deepskyblue"

    gdf["fill_color"] = gdf["default_color"].copy()
    gdf["line_color"] = gdf["default_color"]
    gdf["fill_alpha"] = 0.7
    gdf["default_alpha"] = gdf["fill_alpha"]

    vdims = [
        "id",
        "name",
        "type",
        "fill_color",
        "default_color",
        "line_color",
        "fill_alpha",
        "default_alpha",
    ]
    hover_tooltips = [
        ("id", "@id"),
        ("Name", "@name"),
        ("Type", "@type"),
    ]

    if file_list is not None:
        td = get_target_dict(file_list)
        is_L2 = "_L2_" in list(next(iter(next(iter(td.values())).values())).values())[0]
        vdims += ["ncollections", "collections"]
        hover_tooltips += [("# Collects", "@ncollections")]
        gdf["ncollections"] = 0
        gdf["collections"] = ""
        scatter_df_columns = ["File", "id"]
        if is_L2:
            gdf["qaqc_files"] = ""
            vdims += ["qaqc_files"]
            scatter_df_columns += ["qaqc_file"]
        scatter_df = pd.DataFrame(columns=scatter_df_columns)
        for t in td:
            gdf.loc[gdf["id"] == t, "collections"] = "\n".join(
                [td[t][c][p] for c in td[t] for p in td[t][c]]
            )
            gdf.loc[gdf["id"] == t, "ncollections"] = len(list(td[t].keys()))
            if is_L2:
                gdf.loc[gdf["id"] == t, "qaqc_files"] = "\n".join(
                    [derive_L2_qaqc_path(td[t][c][p]) for c in td[t] for p in td[t][c]]
                )
            for c in td[t]:
                for p in td[t][c]:
                    if is_L2:
                        scatter_df.loc[len(scatter_df)] = [
                            td[t][c][p],
                            t,
                            derive_L2_qaqc_path(td[t][c][p]),
                        ]
                    else:
                        scatter_df.loc[len(scatter_df)] = [td[t][c][p], t]
        gdf.loc[gdf["ncollections"] == 0, "default_color"] = "lightgray"
        gdf.loc[gdf["ncollections"] == 0, "fill_color"] = "lightgray"
        gdf.loc[gdf["ncollections"] == 0, "fill_alpha"] = 0.5
        gdf.loc[gdf["ncollections"] == 0, "default_alpha"] = 0.5

        scatter_df["timestamps"] = scatter_df["File"].apply(extract_timestamp)
        scatter_df["counts"] = 1
        scatter_df.loc[pd.isna(scatter_df["timestamps"]), "counts"] = 0
        scatter_df = scatter_df.sort_values(by=["timestamps"]).reset_index()
        scatter_df["cumulcounts"] = scatter_df["counts"].cumsum()
        scatter_df["color"] = "#1f77b4"
        scatter_df["size"] = 4

    base_map = gv.tile_sources.EsriImagery()
    polygons = gv.Polygons(gdf, vdims=vdims)
    plot = base_map * polygons.opts(
        hv.opts.Polygons(
            tools=["hover", "fullscreen", "tap"],
            active_tools=["pan", "wheel_zoom", "tap"],
            fill_color="fill_color",
            fill_alpha="fill_alpha",
            line_color="line_color",
            width=1100,
            height=800,
            hover_tooltips=hover_tooltips,
            title=title,
        )
    )
    bokeh_plot = hv.render(plot, backend="bokeh")

    poly_source = bokeh_plot.renderers[1].data_source

    inp = NumericInput(value=None, title="Highlight this target id:")

    # callback to highlight the polygon corresponding to the target in the input widget
    inp_callback_code = """
    var data = poly_source.data;
    var ids = Array.from(data['id']);
    var selected_id = cb_obj.value;
    var color = data['fill_color'];
    var default_color = data['default_color'];
    var line_color = data['line_color'];
    var xs = data['xs'];
    var ys = data['ys'];
    var name = data['name'];
    var type = data['type'];
    var collections = data['collections'];
    var ncollections = Array.from(data['ncollections']);
    var alpha = Array.from(data['fill_alpha']);
    if ('qaqc_files' in data){
        var qaqc_files = data['qaqc_files'];
    }

    var hid = -1;
    for (var i=0;i<ids.length;i++){
        if (ids[i]==selected_id) {
            hid=i;
            break;
        }
    }

    if (hid>=0){
        name.push(name.splice(hid,1)[0]);
        type.push(type.splice(hid,1)[0]);
        xs.push(xs.splice(hid,1)[0]);
        ys.push(ys.splice(hid,1)[0]);
        ids.push(ids.splice(hid,1)[0]);
        collections.push(collections.splice(hid,1)[0]);
        if ('qaqc_files' in data) qaqc_files.push(qaqc_files.splice(hid,1)[0]);
        ncollections.push(ncollections.splice(hid,1)[0]);
        alpha.push(alpha.splice(hid,1)[0]);
        data['ncollections'] = new Int32Array(ncollections);
        data['fill_alpha'] = new Float32Array(alpha);
        data['id'] = new Int32Array(ids);
        default_color.push(default_color.splice(hid,1)[0]);
        line_color.push(line_color.splice(hid,1)[0]);
        for (var i=0;i<color.length;i++){
            color[i] = default_color[i];
        }
        color[color.length-1] = 'red';
    } else {
        color[color.length-1] = default_color[default_color.length-1];
    }
    poly_source.change.emit();
    """

    inp_callback_args = {"poly_source": poly_source}

    poly_hover = bokeh_plot.select_one(HoverTool)
    poly_hover.callback = CustomJS(
        args={"inp": inp, "poly_source": poly_source},
        code="""
        const selected = cb_data["index"].indices;

        if (selected.length>0) {
            const index = selected[selected.length-1];
            inp.value = poly_source.data["id"][index];
        }
        """,
    )

    if file_list is not None:
        taptool = bokeh_plot.select_one(TapTool)
        taptool_callback_args = {"poly_source": poly_source}
        if is_L2:
            file_type_select = Select(
                options=["L2-pp", "QAQC plots"], value="L2-pp", title="File type"
            )
            taptool_callback_args["file_type_select"] = file_type_select
        # callback to copy the corresponding files when clicking on a polygon
        taptool.callback = CustomJS(
            args=taptool_callback_args,
            code="""
            const selected_indices = poly_source.selected.indices;
            let key;
            if (typeof file_type_select !== 'undefined'){
                if (file_type_select.value==='L2-pp') {
                    key = 'collections';
                } else {
                    key = 'qaqc_files';
                }
            } else {
                key = 'collections';
            }

            if (selected_indices.length > 0) {
                let all_collections = [];
                
                // Loop over all selected polygons
                for (let i = 0; i < selected_indices.length; i++) {
                    const idx = selected_indices[i];
                    const collections = poly_source.data[key][idx];
                    all_collections.push(collections);
                }
                
                // Join file paths with newlines
                const combined_paths = all_collections.join('\\n');
                
                // Copy to clipboard
                navigator.clipboard.writeText(combined_paths).then(function() {
                    alert('File paths copied to clipboard:\\n' + combined_paths);
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
            title=f"{scatter_df['counts'].sum()} collects over {len(list(td.keys()))} targets",
            width=350,
            height=300,
            x_axis_type="datetime",
            tools=["pan,wheel_zoom,box_zoom,reset,tap"],
        )
        scatter = fig.scatter(
            "timestamps", "cumulcounts", source=scatter_source, color="color", size="size"
        )
        scatter_hover = HoverTool(tooltips=None, renderers=[scatter])
        fig.add_tools(scatter_hover)
        scatter_taptool = fig.select_one(TapTool)
        scatter_taptool_callback_args = {"scatter_source": scatter_source}
        if is_L2:
            scatter_taptool_callback_args["file_type_select"] = file_type_select
        # callback to copy the corresponding file path when clicking on the scatter points
        scatter_taptool.callback = CustomJS(
            args=scatter_taptool_callback_args,
            code="""
            const selected = scatter_source.selected.indices;
            let key;
            if (typeof file_type_select !== 'undefined'){
                if (file_type_select.value==='L2-pp') {
                    key = 'File';
                } else {
                    key = 'qaqc_file';
                }
            } else {
                key = 'File';
            }
            const file_path = scatter_source.data[key][selected[selected.length-1]];

            navigator.clipboard.writeText(file_path).then(function() {
                    alert('File path copied to clipboard:\\n' + file_path);
                }, function(err) {
                    console.error('Failed to copy text: ', err);
                });;

            scatter_source.selected.indices = [];
            scatter_source.change.emit()

            """,
        )

        # CustomJS Callback to Highlight Polygons and scatter points on scatter Hover
        # this does it by changing the value of the input widget, which triggers the input callback
        # that will also highlight the corresponding scatter points
        scatter_hover.callback = CustomJS(
            args=dict(scatter_source=scatter_source, inp=inp),
            code="""
            // Get hovered file from scatter
            const hovered_indices = cb_data["index"].indices;
            const scatter_data = scatter_source.data;

            if (hovered_indices.length>0){
                const hovered_index = hovered_indices[hovered_indices.length-1];
                const target_id = scatter_source.data["id"][hovered_index];
                inp.value = target_id;
            } 

            """,
        )

        inp_callback_args["scatter_source"] = scatter_source
        # callback to highlight scatter points corresponding to the target in the input widget
        inp_callback_code += """
        var scatter_colors = scatter_source.data["color"];
        var scatter_size = scatter_source.data["size"];
        var ids = scatter_source.data["id"];

        for (let i = 0; i < ids.length; i++) {
            scatter_colors[i] = (ids[i] === selected_id) ? "red" : "#1f77b4";
            scatter_size[i] = (ids[i] === selected_id) ? 8 : 4;
        }

        scatter_source.change.emit();        
        """

        creation_time_div = Div(
            text=f"Last update: {pd.Timestamp.strftime(pd.Timestamp.utcnow(),'%Y-%m-%d %H:%M UTC')}",
            width=300,
        )

        start = scatter_df["timestamps"][0].date()
        end = scatter_df["timestamps"][len(scatter_df.index) - 1].date()
        date_slider = DateRangeSlider(
            value=(start, end),
            start=start,
            end=end,
            width=300,
        )
        date_slider_info_div = Div(
            text=f"{scatter_df['counts'].sum()} collects in selected date range", width=400
        )
        # callback to update a text Div with the number of collects in the selected date range
        date_slider_callback = CustomJS(
            args={"info_div": date_slider_info_div, "scatter_source": scatter_source},
            code="""
            const timestamps = scatter_source.data["timestamps"];
            const start = cb_obj.value[0];
            const end = cb_obj.value[1] + 86400000; // make the last date inclusive
            const count = timestamps.filter(x => x >= start && x <= end).length;
            info_div.text = `${count} collects in selected date range`;
            """,
        )
        date_slider.js_on_change("value_throttled", date_slider_callback)
        date_slider_button = Button(label="Copy collection paths in selected date range", width=300)
        # callback to copy the files in the selected date range
        date_slider_button_callback_args = {
            "scatter_source": scatter_source,
            "date_slider": date_slider,
        }
        if is_L2:
            date_slider_button_callback_args["file_type_select"] = file_type_select
        date_slider_button.js_on_click(
            CustomJS(
                args=date_slider_button_callback_args,
                code="""
            const timestamps = scatter_source.data["timestamps"];
            const ids = scatter_source.data["id"];
            let key;
            if (typeof file_type_select !== 'undefined'){
                if (file_type_select.value==='L2-pp') {
                    key = 'File';
                } else {
                    key = 'qaqc_file';
                }
            } else {
                key = 'File';
            }

            const paths = scatter_source.data[key];
            const start = date_slider.value[0];
            const end = date_slider.value[1] + 86400000; // make the last date inclusive
            let selected_paths = [];
            let selected_ids = [];
            for (let i=0;i<timestamps.length;i++){
                if (timestamps[i] >= start && timestamps[i]<= end){
                    selected_paths.push(paths[i]);
                    selected_ids.push(ids[i]);
                }
            }

            // sort the selected paths by target id
            const sortedIndices = selected_ids.map((value, index) => index)
                        .sort((i, j) => selected_ids[i] - selected_ids[j]);
            const sorted_paths = sortedIndices.map(i => selected_paths[i]);

            const combined_paths = sorted_paths.join('\\n');
            
            // Copy to clipboard
            navigator.clipboard.writeText(combined_paths).then(function() {
                alert(`File paths copied to clipboard`);
            }, function(err) {
                console.error('Failed to copy text: ', err);
            });
            """,
            )
        )
    # enf of if file_list is not None

    inp_callback = CustomJS(args=inp_callback_args, code=inp_callback_code)
    inp.js_on_change("value", inp_callback)

    legend_div = Div(
        text="""
    <div style="padding:10px; width:150px;">

      <div style="display:flex; align-items:center; margin-top:5px;">
        <div style="width:15px; height:15px; background-color:purple; margin-right:5px;"></div>
        <span>Oil & Gas</span>
      </div>
      <div style="display:flex; align-items:center; margin-top:5px;">
        <div style="width:15px; height:15px; background-color:green; margin-right:5px;"></div>
        <span>Agriculture</span>
      </div>
      <div style="display:flex; align-items:center; margin-top:5px;">
        <div style="width:15px; height:15px; background-color:deepskyblue; margin-right:5px;"></div>
        <span>Cal/Val</span>
      </div>
      <div style="display:flex; align-items:center; margin-top:5px;">
        <div style="width:15px; height:15px; background-color:lightgray; margin-right:5px;"></div>
        <span>No collects</span>
      </div>
    </div>
    """
    )

    alpha_button = Button(label="Zero Polygon Alpha", button_type="warning")
    # callback to change the alpha of the polygons
    alpha_button_callback = CustomJS(
        args={"poly_source": poly_source, "alpha_button": alpha_button},
        code="""
        var alpha = poly_source.data["fill_alpha"];
        const default_alpha = poly_source.data["default_alpha"];

        if (alpha_button.button_type==="warning") {
            alpha_button.button_type = "primary";
            alpha_button.label = "Raise Polygon Alpha";
            alpha.fill(0);
        } else {
            alpha_button.button_type = "warning";
            alpha_button.label = "Zero Polygon Alpha";
            for (let i = 0; i < alpha.length; i++) {
                alpha[i] = default_alpha[i];
            }
        }

        poly_source.change.emit();
    """,
    )
    alpha_button.js_on_click(alpha_button_callback)

    if file_list is not None:
        if is_L2:
            layout = Row(
                bokeh_plot,
                Column(
                    inp,
                    legend_div,
                    fig,
                    date_slider,
                    date_slider_info_div,
                    date_slider_button,
                    file_type_select,
                    creation_time_div,
                    alpha_button,
                ),
            )
        else:
            layout = Row(
                bokeh_plot,
                Column(
                    inp,
                    legend_div,
                    fig,
                    date_slider,
                    date_slider_info_div,
                    date_slider_button,
                    creation_time_div,
                    alpha_button,
                ),
            )
    else:
        layout = Row(bokeh_plot, Column(inp, legend_div, alpha_button))

    with open(outfile, "w") as out:
        out.write(file_html(layout, CDN, "MethaneSAT targets", suppress_callback_warning=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="full path to the input geojson file")
    parser.add_argument("outfile", help="full path to the output html file")
    parser.add_argument("-t", "--title", default="MethaneSAT targets", help="Map title")
    parser.add_argument(
        "-f",
        "--file-list",
        default=None,
        help="full path to a list of bucket path, the collections file paths will be added to the targets hover tooltips",
    )
    args = parser.parse_args()

    make_msat_targets_map(args.infile, args.outfile, args.title, args.file_list)


if __name__ == "__main__":
    main()
