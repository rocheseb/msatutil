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
    gdf.loc[gdf["type"] == "CalVal", "default_color"] = "yellow"

    gdf["color"] = gdf["default_color"].copy()
    gdf["line_color"] = "black"
    gdf["alpha"] = 0.7

    vdims = ["id", "name", "type", "color", "default_color", "line_color", "alpha"]
    hover_tooltips = [
        ("id", "@id"),
        ("Name", "@name"),
        ("Type", "@type"),
    ]

    if file_list is not None:
        td = get_target_dict(file_list)
        vdims += ["ncollections", "collections"]
        hover_tooltips += [("# Collects", "@ncollections")]
        gdf["collections"] = ""
        gdf["ncollections"] = 0
        scatter_df = pd.DataFrame(columns=["File", "id"])
        for t in td:
            gdf.loc[gdf["id"] == t, "collections"] = "\n".join(
                [td[t][c][p] for c in td[t] for p in td[t][c]]
            )
            gdf.loc[gdf["id"] == t, "ncollections"] = len(list(td[t].keys()))
            for c in td[t]:
                for p in td[t][c]:
                    scatter_df.loc[len(scatter_df)] = [td[t][c][p], t]
        gdf.loc[gdf["ncollections"] == 0, "line_color"] = gdf.loc[
            gdf["ncollections"] == 0, "default_color"
        ]
        gdf.loc[gdf["ncollections"] == 0, "default_color"] = "lightgray"
        gdf.loc[gdf["ncollections"] == 0, "color"] = "lightgray"
        gdf.loc[gdf["ncollections"] == 0, "alpha"] = 0.5

        scatter_df["timestamps"] = scatter_df["File"].apply(extract_timestamp)
        scatter_df["counts"] = 1
        scatter_df.loc[pd.isna(scatter_df["timestamps"]), "counts"] = 0
        scatter_df = scatter_df.sort_values(by=["timestamps"])
        scatter_df["cumulcounts"] = scatter_df["counts"].cumsum()
        scatter_df["color"] = "#1f77b4"
        scatter_df["size"] = 4

    base_map = gv.tile_sources.EsriImagery()
    polygons = gv.Polygons(gdf, vdims=vdims)
    plot = base_map * polygons.opts(
        hv.opts.Polygons(
            tools=["hover", "fullscreen", "tap"],
            active_tools=["pan", "wheel_zoom", "tap"],
            color="color",
            alpha="alpha",
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

    inp_callback_code = """
    var data = poly_source.data;
    var ids = Array.from(data['id']);
    var selected_id = cb_obj.value;
    var color = data['color'];
    var default_color = data['default_color'];
    var line_color = data['line_color'];
    var xs = data['xs'];
    var ys = data['ys'];
    var name = data['name'];
    var type = data['type'];
    var collections = data['collections'];
    var ncollections = Array.from(data['ncollections']);
    var alpha = Array.from(data['alpha']);

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
        ncollections.push(ncollections.splice(hid,1)[0]);
        alpha.push(alpha.splice(hid,1)[0]);
        data['ncollections'] = new Int32Array(ncollections);
        data['alpha'] = new Float32Array(alpha);
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

        taptool.callback = CustomJS(
            args=dict(source=poly_source),
            code="""
            const selected_indices = source.selected.indices;
            if (selected_indices.length > 0) {
                let all_collections = [];
                
                // Loop over all selected polygons
                for (let i = 0; i < selected_indices.length; i++) {
                    const idx = selected_indices[i];
                    const collections = source.data['collections'][idx];
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
        hover = HoverTool(tooltips=None, renderers=[scatter])
        fig.add_tools(hover)
        scatter_taptool = fig.select_one(TapTool)

        scatter_taptool.callback = CustomJS(
            args={"scatter_source": scatter_source, "inp": inp},
            code="""
            const selected = scatter_source.selected.indices;
            const file_path = scatter_source.data["File"][selected[selected.length-1]];

            navigator.clipboard.writeText(file_path).then(function() {
                    alert('File path copied to clipboard:\\n' + file_path);
                }, function(err) {
                    console.error('Failed to copy text: ', err);
                });;

            inp.value = scatter_source.data["id"][selected[selected.length-1]];

            scatter_source.selected.indices = [];
            scatter_source.change.emit()

            """,
        )

        # CustomJS Callback to Highlight Polygons on Hover
        hover.callback = CustomJS(
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
        <div style="width:15px; height:15px; background-color:yellow; margin-right:5px;"></div>
        <span>Cal/Val</span>
      </div>
      <div style="display:flex; align-items:center; margin-top:5px;">
        <div style="width:15px; height:15px; background-color:lightgray; margin-right:5px;"></div>
        <span>No collects</span>
      </div>
    </div>
    """
    )

    if file_list is not None:
        layout = Row(bokeh_plot, Column(inp, legend_div, fig, creation_time_div))
    else:
        layout = Row(bokeh_plot, Column(inp, legend_div))

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
