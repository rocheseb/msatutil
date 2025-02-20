import argparse
import holoviews as hv
import geoviews as gv
from bokeh.models import NumericInput, CustomJS, Row, TapTool
from bokeh.embed import file_html
from bokeh.resources import CDN
import geopandas as gpd
from typing import Optional
from pathlib import Path

gv.extension("bokeh")


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

    vdims = ["id", "name", "type", "color", "default_color"]
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
        for t in td:
            gdf.loc[gdf["id"] == t, "collections"] = "\n".join(
                [td[t][c][p] for c in td[t] for p in td[t][c]]
            )
            gdf.loc[gdf["id"] == t, "ncollections"] = len(list(td[t].keys()))

    base_map = gv.tile_sources.EsriImagery()
    polygons = gv.Polygons(gdf, vdims=vdims)
    plot = base_map * polygons.opts(
        hv.opts.Polygons(
            tools=["hover", "fullscreen", "tap"],
            active_tools=["pan", "wheel_zoom", "tap"],
            color="color",
            alpha=0.7,
            line_color="black",
            width=1100,
            height=800,
            hover_tooltips=hover_tooltips,
            title=title,
        )
    )
    bokeh_plot = hv.render(plot, backend="bokeh")

    if file_list is not None:
        taptool = bokeh_plot.select_one(TapTool)

        taptool.callback = CustomJS(
            args=dict(source=bokeh_plot.renderers[-1].data_source),
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

    inp = NumericInput(value=None, title="Highlight this target id:")

    code = """
    var data = source.data;
    var ids = Array.from(data['id']);
    var selected_id = cb_obj.value;
    var color = data['color'];
    var default_color = data['default_color'];
    var xs = data['xs'];
    var ys = data['ys'];
    var name = data['name'];
    var type = data['type'];

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
        data['id'] = new Int32Array(ids);
        default_color.push(default_color.splice(hid,1)[0]);
        for (var i=0;i<color.length;i++){
            color[i] = default_color[i];
        }
        color[color.length-1] = 'red';
    } else {
        color[color.length-1] = default_color[default_color.length-1];
    }
    source.change.emit()
    """

    callback = CustomJS(args=dict(source=bokeh_plot.renderers[1].data_source), code=code)
    inp.js_on_change("value", callback)

    layout = Row(bokeh_plot, inp)

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
