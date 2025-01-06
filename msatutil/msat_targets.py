import argparse
import holoviews as hv
import geoviews as gv
from bokeh.models import NumericInput, CustomJS, Row
from bokeh.embed import file_html
from bokeh.resources import CDN
import geopandas as gpd

gv.extension("bokeh")


def make_msat_targets_map(infile: str, outfile: str, title: str = "MethaneSAT targets"):
    """
    Read the list of targets from the infile geojson file

    Inputs:
            infile (str): input geojson file with all the target polygons
            title (str): map title
    """

    gdf = gpd.read_file(infile)

    gdf.loc[gdf["type"] == "Oil And Gas", "default_color"] = "purple"
    gdf.loc[gdf["type"] == "Agriculture", "default_color"] = "green"
    gdf.loc[gdf["type"] == "CalVal", "default_color"] = "yellow"

    gdf["color"] = gdf["default_color"].copy()

    base_map = gv.tile_sources.EsriImagery()
    polygons = gv.Polygons(gdf, vdims=["id", "name", "type", "color", "default_color"])
    plot = base_map * polygons.opts(
        hv.opts.Polygons(
            tools=["hover", "fullscreen"],
            active_tools=["pan", "wheel_zoom"],
            color="color",
            alpha=0.7,
            line_color="black",
            width=1100,
            height=800,
            hover_tooltips=[("id", "@id"), ("Name", "@name"), ("Type", "@type")],
            title=title,
        )
    )
    bokeh_plot = hv.render(plot, backend="bokeh")

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
    args = parser.parse_args()

    make_msat_targets_map(args.infile, args.outfile, args.title)


if __name__ == "__main__":
    main()
