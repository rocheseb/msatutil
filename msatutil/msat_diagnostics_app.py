import warnings

warnings.simplefilter("ignore")
import os
from functools import partial
from threading import Thread
from typing import Optional

import dask.array as da
import numpy as np
import zarr
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.events import Tap
from bokeh.io import output_file, output_notebook, show
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColorBar,
    ColumnDataSource,
    Div,
    InlineStyleSheet,
    LinearAxis,
    LinearColorMapper,
    Progress,
    Range1d,
    TextInput,
)
from bokeh.palettes import Greys256
from bokeh.plotting import figure
from bokeh.server.server import Server
from dask.diagnostics import ProgressBar
from netCDF4 import Dataset, Variable
from tornado.ioloop import IOLoop

from msatutil.msat_interface import msat_collection


class BaseBackend:
    """Abstract interface used by the Bokeh callbacks."""

    def load(self):
        raise NotImplementedError

    def get_mean_image(self):
        """Return 2D mean radiance image (rows, cols) as a NumPy array."""
        raise NotImplementedError

    def get_spectrum(self, i, j):
        """Return (wvl, rad, flag) 1D arrays for a single pixel."""
        raise NotImplementedError

    @property
    def shape(self):
        """Return (rows, cols, nspec)."""
        raise NotImplementedError


class NetcdfBackend(BaseBackend):
    def __init__(self, path, wvl_range, progress_chunk, progress_var, doc):
        self.path = path
        self.wvl_range = wvl_range
        self.progress_chunk = progress_chunk
        self.progress_var = progress_var
        self.doc = doc
        self.data = None  # dict of full NumPy arrays
        self._shape = None

    def load(self):
        # Your existing MSAT loading logic, slightly factored
        with msat_collection([self.path], use_dask=False) as c:
            if not c.is_l1:
                raise RuntimeError("Only L1 reader implemented")

            var_names = ["Band1/Radiance", "Band1/Wavelength", "Band1/RadianceFlag"]
            cols = c.dim_size_map["xtrack"]
            rows = c.dim_size_map["atrack"]
            nspec = c.dim_size_map["spectral_channel"]
            self._shape = (rows, cols, nspec)

            self.data = {}
            for idx, name in enumerate(var_names, start=1):
                var = c.dsets[c.ids[0]][name]

                def update_overall(i=idx, name=name, nvars=len(var_names)):
                    self.progress_var.max = nvars
                    self.progress_var.value = i
                    self.progress_var.label = f"Reading variable {i}/{nvars}: {name}"

                self.doc.add_next_tick_callback(update_overall)

                arr = read_var(var, progress=self.progress_chunk, doc=self.doc)
                if isinstance(arr, np.ma.MaskedArray):
                    arr = arr.filled()
                self.data[name] = arr

            # compute masked mean image just like before
            rad = self.data["Band1/Radiance"].copy()
            wvl = self.data["Band1/Wavelength"]
            flag = self.data["Band1/RadianceFlag"]

            sel = flag > 0
            if self.wvl_range is not None:
                wmin, wmax = self.wvl_range
                sel |= (wvl < wmin) | (wvl > wmax)
            rad[sel] = np.nan

            # chunked spectral mean using your helper for progress
            mean_img = spectral_channel_mean(rad, doc=self.doc, progress=self.progress_chunk)
            self.data["MeanRadiance"] = mean_img

    def get_mean_image(self):
        return self.data["MeanRadiance"]

    def get_spectrum(self, i, j):
        rad = self.data["Band1/Radiance"][i, j, :].copy()
        wvl = self.data["Band1/Wavelength"][i, j, :]
        flag = self.data["Band1/RadianceFlag"][i, j, :]
        return wvl, rad, flag

    @property
    def shape(self):
        return self._shape


class ZarrBackend(BaseBackend):
    def __init__(self, path):
        self.path = path
        self.store = None
        self.rad = None
        self.wvl = None
        self.flag = None
        self.mean = None

    def load(self):
        self.store = zarr.open_group(self.path, mode="r")
        self.rad = self.store["Band1/Radiance"]
        self.wvl = self.store["Band1/Wavelength"]
        self.flag = self.store["Band1/RadianceFlag"]
        self.mean = self.store["MeanRadiance"]

    def get_mean_image(self):
        return self.mean[:]  # 2D (rows, cols)

    def get_spectrum(self, i, j):
        # 1D slices; small and fast
        rad = self.rad[i, j, :].astype("float32")
        wvl = self.wvl[i, j, :].astype("float32")
        flag = self.flag[i, j, :]
        return wvl, rad, flag

    @property
    def shape(self):
        rows, cols, nspec = self.rad.shape
        return rows, cols, nspec


def infer_wvl_range_from_name(name: str):
    """
    Replicate the logic you use in bk_app to pick a wavelength range.
    """
    if "_CH4_" in name or "_CO2_" in name:
        return (1597.7, 1682.1)
    elif "_O2_" in name:
        return (1249.0, 1289.0)
    elif "_H2O_" in name:
        return (1290.0, 1295.0)
    else:
        # fall back to no range restriction
        return None


def read_var(
    var: Variable,
    along_track_chunk_size: int = 20,
    atrack_dim_id: int = 0,
    doc=None,
    progress=None,
):
    nalong = var.shape[atrack_dim_id]
    data = np.empty(var.shape, dtype=var.dtype)

    def init_bar():
        progress.max = nalong
        progress.value = 0

    if progress is not None:
        if doc is not None:
            doc.add_next_tick_callback(init_bar)
        else:
            init_bar()

    ichunk = 0
    while ichunk < nalong:
        iend = min(ichunk + along_track_chunk_size, nalong)

        slices = [slice(None)] * var.ndim
        slices[atrack_dim_id] = slice(ichunk, iend)
        data[tuple(slices)] = var[tuple(slices)]

        ichunk = iend

        if progress is not None and doc is not None:
            current = ichunk

            def _update(current=current):
                progress.value = current

            doc.add_next_tick_callback(_update)

    return data


def spectral_channel_mean(
    var: np.ndarray,
    along_track_chunk_size: int = 20,
    atrack_dim_id: int = 0,
    spectral_dim_id: int = 2,
    doc=None,
    progress=None,
):
    nalong = var.shape[atrack_dim_id]

    # build output shape: drop the spectral dimension
    out_shape = list(var.shape)
    spectral_dim_id_norm = spectral_dim_id % var.ndim
    out_shape.pop(spectral_dim_id_norm)
    data = np.empty(out_shape, dtype=var.dtype)

    def init_bar():
        progress.max = nalong
        progress.value = 0

    if progress is not None:
        if doc is not None:
            doc.add_next_tick_callback(init_bar)
        else:
            init_bar()

    ichunk = 0
    while ichunk < nalong:
        iend = min(ichunk + along_track_chunk_size, nalong)

        # input slices
        in_slices = [slice(None)] * var.ndim
        in_slices[atrack_dim_id] = slice(ichunk, iend)

        # output slices (same, but with spectral dim removed)
        out_slices = [slice(None)] * (var.ndim - 1)
        k = 0
        for ax in range(var.ndim):
            if ax == spectral_dim_id_norm:
                continue
            if ax == atrack_dim_id:
                out_slices[k] = slice(ichunk, iend)
            else:
                out_slices[k] = slice(None)
            k += 1

        data[tuple(out_slices)] = np.nanmean(var[tuple(in_slices)], axis=spectral_dim_id_norm)

        ichunk = iend

        if progress is not None and doc is not None:
            current = ichunk

            def _update(current=current):
                progress.value = current

            doc.add_next_tick_callback(_update)

    return data


def bk_app(doc):
    doc.title = "MethaneSAT diagnostics"
    wvl_range = None

    description = """
    Put the full file path to a MSAT file in the File A input.<br>
    Each spectral field is ~6-8 GB and data can take ~2-3 min to load when using netCDF files.<br>
    For near-instant load times first use msatutil.msat_to_zarr to convert the L1 netcdf file to zarr.<br>
    The File B input can be used to compare File A to a different version of the same file.
    """

    intro_div = Div(text=description, width=600, height=80)

    input_a = TextInput(value="", width=600, title="File A")
    input_b = TextInput(value="", width=600, title="File B")
    load_a = Button(label="Read File A", width=100)
    load_b = Button(label="Read File B", width=100)

    inputs = column(row(input_a, load_a), row(input_b, load_b))

    status_div = Div(text="", width=600, height=20)

    img_source = ColumnDataSource(data={"image": []})
    spec_source_a = ColumnDataSource(data={"x": [], "y": []})
    spec_source_b = ColumnDataSource(data={"x": [], "y": []})
    dif_source = ColumnDataSource(data={"x": [], "y": []})

    color_mapper = LinearColorMapper(palette=Greys256)

    p1 = figure(
        title="Mean Radiance (Click a Pixel)",
        height=300,
        sizing_mode="stretch_width",
        x_range=(0, 2048),
        y_range=(0, 500),
        tools="tap,pan,wheel_zoom,reset",
        y_axis_label="along-track index",
        x_axis_label="across-track index",
        min_border_left=100,
    )

    progress_var = Progress(
        value=0,
        min=0,
        max=10,
        visible=False,
        width=300,
        stylesheets=[
            InlineStyleSheet(css=".bk-bar {--active-fg: lightgreen; --active-bg: #eff0ee}")
        ],
    )  # track all variables
    progress_chunk = Progress(
        value=0,
        min=0,
        max=1,
        visible=False,
        width=300,
        stylesheets=[
            InlineStyleSheet(css=".bk-bar {--active-fg: lightblue; --active-bg: #eff0ee}")
        ],
    )  # track a given variable

    all_backends = {}

    def make_backend(path, wvl_range, progress_chunk, doc):
        # simple heuristic: directory ending with .zarr = Zarr backend
        if os.path.isdir(path) and path.endswith(".zarr"):
            return ZarrBackend(path)
        else:
            return NetcdfBackend(
                path,
                wvl_range=wvl_range,
                progress_chunk=progress_chunk,
                progress_var=progress_var,
                doc=doc,
            )

    def load_data(is_input_a: bool = False):
        nonlocal all_backends, wvl_range
        new = input_a.value if is_input_a else input_b.value
        key = "A" if is_input_a else "B"

        if not new:
            all_backends[key] = None
            return
        elif not os.path.exists(new):
            status_div.text = f"File not found: {new}"
            return
        elif not is_input_a and not input_a.value:
            status_div.text = "Need to load File A first"
            return

        name = os.path.basename(new)
        wvl_range = infer_wvl_range_from_name(name)

        # create backend
        backend = make_backend(new, wvl_range, progress_chunk, doc)
        all_backends[key] = backend

        # show progress bars etc. (unchanged)
        def show_bars_for_read():
            progress_var.visible = True
            progress_chunk.visible = True
            progress_var.min = 0
            progress_var.max = 1  # we just track 1 "load" step for backend
            progress_var.value = 0
            progress_var.label = "Reading variables..."
            progress_chunk.label = "Reading frames @{index} of @{total} (@{percent}%)"

        doc.add_next_tick_callback(show_bars_for_read)

        # do the heavy I/O
        backend.load()

        rows, cols, nspec = backend.shape
        spec_zeros = np.zeros(nspec)

        if is_input_a:

            def update_models_from_A():
                mean_img = backend.get_mean_image()
                color_mapper.low = np.nanmin(mean_img)
                color_mapper.high = np.nanmax(mean_img)

                img_source.data = {"image": [mean_img]}
                spec_source_a.data = {"x": spec_zeros, "y": spec_zeros}
                dif_source.data = {"x": spec_zeros, "y": spec_zeros}

                p1.x_range.start = 0
                p1.x_range.end = cols
                p1.x_range.bounds = (0, cols)
                p1.y_range.start = 0
                p1.y_range.end = rows
                p1.y_range.bounds = (0, rows)

            doc.add_next_tick_callback(update_models_from_A)
        else:

            def update_models_from_B():
                spec_source_b.data = {"x": spec_zeros, "y": spec_zeros}

            doc.add_next_tick_callback(update_models_from_B)

        def finish():
            progress_var.visible = False
            progress_chunk.visible = False

        doc.add_next_tick_callback(finish)

    def start_read(is_input_a: bool = False):
        Thread(target=partial(load_data, is_input_a=is_input_a), daemon=True).start()

    load_a.on_click(partial(start_read, is_input_a=True))
    load_b.on_click(start_read)

    # marker position (plot coords)
    marker_source = ColumnDataSource(data=dict(x=[], y=[]))
    # currently selected pixel indices
    pixel_source = ColumnDataSource(data=dict(i=[], j=[]))

    p1.image(
        image="image",
        x=0,
        y=0,
        dw=p1.x_range.end,
        dh=p1.y_range.end,
        source=img_source,
        color_mapper=color_mapper,
    )

    p1.scatter(
        "x",
        "y",
        marker="square",
        size=10,
        color="red",
        line_color=None,
        source=marker_source,
    )

    p2 = figure(
        title="Spectrum at Selected Pixel",
        height=300,
        sizing_mode="stretch_width",
        x_axis_label="Wavelength",
        y_axis_label="Radiance",
        min_border_left=100,
    )
    p2.line("x", "y", source=spec_source_a, line_width=2, color="firebrick", legend_label="A")
    p2.line("x", "y", source=spec_source_b, line_width=2, color="navy", legend_label="B")
    p2.legend.click_policy = "hide"

    p2.extra_x_ranges = {"index": Range1d(start=0, end=2048)}
    # second axis at the top, linked to "index" range
    index_axis = LinearAxis(x_range_name="index", axis_label="Band index")
    p2.add_layout(index_axis, "above")

    p3 = figure(
        title="A minus B",
        height=200,
        sizing_mode="stretch_width",
        x_axis_label="Wavelength",
        y_axis_label="Residual",
        min_border_left=100,
    )
    p3.line("x", "y", source=dif_source, line_width=2, color="black")

    boxes = CheckboxGroup(
        labels=["Show full spectral dimension", "Mask flag>0", "Normalize", "Relative residuals"],
        active=[1],
    )

    def compute_and_update_spectrum(i, j):
        backendA = all_backends.get("A")
        backendB = all_backends.get("B")

        A_loaded = backendA is not None
        B_loaded = backendB is not None
        if not A_loaded:
            return

        rows, cols, _ = backendA.shape
        if not (0 <= i < rows and 0 <= j < cols):
            return

        # get spectra from backend(s)
        wvl_a, rad_a, flag_a = backendA.get_spectrum(i, j)
        selection_a = np.zeros_like(wvl_a, dtype=bool)

        if B_loaded:
            wvl_b, rad_b, flag_b = backendB.get_spectrum(i, j)
            selection_b = np.zeros_like(wvl_b, dtype=bool)

        # Show full spectral dimension?
        if 0 not in boxes.active and wvl_range is not None:
            selection_a |= (wvl_a < wvl_range[0]) | (wvl_a > wvl_range[1])
            if B_loaded:
                selection_b |= (wvl_b < wvl_range[0]) | (wvl_b > wvl_range[1])

        # Mask flag>0?
        if 1 in boxes.active:
            selection_a |= flag_a != 0
            if B_loaded:
                selection_b |= flag_b != 0

        rad_a[selection_a] = np.nan
        if B_loaded:
            rad_b[selection_b] = np.nan

        # Normalize?
        if 2 in boxes.active:
            maxv = np.nanmax(rad_a)
            if np.isfinite(maxv) and maxv != 0:
                rad_a = rad_a / maxv
            if B_loaded:
                maxv = np.nanmax(rad_b)
                if np.isfinite(maxv) and maxv != 0:
                    rad_b = rad_b / maxv

        spec_source_a.data = {"x": wvl_a, "y": rad_a}
        if B_loaded:
            spec_source_b.data = {"x": wvl_b, "y": rad_b}
            if backendA.shape[2] == backendB.shape[2]:
                dif = rad_a - rad_b
                title = "A minus B"
                if 3 in boxes.active:
                    dif = 100 * (dif) / rad_b
                    title = "100*(A-B)/B"
                dif_source.data = {"x": wvl_a, "y": dif}
                p3.title.text = title
                p3.x_range = p2.x_range
            else:
                p3.title.text = "A and B don't have the same shape, no diff to show"

        p2.title.text = f"Spectrum at (along-track:{i}, across-track:{j})"
        marker_source.data = dict(x=[j + 0.5], y=[i + 0.5])

    def update_spectrum(event):
        j, i = int(event.x), int(event.y)
        # store indices
        pixel_source.data = dict(i=[i], j=[j])
        compute_and_update_spectrum(i, j)

    p1.on_event(Tap, update_spectrum)

    def boxes_changed(attr, old, new):
        # recompute for last selected pixel, if any
        if len(pixel_source.data["i"]) == 1:
            i = int(pixel_source.data["i"][0])
            j = int(pixel_source.data["j"][0])
            compute_and_update_spectrum(i, j)

    boxes.on_change("active", boxes_changed)

    doc.add_root(
        column(
            row(intro_div,inputs),
            row(boxes,status_div),
            progress_var,
            progress_chunk,
            p1,
            p2,
            p3,
            sizing_mode="stretch_width",
        )
    )


def main():
    """Launch the server and connect to it."""
    print("Preparing a bokeh application.")
    io_loop = IOLoop.current()
    bokeh_app = Application(FunctionHandler(bk_app))

    server = Server({"/": bokeh_app}, io_loop=io_loop)
    server.start()
    print("Opening Bokeh application on http://localhost:5006/")

    io_loop.add_callback(server.show, "/")
    io_loop.start()


if __name__ == "__main__":
    main()
