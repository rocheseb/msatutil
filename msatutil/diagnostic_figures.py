import sys
from datetime import datetime
from typing import Annotated, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import BoundaryNorm
from msatutil.make_hist import make_hist
from msatutil.msat_interface import msat_collection, set_clim


def _show_figure(fig):
    """Show a figure, respecting notebook/IPython and interactive mode."""
    in_notebook = "ipykernel" in sys.modules

    if in_notebook:
        # Jupyter: prefer IPython.display
        try:
            from IPython.display import display
        except ImportError:
            # Fallback to normal Matplotlib behavior
            if not plt.isinteractive():
                plt.show()
        else:
            display(fig)
            plt.close("all")
    else:
        # Non-notebook
        if not plt.isinteractive():
            # Blocking show in non-interactive mode
            plt.show()
        else:
            # Interactive mode: draw without blocking
            fig.canvas.draw_idle()


def compare_heatmaps(
    msat_1: msat_collection,
    msat_2: msat_collection,
    labs: Annotated[Sequence[str], 2],
    var: str,
    grp: Optional[str] = None,
    sv_var: Optional[str] = None,
    option: Optional[str] = None,
    option_axis_dim: Optional[str] = None,
    hist_nbins: int = 100,
    hist_xlim: Annotated[Sequence[float], 2] = None,
    vminmax: Optional[Annotated[Sequence[float], 2]] = None,
    latlon: bool = False,
    ratio: bool = False,
    save_path: Optional[str] = None,
    extra_id: Optional[int] = None,
    extra_id_dim: Optional[str] = None,
    data_in: Optional[np.ndarray] = None,
    lon_lim: Optional[Annotated[Sequence[float], 2]] = None,
    lat_lim: Optional[Annotated[Sequence[float], 2]] = None,
    res: float = 20,
    scale: float = 1.0,
    exp_fmt: bool = True,
) -> Tuple[plt.Figure, Annotated[Sequence[plt.Axes], 3], np.ndarray, np.ndarray]:
    """
    Make a 3-panel plot comparing a given variables between two sets of msat files by showing one heatmap for each and one histogram
    msat_1: first msat_interface.msat_collection object
    msat_2: second msat_interface.msat_collection object
    labs: list of 2 labels to use for the legend and subplots titles for msat_1 and msat_2, respectively
    var: partial variable name (will use msat_nc.fetch method to get it)
    grp: if givem use msat_nc.get_var instead of msat_nc.fetch and var must be the exact variable name
    sv_var: when the variable is one of APosterioriState or APrioriState, this selects for the state vector variable
    option: can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std')
    option_axis_dim: the axis along which the stat is applied (from the set of common dimension names: ["xtrack","atrack","xtrack_edge","atrack_edge","lev","lev_edge","corner","spectral_channel","xmx","nsubx"])
    hist_nbins: number of bins for the histogram
    hist_xlim: horizontal axis range for the histogram
    vminmax: [min,max] of the heatmap colorbars
    latlon: if True, make the plot on latitude/longitude instead of xtrack/atrack
    ratio: if True, divide the variable by its median
    save_path: full path to save the figure
    extra_id: when using 3D data, slice the 3rd dimension with this index
    extra_id_dim: name of the dimension along which extra_id will be selected
    data_in: list of the data to plot [x1,x2] corresponding to msat_1 and msat_2, if given, uses this data instead of reading the variable with pmesh_prep
    lon_lim: [min,max] longitudes for the gridding
    lat_lim: [min,max] latitudes for the gridding
    res: the resolution (in meters) of the grid with lon_lim and lat_lim are given
    scale: quantity to multiply the variable with (can be useful to avoid overflow in the standard deviation of column amounts)
    exp_fmt: if True, use .3e format for stats in the histogram legend. If false use .2f format
    """

    fig, ax = plt.subplot_mosaic(
        [["upper left", "right"], ["lower left", "right"]],
        gridspec_kw={"width_ratios": [2.5, 2]},
    )
    fig.set_size_inches(12, 10)
    plt.subplots_adjust(wspace=0.3)

    gridded = (lon_lim is not None) and (lat_lim is not None)

    ax["upper left"].set_title(labs[0])
    ax["lower left"].set_title(labs[1])
    if gridded:
        for i in ["upper left", "lower left"]:
            ax[i].set_ylim(lat_lim)
            ax[i].set_xlim(lon_lim)

    for curax in [ax["upper left"], ax["lower left"]]:
        if gridded:
            curax.set_ylabel("Latitude")
        else:
            if latlon:
                curax.set_ylabel("Latitude")
            else:
                curax.set_ylabel("along-track index")
    if gridded:
        ax["lower left"].set_xlabel("Longitude")
    else:
        if latlon:
            ax["lower left"].set_xlabel("Longitude")
        else:
            ax["lower left"].set_xlabel("across-track index")

    fig.suptitle(
        f"{datetime.strftime(msat_1.start_dates[0],'%Y%m%dT%H%M%S')} to {datetime.strftime(msat_1.end_dates[-1],'%Y%m%dT%H%M%S')}"
    )

    # make the heatmaps
    if data_in:
        x1, x2 = data_in
    else:
        if gridded:
            lon_grid1, lat_grid1, x1 = msat_1.grid_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                extra_id_dim=extra_id_dim,
                option=option,
                option_axis_dim=option_axis_dim,
                lon_lim=lon_lim,
                lat_lim=lat_lim,
                res=res,
            )
            lon_grid2, lat_grid2, x2 = msat_2.grid_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                extra_id_dim=extra_id_dim,
                option=option,
                option_axis_dim=option_axis_dim,
                lon_lim=lon_lim,
                lat_lim=lat_lim,
                res=res,
            )
        else:
            x1 = msat_1.pmesh_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                extra_id_dim=extra_id_dim,
                option=option,
                option_axis_dim=option_axis_dim,
            )
            x2 = msat_2.pmesh_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                extra_id_dim=extra_id_dim,
                option=option,
                option_axis_dim=option_axis_dim,
            )
            if latlon:
                lat_1 = msat_1.pmesh_prep("Latitude")
                lon_1 = msat_1.pmesh_prep("Longitude")
                lat_2 = msat_2.pmesh_prep("Latitude")
                lon_2 = msat_2.pmesh_prep("Longitude")

        x1 = x1 * scale
        x2 = x2 * scale

        if ratio:
            x1 = x1 / np.nanmedian(x1)
            x2 = x2 / np.nanmedian(x2)
        if msat_1.use_dask:
            x1 = x1.compute()
        if msat_2.use_dask:
            x2 = x2.compute()

    if vminmax is None:
        vmin = np.min([np.nanmin(x1), np.nanmin(x2)])
        vmax = np.max([np.nanmax(x1), np.nanmax(x2)])
        vminmax = [vmin, vmax]
    if gridded:
        m1 = ax["upper left"].pcolormesh(
            lon_grid1, lat_grid1, x1, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1]
        )
        m2 = ax["lower left"].pcolormesh(
            lon_grid2, lat_grid2, x2, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1]
        )
    else:
        x1 = x1[:, msat_1.valid_xtrack]
        x2 = x2[:, msat_2.valid_xtrack]
        if latlon:
            lon_1 = lon_1[:, msat_1.valid_xtrack]
            lat_1 = lat_1[:, msat_1.valid_xtrack]
            lon_2 = lon_2[:, msat_2.valid_xtrack]
            lat_2 = lat_2[:, msat_2.valid_xtrack]
            m1 = ax["upper left"].pcolormesh(
                lon_1, lat_1, x1, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1]
            )
            m2 = ax["lower left"].pcolormesh(
                lon_2, lat_2, x2, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1]
            )
        else:
            m1 = ax["upper left"].pcolormesh(x1, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1])
            m2 = ax["lower left"].pcolormesh(x2, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1])
    if var == "dp":
        lab = rf"$\Delta P$"
    elif sv_var:
        lab = sv_var
    else:
        lab = var

    units = msat_1.fetch_units(var)
    if units:
        lab = f"{lab} ({units})"
    print("units", units)

    plt.colorbar(m1, label=lab, ax=[ax["upper left"], ax["lower left"]])

    if hist_xlim is None:
        hist_xlim = [
            np.nanmin([np.nanmin(x1), np.nanmin(x2)]),
            np.nanmax([np.nanmax(x1), np.nanmax(x2)]),
        ]

    # make the histograms
    maxval1 = make_hist(
        ax["right"],
        x1[np.isfinite(x1)].flatten(),
        labs[0],
        "blue",
        rng=hist_xlim,
        nbins=hist_nbins,
        exp_fmt=exp_fmt,
    )
    maxval2 = make_hist(
        ax["right"],
        x2[np.isfinite(x2)].flatten(),
        labs[1],
        "red",
        rng=hist_xlim,
        nbins=hist_nbins,
        exp_fmt=exp_fmt,
    )

    ax["right"].set_ylim(0, 1.25 * np.max([maxval1, maxval2]))

    if save_path:
        fig.savefig(save_path)

    return fig, ax, x1, x2


def heatmap_hist(
    data,
    flags=None,
    heatmap_range=None,
    hist_range=None,
    exp_fmt=False,
    label="",
    scale=1.0,
    show=False,
    title="",
    cmap="viridis",
    bins=100,
):
    """
    Make a 2-panel figure with a heatmap and histograms
    The left panel is a 2D heatmap of `data`. The right panel contains one or
    more 1D histograms, each corresponding to a Boolean mask or index array in
    `flags`.

    Parameters
    ----------
    data : array-like
        2D array of values to plot as a heatmap. Also used as the source for
        histogram data (masked by `flags`).
    flags : dict[str, array-like] or None, optional
        Mapping from a label (string) to a mask or index array selecting
        elements from `data` for each histogram. Each `v` in `flags.items()`
        is passed as `data[v]`. If None, a single entry
        `{"all": np.ones(data.shape)}` is used, so one histogram is drawn
        over all elements of `data`.
    heatmap_range : (float, float) or None, optional
        Value range `(vmin, vmax)` for the heatmap color scaling. Passed to
        `pcolormesh` as `vmin` and `vmax`. If None, Matplotlib’s defaults are
        used (current code assumes a non-None value).
    hist_range : (float, float) or None, optional
        Value range `(min, max)` for the histograms. Passed to `make_hist`
        as `range`. If None, the histogram range is determined automatically.
    exp_fmt : bool, optional
        If True, `make_hist` is expected to format the histogram x-axis in
        exponential/scientific notation. Exact effect depends on `make_hist`.
    label : str, optional
        Label for the colorbar and the x-axis of the histograms (e.g. the
        physical quantity or units).
    scale : float, optional
        Scalar factor applied to `data` before plotting and histogramming
        (`data * scale`). Useful for unit conversions.
    show : bool, optional
        If True, the figure is displayed via `_show_figure(fig)` and then
        If False, the caller is responsible for showing or saving the figure.
    title : str, optional
        Figure-wide title, applied via `fig.suptitle(title)`.
    cmap : str or Colormap, optional
        Colormap used for the heatmap, passed to `pcolormesh` (default
        "viridis").
    bins : int or sequence, optional
        Number of histogram bins or bin edges, passed through to `make_hist`
        as `bins`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created Matplotlib figure.
    ax : ndarray of matplotlib.axes.Axes
        Array of two Axes: `ax[0]` for the heatmap, `ax[1]` for the histograms.
    """
    if flags is None:
        flags = {"all": np.ones(data.shape)}
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)
    m = ax[0].pcolormesh(data * scale, vmin=heatmap_range[0], vmax=heatmap_range[1], cmap=cmap)
    plt.colorbar(m, ax=ax[0], label=label)
    for k, v in flags.items():
        try:
            make_hist(
                ax=ax[1],
                x=data[v] * scale,
                bins=bins,
                range=hist_range,
                label=k,
                exp_fmt=exp_fmt,
            )
        except Exception:
            continue
    ax[1].set_xlabel(label)
    fig.suptitle(title)
    if show:
        _show_figure(fig)

    return fig, ax


def surface_type_flags(
    l2: msat_collection, show: bool = True, title: str = ""
) -> dict[str, np.ndarray]:
    """
    Use a O2 L2 file to derive surface type flags
    """
    alb = l2.pmesh_prep("refwvl", use_valid_xtrack=True).compute()
    dp = l2.pmesh_prep("dp", use_valid_xtrack=True).compute()
    agdof = l2.pmesh_prep("O2DG_ScaleFactor_DoFS", use_valid_xtrack=True).compute()

    minland = 0.6
    maxland = 0.9
    mindp = -20

    missing = np.isnan(alb)
    land = (agdof >= minland) & (agdof < maxland) & (dp > mindp)
    water = agdof >= maxland
    rest = (~water) & (~land) & (~missing)
    if show:
        surfaces = np.zeros(rest.shape) * np.nan
        surfaces[land] = 1
        surfaces[water] = 2
        surfaces[rest] = 0

        cmap = plt.get_cmap("viridis", 3)  # 3 discrete colors
        bounds = [-0.5, 0.5, 1.5, 2.5]  # boundaries between values
        norm = BoundaryNorm(bounds, ncolors=cmap.N)

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(15, 5)
        fig.suptitle(title)
        m = ax[0].pcolormesh(alb, cmap="Greys_r")
        plt.colorbar(m, ax=ax[0], label="albedo")
        m = ax[1].pcolormesh(surfaces, cmap=cmap, norm=norm)
        cbar = plt.colorbar(m, ax=ax[1], ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(
            ["rest", rf"{minland}<=DOF<{maxland} & $\Delta$P>-{mindp}", f"DOF>={maxland}"]
        )
        for cax in ax:
            cax.set_xlabel("across-track index")
            cax.set_ylabel("along-track index")
        if show:
            _show_figure(fig)

    return {"land": land, "water": water, "rest": rest}


def compare_l2(
    m1: msat_collection,
    m2: msat_collection,
    lab1: str,
    lab2: str,
    title: str,
    var: str,
    flags=None,
    vrange=None,
    sv_var=None,
    dif_range=None,
    exp_fmt=True,
    extra_id=None,
    lab=None,
    scale=None,
    cmap="viridis",
):
    first = m1.pmesh_prep(var, use_valid_xtrack=True, sv_var=sv_var).compute()
    second = m2.pmesh_prep(var, use_valid_xtrack=True, sv_var=sv_var).compute()
    if scale is not None:
        first = scale * first
        second = scale * second
    if extra_id is not None:
        first = first[extra_id]
        second = second[extra_id]
    if flags is None:
        flags = {"all": np.ones(second.shape)}

    if lab:
        var = lab

    if vrange is None:
        vrange = set_clim(second.data, n_std=5)

    heatmap_hist(
        second,
        flags,
        heatmap_range=vrange,
        hist_range=vrange,
        show=True,
        title=f"{title} {lab2} {var}",
        exp_fmt=exp_fmt,
        cmap=cmap,
    )
    heatmap_hist(
        first,
        flags,
        heatmap_range=vrange,
        hist_range=vrange,
        show=True,
        title=f"{title} {lab1} {var}",
        exp_fmt=exp_fmt,
        cmap=cmap,
    )
    if dif_range is not None:
        dif = first - second
        if "auto" in dif_range:
            q25, q75 = np.nanpercentile(dif, [25, 75])
            nstd = 5 * 0.74 * (q75 - q25)
            med = np.nanmedian(dif)
            dif_range = (-med - nstd, med + nstd)
        heatmap_hist(
            dif,
            flags,
            heatmap_range=dif_range,
            hist_range=dif_range,
            show=True,
            title=f"{title} {lab1} minus {lab2} {var}",
            exp_fmt=exp_fmt,
            cmap=cmap,
        )


def box(
    it=None,
    var=None,
    labs=None,
    data=None,
    xlabel="Iterations",
    ylabel="",
    title="",
    add_line=1,
    ax=None,
    add_legend=True,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    if it is not None and var is not None:
        if labs is None:
            labs = [f"{i:.0f}" for i in np.unique(it) if not np.isnan(i)]
            lab_vals = np.array([float(i) for i in labs])
        else:
            lab_vals = np.array([float(i) for i in labs])
        data = [var[it == v] for v in lab_vals]
    elif data is not None and labs is None:
        labs = range(len(data))
    ax.boxplot(data, tick_labels=labs, **kwargs)
    if add_line is not None:
        ax.axhline(y=add_line, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if add_legend:
        ax.legend()


def box_diag(
    m1: msat_collection,
    m2: msat_collection,
    lab1: str,
    lab2: str,
    title: str,
    var: str,
    ylabel: str,
    flags=None,
    flag_keys: list[str] = ["all"],
    colors1: dict[str, str] = {"all": "lightblue"},
    colors2: dict[str, str] = {"all": "blue"},
    add_line=1,
    sv_var=None,
    extra_id=None,
    scale=None,
    show=True,
):
    second = m2.pmesh_prep(var, use_valid_xtrack=True, sv_var=sv_var).compute()
    ite_second = m2.pmesh_prep("iterations", use_valid_xtrack=True).compute()
    first = m1.pmesh_prep(var, use_valid_xtrack=True, sv_var=sv_var).compute()
    ite_first = m1.pmesh_prep("iterations", use_valid_xtrack=True).compute()
    if scale is not None:
        first = scale * first
        second = scale * second
    if extra_id is not None:
        first = first[extra_id]
        second = second[extra_id]
    if flags is None:
        flags = {"all": np.ones(second.shape)}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    labs = np.arange(2, 16)
    for key in flag_keys:
        box(
            ite_first[flags[key]],
            first[flags[key]],
            labs=labs,
            ylabel=ylabel,
            add_line=add_line,
            title=f"{title} {lab2}",
            ax=ax,
            boxprops={"color": colors1[key]},
            medianprops={"color": colors1[key]},
            whiskerprops={"color": colors1[key]},
            capprops={"color": colors1[key]},
            label="test water",
            showfliers=False,
        )
        box(
            ite_second[flags[key]],
            second[flags[key]],
            labs=labs,
            ylabel=ylabel,
            add_line=add_line,
            title=f"{title} {lab2}",
            ax=ax,
            boxprops={"color": colors2[key]},
            medianprops={"color": colors2[key]},
            whiskerprops={"color": colors2[key]},
            capprops={"color": colors2[key]},
            label="{lab2} water",
            showfliers=False,
        )
    if show:
        _show_figure(fig)

    return fig, ax
