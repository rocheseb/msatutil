from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Sequence, Tuple, Annotated

from msatutil.msat_interface import msat_collection
from msatutil.make_hist import make_hist


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
        lab = "$\Delta P$"
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
