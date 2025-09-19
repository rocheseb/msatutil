from __future__ import annotations

import argparse
import os
from typing import Annotated, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import netCDF4 as ncdf
import numpy as np

from msatutil.msat_interface import msat_collection


def msat_heatmap(
    msat: msat_collection,
    var: str,
    sv_var: Optional[str] = None,
    vminmax: Optional[Annotated[Sequence[float], 2]] = None,
    ratio: bool = False,
    ylim: Optional[Annotated[Sequence[float], 2]] = None,
    grey_path: Optional[str] = None,
    save_path: Optional[str] = None,
    extra_id: Optional[int] = None,
    extra_id_dim: Optional[str] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes], np.ndarray]:
    """
    Make a figure with 2 panels, the top is a greyscale and the bottom is a retrieved variable
    msat: msat_collection object
    var: key contained in the variable to search (uses msat_nc fetch method)
    vminmax: min and max value to be shown with the colorbar
    ratio: if True, plots the variable divided by its median
    ylim: sets the vertical axis range (in cross track pixel indices)
    """
    msat.init_plot(2)
    fig, ax = msat.fig, msat.ax
    fig.set_size_inches(8, 10)
    make_grey(grey_path, ax=ax[0], ylim=ylim)
    if ylim:
        for curax in ax:
            curax.set_ylim(*ylim)
            curax.set_xlabel("along-track index")
            curax.set_ylabel("cross-track index")

    x = msat.pmesh_prep(var, sv_var=sv_var, extra_id=extra_id, extra_id_dim=extra_id_dim)
    if ratio:
        x = x / np.nanmedian(x)
    if vminmax is None:
        m = ax[1].pcolormesh(x, cmap="viridis")
    else:
        m = ax[1].pcolormesh(x, cmap="viridis", vmin=vminmax[0], vmax=vminmax[1])
    if var == "dp":
        lab = "$\Delta P$ (hPa)"
    else:
        lab = var
    plt.colorbar(m, label=lab, ax=ax[1])
    if save_path:
        fig.savefig(save_path)
    return fig, ax, x


def make_grey(
    path,
    ax: Optional[plt.Axes] = None,
    var: str = "alb",
    ylim: Optional[Annotated[Sequence[float], 2]] = None,
):
    """
    path: path to the netcdf file output by get_l1_rad_splat.py
    ax: matplotlib axis object
    var: the variable to plot
    ylim: sets the vertical axis range (in cross track pixel indices)
    """
    noax = ax is None
    with ncdf.Dataset(path) as f:
        if noax:
            fig, ax = plt.subplots()
            fig.set_size_inches(8, 5)
        m = ax.pcolormesh(f[var][:], cmap="Greys_r")
        if var == "alb":
            lab = "Albedo"
        else:
            lab = var
        plt.colorbar(m, label=lab, ax=ax)
        ax.set_xlim(0, f.dimensions["t"].size)
        ax.set_ylim(0, f.dimensions["x"].size)
        ax.set_ylim(*ylim)
    if noax:
        return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Make a greyscale plot and a retrieved variable plot for a given scene"
    )
    parser.add_argument("path", help="full path to the directory containing msat files")
    parser.add_argument(
        "grey_path",
        help="full path to the netcdf file containing the output of get_l1_rad_splat.py",
    )
    parser.add_argument("var", help="variable name")
    parser.add_argument(
        "--sv-var", default=None, help="exact SubStateName of the state vector variable"
    )
    parser.add_argument(
        "--extra-id",
        type=int,
        default=0,
        help="integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables",
    )
    parser.add_argument(
        "--extra-id-dim",
        default="spectral_channel",
        help="dimension name of dimension where extra_id will be applied",
    )
    parser.add_argument("--search", default="proxy.nc", help="string pattern to select msat files")
    parser.add_argument(
        "-r",
        "--ratio",
        action="store_true",
        help="if given, plots the variable divided by its median",
    )
    parser.add_argument(
        "--vminmax",
        nargs=2,
        type=float,
        default=None,
        help="min and max values for the colorbar",
    )
    parser.add_argument(
        "--ylim",
        nargs=2,
        type=float,
        default=[25, 200],
        help="sets vertical axis limits",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        default="",
        help="full filepath to save the plot (includes filename)",
    )
    parser.add_argument(
        "--use-dask", action="store_true", help="if given, use dask to handle the data"
    )
    args = parser.parse_args()

    if args.sv_var and args.var not in ["APosterioriState", "APrioriState"]:
        raise Exception(
            'When --sv-var is given, var must be one of ["APrioriState","APosterioriState"]'
        )

    if not os.path.isdir(args.path):
        raise Exception(f"{args.path} is not a valid path")
    if not os.path.isfile(args.grey_path):
        raise Exception(f"{args.grey_path} is not a valid path")
    if args.save_path and not os.path.exists(os.path.dirname(args.save_path)):
        raise Exception(f"{args.save_path} is not a valid path")

    msat = msat_collection(
        [
            os.path.join(args.path, i)
            for i in os.listdir(args.path)
            if args.search in i and i.endswith(".nc")
        ],
        use_dask=args.use_dask,
    )

    return msat_heatmap(
        msat,
        args.var,
        sv_var=args.sv_var,
        vminmax=args.vminmax,
        ratio=args.ratio,
        ylim=args.ylim,
        grey_path=args.grey_path,
        save_path=args.save_path,
        extra_id=args.extra_id,
        extra_id_dim=args.extra_id_dim,
    )


if __name__ == "__main__":
    main()
