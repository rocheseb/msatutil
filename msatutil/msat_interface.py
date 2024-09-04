from __future__ import annotations
import os
import sys
import glob
import re
import numpy as np
import netCDF4 as ncdf
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import argparse
from msatutil.msat_nc import msat_nc, MSATError
from collections import OrderedDict
from typing import Optional, Sequence, Tuple, Union, Annotated, List, Dict
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import time
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from scipy.interpolate.interpnd import _ndim_coords_from_arrays
import pickle
from tqdm import tqdm
from msatutil.make_hist import make_hist
from msatutil.msat_dset import gs_list


@dask.delayed
def get_msat_file(file_path: str):
    """
    Function to open msat_collection object faster when there are many files

    Inputs:
        file_path (str): full path to the input netcdf file
    """

    return msat_file(file_path, use_dask=True)


def meters_to_lat_lon(x: float, lat: float) -> float:
    """
    Convert a distance in meters to latitudinal and longitudinal angles at a given latitude
    https://en.wikipedia.org/wiki/Geographic_coordinate_system
    Uses WGS84 https://en.wikipedia.org/wiki/World_Geodetic_System

    Inputs:
        x: distance (meters)
        lat: latitude (degrees)
    Outputs:
        (lat_deg,lon_deg): latitudinal and longitudinal angles corresponding to x (degrees)

    """
    lat = np.deg2rad(lat)
    meters_per_lat_degree = (
        111132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) - 0.0023 * np.cos(6 * lat)
    )

    meters_per_lon_degree = (
        111412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat) + 0.118 * np.cos(5 * lat)
    )

    return x / meters_per_lat_degree, x / meters_per_lon_degree


def filter_large_triangles(points: np.ndarray, tri: Optional[Delaunay] = None, coeff: float = 2.0):
    """
    Filter out triangles that have an edge > coeff * median(edge)
    Inputs:
        tri: scipy.spatial.Delaunay object
        coeff: triangles with an edge > coeff * median(edge) will be filtered out
    Outputs:
        valid_slice: boolean array that select for the triangles that
    """
    if tri is None:
        tri = Delaunay(points)

    edge_lengths = np.zeros(tri.simplices.shape)
    seen = {}
    # loop over triangles
    for i, vertex in enumerate(tri.simplices):
        # loop over edges
        for j in range(3):
            id0 = vertex[j]
            id1 = vertex[(j + 1) % 3]

            # avoid calculating twice for non-border edges
            if (id0, id1) in seen:
                edge_lengths[i, j] = seen[(id0, id1)]
            else:
                edge_lengths[i, j] = np.linalg.norm(points[id1] - points[id0])

                seen[(id0, id1)] = edge_lengths[i, j]

    median_edge = np.median(edge_lengths.flatten())

    valid_slice = np.all(edge_lengths < coeff * median_edge, axis=1)

    return valid_slice


def chunked(lst: List, n: int):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f"{func.__name__} done in {time.time()-start} s")

    return wrapper


def get_msat(
    indir: str,
    date_range: Optional[Annotated[Sequence[datetime.datetime], 2]] = None,
    date_pattern: str = r"%Y%m%dT%H%M%S",
    srchstr="Methane*.nc",
):
    """
    Function to get the L1B or L2 files under indir into a msat_collection object

    Inputs:
        indir (str): full path for the input directory, can be a gs:// path
        date_range (Optional[Annotated[Sequence[datetime.datetime], 2]]): only the files within the date range will be kept
        date_pattern (str): regex pattern for the date in the filenames
        srchstr (str): search pattern for the files (accepts wildcards *)
    """
    if indir.startswith("gs://"):
        flist = gs_list(indir, srchstr=srchstr)
    else:
        flist = glob.glob(os.path.join(indir, srchstr))
    return msat_collection(flist, date_range=date_range, date_pattern=date_pattern, use_dask=True)


class msat_file(msat_nc):
    """
    Class to interface with a single MethaneSAT/AIR L1B or L2 file
    It has methods to make simple plots.
    """

    def __init__(self, msat_file: str, use_dask: bool = False) -> None:
        super().__init__(msat_file, use_dask=use_dask)

    def spec_plot(self, ax: plt.Axes, j: int, i: int, label: str) -> None:
        """
        Make a spectrum+residuals plot for a given pixel
        ax: a list of 2 matplotlib axes
        j: along-track pixel index
        i: cross-track pixel index
        label: label for the legend
        """
        ax[0].axhline(y=0, linestyle="--", color="black")
        line = self.plot_var(ax[0], "Posteriori_RTM_Band1", "ResidualRadiance", j, i, label)
        self.plot_var(
            ax[1],
            "Posteriori_RTM_Band1",
            "ObservedRadiance",
            j,
            i,
            "Obs",
            color=line.get_color(),
        )
        self.plot_var(
            ax[1],
            "Posteriori_RTM_Band1",
            "Radiance_I",
            j,
            i,
            "Calc",
            color=line.get_color(),
            linestyle="--",
        )
        ax[0].get_shared_x_axes().join(ax[0], ax[1])

    def plot_var(
        self,
        ax: plt.Axes,
        grp: str,
        var: str,
        j: int,
        i: int,
        label: str,
        color: Optional[str] = None,
        linestyle: Optional[str] = None,
    ) -> matplotlib.lines.Line2D:
        """
        Plot a given variable for a given pixel
        ax: matplotlib axes object
        grp: complete group name
        var: complete variable name
        j: along-track pixel index
        i: cross-track pixel index
        label: label for the legend
        color: line color
        linestyle: matplotlib linestyle
        """
        nc_grp = self.nc_dset[grp]
        if nc_grp[var].shape[0] == 1024:
            if nc_grp[var].dimensions[0].startswith("wmx"):
                obs_rad = nc_grp["ObservedRadiance"][:, j, i]
            elif nc_grp[var].dimensions[0].startswith("w1"):  # Native L1 files
                obs_rad = nc_grp["Radiance"][:, j, i]
            obs_rad = np.ma.masked_where(obs_rad == 0, obs_rad)
            if not obs_rad.mask:
                obs_rad.mask = np.zeros(obs_rad.size, dtype=bool)
            if var == "ResidualRadiance":
                rms = 100 * self.get_pixel_rms(j, i)
                sp_slice = self.get_sv_slice("SurfacePressure")
                dp = self.get_pixel_dp(j, i)
                label = f"{label}; rms={rms:.4f}; dP={dp:.3f}"
                ax.set_ylabel("Residuals (%)")
                line = ax.plot(
                    nc_grp["Wavelength"][~obs_rad.mask, j, i],
                    100 * nc_grp[var][~obs_rad.mask, j, i] / obs_rad[~obs_rad.mask],
                    label=label,
                )
            else:
                print(obs_rad.mask)
                print(nc_grp[var][~obs_rad.mask, j, i])
                print(nc_grp["Wavelength"][~obs_rad.mask, j, i])
                line = ax.plot(
                    nc_grp["Wavelength"][~obs_rad.mask, j, i],
                    nc_grp[var][~obs_rad.mask, j, i],
                    label=f'{label.split("_")[3]} {j} {i}',
                )
            ax.set_xlabel("Wavelength (nm)")
        elif nc_grp[var].dimensions[0] == "one":
            line = ax.axhline(y=nc_grp[var][:, j, i], label=label)
            ax.set_ylabel()
        elif nc_grp[var].dimensions[0].startswith("lmx"):
            line = ax.plot(
                self.nc_dset[grp][var][:, j, i],
                self.nc_dset[grp]["PressureMid"][:, j, i],
                label=label,
            )
            ax.set_ylabel("Pressure (hPa)")

        if color:
            line[0].set_color(color)
        if linestyle:
            line[0].set_linestyle(linestyle)

        return line[0]


class msat_collection:
    """
    Class to interface with a list of MethaneSAT/AIR L1B or L2 files.
    Its methods help create quick plots for multiple granules.
    The main methods are pmesh_prep, grid_prep, and heatmap
    pmesh_prep: helps read in a given variable from all the files in the collection by concatenating in the along-track dimension
    grid_prep: is similar to pmesh_prep but puts the data onto a regular lat-lon grid
    heatmap: show a pcolormesh plot of the given variable
    """

    def __init__(
        self,
        file_list: List[str],
        date_range: Optional[Annotated[Sequence[datetime.datetime], 2]] = None,
        date_pattern: str = r"%Y%m%dT%H%M%S",
        use_dask: bool = False,
    ) -> None:
        """
        file_list (List[str]): list of file paths
        date_range (Optional[Annotated[Sequence[datetime.datetime], 2]]): only the files within the date range will be kept
        date_pattern (str): regex pattern for the date in the filenames
        use_dask (bool): if True, use dask to open files and read data
        """
        self.set_use_dask(use_dask)
        self.file_list = np.array(file_list)

        self.parse_dates(date_range, date_pattern)

        self.file_paths = self.file_list
        self.file_names = np.array([os.path.basename(file_path) for file_path in self.file_list])
        self.ids = OrderedDict([(i, file_path) for i, file_path in enumerate(self.file_list)])
        self.ids_rev = {val: key for key, val in self.ids.items()}
        if use_dask:
            results = [get_msat_file(file_path) for file_path in self.file_list]
            if len(results) > 50:
                with ProgressBar():
                    msat_file_list = dask.compute(*results)
            else:
                msat_file_list = dask.compute(*results)
            self.msat_files = OrderedDict(
                [(file_path, msat_file_list[i]) for i, file_path in enumerate(self.file_list)]
            )
        else:
            self.msat_files = OrderedDict(
                [
                    (file_path, msat_file(file_path, use_dask=use_dask))
                    for file_path in self.file_list
                ]
            )
        self.dsets = {key: val.nc_dset for key, val in self.msat_files.items()}

        self.is_l1 = self.msat_files[self.ids[0]].is_l1
        self.is_l2 = self.msat_files[self.ids[0]].is_l2
        self.is_l2_met = self.msat_files[self.ids[0]].is_l2_met
        self.is_postproc = self.msat_files[self.ids[0]].is_postproc
        self.is_l3 = self.msat_files[self.ids[0]].is_l3
        self.valid_xtrack = self.get_valid_xtrack()
        self.dim_size_map = self.msat_files[self.ids[0]].dim_size_map
        # when using self.get_dim_map(var_path), the result maps dimensions to dimension axis using the common set of dimensions names common_dim_set
        self.common_dim_set = [
            "one",
            "xtrack",
            "atrack",
            "xtrack_edge",
            "atrack_edge",
            "lev",
            "ch4_lev",
            "o2_lev",
            "lev_edge",
            "corner",
            "spectral_channel",
            "xmx",
            "nsubx",
            "iter_x",
            "iter_w",
            "err_col",
            "err_proxy",
            "lat",  # L3 dims
            "lon",  # L3 dims
        ]

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        for f in self.msat_files.values():
            f.close()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"""msat_collection:
        number of files: {len(self.file_paths)}
        valid_xtrack: {self.valid_xtrack}
        use_dask: {self.use_dask}
        is_l1: {self.is_l1}
        is_l2: {self.is_l2}
        is_l3: {self.is_l3}
        is_postproc: {self.is_postproc}
        is_l2_met: {self.is_l2_met}
        """

    @staticmethod
    def convert_time_format(time_format: str):
        """
        Return the regex corresponding to time_format
        """
        format_mapping = {
            "%H": r"\d{2}",
            "%M": r"\d{2}",
            "%S": r"\d{2}",
            "%m": r"\d{2}",
            "%d": r"\d{2}",
            "%y": r"\d{2}",
            "%Y": r"\d{4}",
        }

        re_format = time_format
        for code, pattern in format_mapping.items():
            if re.search(code, re_format):
                re_format = re_format.replace(code, pattern)

        return re_format

    def parse_dates(
        self,
        date_range: Optional[Annotated[Sequence[datetime.datetime], 2]] = None,
        date_pattern: str = r"%Y%m%dT%H%M%S",
    ):
        """
        sort the files
        """
        regex_pattern = self.convert_time_format(date_pattern)
        date_pattern_matches = re.findall(regex_pattern, os.path.basename(self.file_list[0]))
        n_matches = len(date_pattern_matches)

        if date_pattern_matches:
            start_dates = np.array(
                [
                    datetime.strptime(
                        re.findall(regex_pattern, os.path.basename(file_path))[0], date_pattern
                    )
                    for file_path in self.file_list
                ]
            )
            sort_ids = np.argsort(start_dates)
            self.start_dates = start_dates[sort_ids]
            self.file_list = self.file_list[sort_ids]

            if n_matches >= 2:
                self.end_dates = np.array(
                    [
                        datetime.strptime(
                            re.findall(regex_pattern, os.path.basename(file_path))[1], date_pattern
                        )
                        for file_path in self.file_list
                    ]
                )
            if date_range:
                date_ids = (self.start_dates >= date_range[0]) & (self.start_dates < date_range[1])
                self.file_list = self.file_list[date_ids]
                self.start_dates = self.start_dates[date_ids]
                if n_matches >= 2:
                    self.end_dates = self.end_dates[date_ids]
        else:
            print(f"/!\\ No matches for {regex_pattern} in filenames")
            self.start_dates = None
            self.end_dates = None

    def get_valid_xtrack(self, varpath: Optional[str] = None):
        """
        Get the valid cross track indices
        """
        return self.msat_files[self.ids[0]].get_valid_xtrack(varpath)

    def get_valid_rad(self):
        """
        Get the valid radiance indices
        """
        return self.msat_files[self.ids[0]].get_valid_rad()

    def get_dim_map(self, var_path: str) -> Dict[str, str]:
        """
        Get the dimension map of the given variable
        """
        return self.msat_files[self.ids[0]].get_dim_map(var_path)

    def subset(
        self,
        ids: Optional[list] = None,
        date_range: Optional[Annotated[Sequence[datetime.datetime], 2]] = None,
        use_dask: bool = True,
    ) -> msat_collection:
        """
        Return a subset msat_collection object corresponding to the ids given (must be present in the keys of self.ids)
        """

        if date_range is not None:
            ids = np.where(
                (self.start_dates >= date_range[0]) & (self.start_dates < date_range[1])
            )[0]

        return msat_collection([self.file_paths[i] for i in ids], use_dask=use_dask)

    def init_plot(self, nplots: int, ratio=[]) -> None:
        """
        Generate an empty figure to be filled by the other methods of this class
        """
        if not ratio:
            ratio = [1 for i in range(nplots)]
        self.fig, self.ax = plt.subplots(nplots, gridspec_kw={"height_ratios": ratio})
        self.lines = {}

        self.fig.set_size_inches(10, 8)

    def plot_var(
        self,
        grp: str,
        var: str,
        j: int,
        i: int,
        ids: Optional[List[int]] = None,
        ax=None,
    ) -> None:
        """
        Plot a given variable at a given pixel
        grp: complete group name
        var: complete variable name
        j: along-track pixel index
        i: cross-track pixel index
        ids: list of ids of the msat files (from the keys of self.ids)
        ax: matplotlib axes object, if not specified it will use self.ax
        """
        if ax is None:
            ax = self.ax
        if ids is None:
            ids = self.ids.keys()
        ax.grid(True)
        if hasattr(self.msat_files[self.ids[0]].nc_dset[grp][var], "units"):
            ax.set_ylabel(f"{grp} {var} {self.msat_files[self.ids[0]].nc_dset[grp][var].units}")
        else:
            ax.set_ylabel(f"{grp} {var}")

        file_list = [self.ids[i] for i in ids]
        for msat_file in file_list:
            self.msat_files[msat_file].plot_var(ax, grp, var, j, i, label=msat_file)
            self.lines[msat_file] = ax.lines[-1]

        box = ax.get_position()
        if box.width == 0.775:
            ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(loc="center left", bbox_to_anchor=[1.04, 0.5], borderaxespad=0)

    def rm_line(self, msat_file, ax=None) -> None:
        """
        Remove a line corresponding to the given msat_file
        msat_file: an msat_file object (from the keys of self.msat_files)
        ax: matplotlib axes object, if not specified it will use self.ax
        """
        if ax is None:
            ax = self.ax
        ax.lines.remove(self.lines[msat_file])

    def pixel_diag(self, j: int, i: int, ids: Optional[List[int]] = None) -> None:
        """
        For a given pixel and file, plot the fit residuals, DP, and the rms of residuals
        j: along-track pixel index
        i: cross-track pixel index
        ids: list of ids of the msat files (from the keys of self.ids)
        """
        if ids is None:
            ids = self.ids.keys()
        self.init_plot(3)
        self.plot_var("Posteriori_RTM_Band1", "ResidualRadiance", j, i, ids=ids, ax=self.ax[0])
        self.ax[0].set_title("% Residuals")

        # dP and rms plots
        file_list = [self.ids[i] for i in ids]
        for msat_file in file_list:
            dp = self.msat_files[msat_file].get_pixel_dp(j, i)
            line = self.ax[1].axhline(y=dp, label=f"{msat_file}; dP={dp:.3f}")
            line.set_color(self.lines[msat_file].get_color())

            rms = 100 * self.msat_files[msat_file].get_pixel_rms(j, i)
            line = self.ax[2].axhline(y=rms, label=f"{msat_file}; rms={rms:.4f}")
            line.set_color(self.lines[msat_file].get_color())

        self.ax[1].set_ylabel(r"$\Delta$P (hPa)")
        self.ax[1].set_title("Surface pressure change")
        self.ax[1].grid()

        self.ax[2].set_ylabel("Residual RMS (%)")
        self.ax[2].set_title("RMS of residuals")
        self.ax[2].grid()

        plt.tight_layout()

    def spec_plot(self, j: int, i: int, ids: Optional[List[int]] = None) -> None:
        """
        Make a spectrum+residuals plot for a given pixel in given files
        j: along-track pixel index
        i: cross-track pixel index
        ids: list of ids of the msat files (from the keys of self.ids)
        """
        if ids is None:
            ids = self.ids.keys()
        self.init_plot(2, ratio=[1, 3])
        file_list = [self.ids[i] for i in ids]
        for msat_file in file_list:
            self.msat_files[msat_file].spec_plot(self.ax, j, i, msat_file)
        for ax in self.ax:
            ax.grid()
            ax.legend()

    def hist(
        self,
        ax: plt.Axes = None,
        label: str = None,
        color: str = None,
        rng: Optional[Annotated[Sequence[float], 2]] = None,
        nbins: int = 100,
        scale: float = 1.0,
        exp_fmt: bool = True,
        var: str = None,
        grp: Optional[str] = None,
        sv_var: Optional[str] = None,
        extra_id: Optional[int] = None,
        extra_id_dim: Optional[str] = None,
        ids: Optional[List[int]] = None,
        ratio: bool = False,
        option: Optional[str] = None,
        option_axis_dim: str = "spectral_channel",
        chunks: Union[str, Tuple] = "auto",
        set_nan: Optional[float] = None,
    ) -> None:
        """
        Plot a histogram of the given variable
        ax: matplotlib axes object
        ## make_hist arguments:
        label: horizontal axis label
        color: the color of the bars
        rng: range of the horizontal axis
        nbins: number of bins for the histogram
        scale: quantity to multiply the variable with (can be useful to avoid overflow in the standard deviation of column amounts)
        exp_fmt: if True, use .3e format for stats in the histogram legend. If false use .2f format
        ## msat_collection.pmesh_prep arguments:
        var: key contained in the variable to search (uses msat_nc fetch method)
        grp: if givem use msat_nc.get_var instead of msat_nc.fetch and var must be the exact variable name
        sv_var: grp will be set to SpecFitDiagnostics and sv_var must be one of APrioriState or APosterioriState, and var must be the exact SubStateName of the state vector variable
        extra_id: integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables
        extra_id_dim: name of the dimension along which extra_id will be selected
        ids: list of ids corresponding to the keys of self.ids, used to select which files are concatenated
        ratio: if True, return the variable divided by its median
        option: can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std')
        option_axis_dim: the axis along which the stat is applied
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        set_nan (Optional[float]): this value will be replaced with nan after a pmesh_prep call
        """
        if ax is None:
            self.init_plot(1)
            fig, ax = self.fig, self.ax
            fig.set_size_inches(8, 5)

        x = self.pmesh_prep(
            var,
            grp,
            sv_var,
            extra_id,
            extra_id_dim,
            ids,
            ratio,
            option,
            option_axis_dim,
            chunks,
            set_nan,
        ).compute()

        x = x[np.isfinite(x)].flatten() * scale

        make_hist(ax, x, label, color, rng, nbins, exp_fmt)

    def pmesh_prep(
        self,
        var: str,
        grp: Optional[str] = None,
        sv_var: Optional[str] = None,
        extra_id: Optional[int] = None,
        extra_id_dim: Optional[str] = None,
        ids: Optional[List[int]] = None,
        ratio: bool = False,
        option: Optional[str] = None,
        option_axis_dim: str = "spectral_channel",
        chunks: Union[str, Tuple] = "auto",
        set_nan: Optional[float] = None,
        use_valid_xtrack: bool = False,
    ) -> Union[np.ndarray, da.core.Array]:
        """
        get a variable ready to plot with plt.pcolormesh(var)
        var: key contained in the variable to search (uses msat_nc fetch method)
        grp: if givem use msat_nc.get_var instead of msat_nc.fetch and var must be the exact variable name
        sv_var: grp will be set to SpecFitDiagnostics and sv_var must be one of APrioriState or APosterioriState, and var must be the exact SubStateName of the state vector variable
        extra_id: integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables
        extra_id_dim: name of the dimension along which extra_id will be selected
        ids: list of ids corresponding to the keys of self.ids, used to select which files are concatenated
        ratio: if True, return the variable divided by its median
        option: can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std')
        option_axis_dim: the axis along which the stat is applied
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        set_nan (Optional[float]): this value will be replaced with nan after a pmesh_prep call
        use_valid_xtrack (bool): if True, only gets the data along the self.valid_xtrack slice
        """
        if ids is None:
            ids = self.ids
        else:
            ids = {i: self.ids[i] for i in ids}
        if (sv_var is not None) and (
            var not in ["APosterioriState", "APrioriState", "IntermediateState"]
        ):
            raise MSATError(
                'var must be one of ["APrioriState","APosterioriState"] when sv_var is given'
            )
        elif sv_var is not None:
            sv_slice = self.get_sv_slice(sv_var)
            grp = "SpecFitDiagnostics"
        else:
            nc_slice = [slice(None)]

        if var == "dp" and self.msat_files[self.ids[0]].dp is None:
            self.read_dp()

        if "/" in var:
            var_path = var
            grp, var = var.split("/")
        else:
            var_path = self.fetch_varpath(var, grp=grp)
        var_dim_map = self.get_dim_map(var_path)
        if sv_var is not None:
            nc_slice = [slice(None) for dim in var_dim_map]
            nc_slice[var_dim_map["xmx"]] = sv_slice
        if self.is_l3:
            atrack_axis = 0
        else:
            atrack_axis = var_dim_map["atrack"]
        x = []
        tqdm_disable = len(list(ids.values())) < 50
        for num, i in tqdm(
            enumerate(ids.values()),
            total=len(ids),
            disable=tqdm_disable,
            leave=False,
            desc=var_path,
        ):
            if var == "dp":
                x.append(self.msat_files[i].dp)
            else:
                if grp is None:
                    x.append(self.msat_files[i].fetch(var, chunks=chunks))
                else:
                    x.append(self.msat_files[i].get_var(var, grp, chunks=chunks)[tuple(nc_slice)])

        if self.use_dask:
            x = da.concatenate(x, axis=atrack_axis)
            x[da.greater(x, 1e29)] = np.nan
            if not self.is_l3:
                x = x.rechunk({atrack_axis: "auto"})
        else:
            x = np.concatenate(x, axis=atrack_axis)
            x[np.greater(x, 1e29)] = np.nan

        x_slices = [slice(None) for i in range(len(x.shape))]
        original_ndim = len(x_slices)
        reduced = False
        if option is not None:
            option_axis = var_dim_map[option_axis_dim]
            if self.use_dask:
                x = getattr(da, option)(x, axis=option_axis)
            else:
                x = getattr(np, option)(x, axis=option_axis)
            x_slices = [slice(None) for i in range(len(x.shape))]
            reduced = len(x_slices) < original_ndim
        elif (extra_id is not None) and (extra_id_dim is not None):
            extra_id_dim_axis = var_dim_map[extra_id_dim]
            x_slices[extra_id_dim_axis] = extra_id
        if use_valid_xtrack and "xtrack" in var_dim_map:
            xtrack_dim_axis = var_dim_map["xtrack"]
            if reduced and option_axis < xtrack_dim_axis:
                xtrack_dim_axis -= 1
            x_slices[xtrack_dim_axis] = self.valid_xtrack
        x = x[tuple(x_slices)]

        x = x.squeeze()

        if set_nan is not None:
            x[x == set_nan] = np.nan

        if ratio:
            if self.use_dask:
                x = x / da.nanmedian(x)
            else:
                x = x / np.nanmedian(x)
        return x

    def grid_prep(
        self,
        var: str,
        lon_lim: Annotated[Sequence[float], 2],
        lat_lim: Annotated[Sequence[float], 2],
        n: Optional[int] = None,
        grp: Optional[str] = None,
        sv_var: Optional[str] = None,
        extra_id: Optional[int] = None,
        extra_id_dim: Optional[str] = None,
        ids: Optional[List[int]] = None,
        ratio: bool = False,
        option: Optional[str] = None,
        option_axis_dim: str = "spectral_channel",
        chunks: Union[str, Tuple] = "auto",
        method: str = "cubic",
        res: float = 20,
        set_nan: Optional[float] = None,
        use_valid_xtrack: bool = False,
    ) -> da.core.Array:
        """
        get a variable ready to plot with plt.pcolormesh(lon_grid,lat_grid,x_grid_avg)
        var: key contained in the variable to search (uses msat_nc fetch method)
        lon_lim: the [min,max] of the longitudes to regrid the data on
        lat_lim: the [min,max] of the latitudes to regrid the data on
        n: the size of chunks to separate the files into (if less than the number of files, there may be white lines in the plot)
        grp: if givem use msat_nc.get_var instead of msat_nc.fetch and var must be the exact variable name
        sv_var: grp will be set to SpecFitDiagnostics and sv_var must be one of APrioriState or APosterioriState, and var must be the exact SubStateName of the state vector variable
        extra_id: integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables
        extra_id_dim: name of the dimension along which extra_id will be selected
        ids: list of ids corresponding to the keys of self.ids, used to select which files are concatenated
        ratio: if True, return the variable divided by its median
        option: can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std')
        option_axis_dim: the axis along which the stat is applied
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        method: griddata interpolation method
        res: grid resolution in meters
        set_nan (Optional[float]): this value will be replaced with nan after a pmesh_prep call
        use_valid_xtrack (bool): if True, only gets the data along the self.valid_xtrack slice
        """
        if not self.use_dask:
            raise MSATError("grid_prep needs self.use_dask==True")

        if ids is None:
            ids = self.ids
        else:
            ids = OrderedDict([(i, self.ids[i]) for i in ids])
        if n is None:
            n = len(ids)

        chunked_ids = list(chunked(list(ids.keys()), n))

        print(
            f"Calling grid_prep on {len(list(ids.keys()))} files, divided in {len(chunked_ids)} chunks of {n} files\n"
        )

        mid_lat = (lat_lim[1] - lat_lim[0]) / 2.0

        lat_res, lon_res = meters_to_lat_lon(res, mid_lat)

        lon_range = da.arange(lon_lim[0], lon_lim[1], lon_res)
        lat_range = da.arange(lat_lim[0], lat_lim[1], lat_res)

        # compute the lat-lon grid now so it doesn't have to be computed for each griddata call
        lon_grid, lat_grid = dask.compute(*da.meshgrid(lon_range, lat_range))

        x_grid_list = []
        for i, ids_slice in enumerate(chunked_ids):
            sys.stdout.write(f"\rgrid_prep now doing chunk {i+1:>3}/{len(chunked_ids)}")
            sys.stdout.flush()

            x = self.pmesh_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                extra_id_dim=extra_id_dim,
                ids=ids_slice,
                option=option,
                option_axis_dim=option_axis_dim,
                ratio=ratio,
                chunks=chunks,
                set_nan=set_nan,
                use_valid_xtrack=use_valid_xtrack,
            )
            lat = self.pmesh_prep("Latitude", ids=ids_slice, chunks=chunks)
            lon = self.pmesh_prep("Longitude", ids=ids_slice, chunks=chunks)

            nonan = ~da.isnan(x)
            flat_x = x[nonan].compute()
            flat_lat = lat[nonan].compute()
            flat_lon = lon[nonan].compute()

            x_grid = griddata(
                (flat_lon, flat_lat),
                flat_x,
                (lon_grid, lat_grid),
                method=method,
                rescale=True,
            )

            cloud_points = _ndim_coords_from_arrays((flat_lon, flat_lat))
            regrid_points = _ndim_coords_from_arrays((lon_grid.ravel(), lat_grid.ravel()))
            tri = Delaunay(cloud_points)

            outside_hull = np.zeros(lon_grid.size).astype(bool)
            if method == "nearest":
                # filter out the extrapolated points when using nearest neighbors
                outside_hull = tri.find_simplex(regrid_points) < 0

            # filter out points that fall in large triangles
            # create a new scipy.spatial.Delaunay object with only the large triangles
            large_triangles = ~filter_large_triangles(cloud_points, tri)
            large_triangle_ids = np.where(large_triangles)[0]
            subset_tri = tri  # this doesn't preserve tri, effectively just a renaming
            # the find_simplex method only needs the simplices and neighbors
            subset_tri.nsimplex = large_triangle_ids.size
            subset_tri.simplices = tri.simplices[large_triangles]
            subset_tri.neighbors = tri.neighbors[large_triangles]
            # update neighbors
            for i, triangle in enumerate(subset_tri.neighbors):
                for j, neighbor_id in enumerate(triangle):
                    if neighbor_id in large_triangle_ids:
                        # reindex the neighbors to match the size of the subset
                        subset_tri.neighbors[i, j] = np.where(large_triangle_ids == neighbor_id)[0]
                    elif neighbor_id >= 0 and (neighbor_id not in large_triangle_ids):
                        # that neighbor was a "normal" triangle that should not exist in the subset
                        subset_tri.neighbors[i, j] = -1
            inside_large_triangles = subset_tri.find_simplex(regrid_points, bruteforce=True) >= 0
            invalid_slice = np.logical_or(outside_hull, inside_large_triangles)
            x_grid[invalid_slice.reshape(x_grid.shape)] = np.nan

            x_grid_list.append(x_grid)

        stacked_grid = da.stack(x_grid_list, axis=0)
        x_grid_avg = da.nanmean(stacked_grid, axis=0)

        return lon_grid, lat_grid, x_grid_avg

    def heatmap_loop(self, id_chunk: int, ax: Optional[plt.Axes] = None, **kwargs):
        """
        Call heatmap in a loop over self.ids

        id_chunk (int): self.ids will be divided into a list a id_chunk successive chunks
        ax (Optional[plt.Axes]): figure artist
        """
        chunked_ids = list(chunked(list(self.ids.keys()), id_chunk))
        if ax is None:
            self.init_plot(1)
            fig, ax = self.fig, self.ax
            fig.set_size_inches(8, 5)
        kwargs["ax"] = ax
        for ids in tqdm(chunked_ids, total=len(chunked_ids)):
            kwargs["ids"] = ids
            self.heatmap(**kwargs)

    def heatmap(
        self,
        var: str,
        grp: Optional[str] = None,
        sv_var: Optional[str] = None,
        vminmax: Optional[Annotated[Sequence[float], 2]] = None,
        over: str = "red",
        under: str = "hotpink",
        latlon: bool = False,
        ratio: bool = False,
        ylim: Optional[Annotated[Sequence[float], 2]] = None,
        save_path: Optional[str] = None,
        extra_id: Optional[int] = None,
        extra_id_dim: Optional[str] = None,
        ids: Optional[List[int]] = None,
        option: Optional[str] = None,
        option_axis_dim: str = "spectral_channel",
        chunks: Union[str, Tuple] = "auto",
        lon_lim: Optional[Annotated[Sequence[float], 2]] = None,
        lat_lim: Optional[Annotated[Sequence[float], 2]] = None,
        n: Optional[int] = None,
        method: str = "cubic",
        save_nc: Optional[Annotated[Sequence[str], 2]] = None,
        ax: Optional[plt.Axes] = None,
        res: float = 20,
        scale: float = 1.0,
        cmap: str = "viridis",
        set_nan: Optional[float] = None,
        use_valid_xtrack: bool = False,
    ) -> None:
        """
        Make a heatmap of the given variable
        var (str): key contained in the variable to search (uses msat_nc fetch method)
        grp (Optional[str]): if givem use msat_nc.get_var instead of msat_nc.fetch and var must be the exact variable name
        vminmax (Optional[Annotated[Sequence[float], 2]]): min and max value to be shown with the colorbar
        over (str): color to use above the max of the color scale (only used if vminmax is not None)
        under (str): color to use under the min of the color scale (only used if vminmax is not None)
        latlon (bool): if True, make the plot on latitude/longitude instead of xtrack/atrack
        ratio (bool): if True, plots the variable divided by its median
        ylim (Optional[Annotated[Sequence[float], 2]]): sets the vertical axis range
        extra_id (Optional[int]): integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables and when "option" is None
        extra_id_dim (str): name of the dimension along which extra_id will be selected
        ratio (bool): if True, return the variable divided by its median
        option (Optional[str]): can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std'), for example to plot a heatmap of the maximum radiance
        option_axis_dim (str): the axis along which the stat is applied (from the set of common dimension names: ["xtrack","atrack","xtrack_edge","atrack_edge","lev","lev_edge","corner","spectral_channel","xmx","nsubx"])
        chunks (Union[str, Tuple]): when self.use_dask is True, sets the chunk size for dask arrays
        lon_lim (Optional[Annotated[Sequence[float], 2]]): [min,max] longitudes for the gridding
        lat_lim (Optional[Annotated[Sequence[float], 2]]): [min,max] latitudes for the gridding
        n (Optional[int]): number of files chunked together for the gridding
        method (str): griddata interpolation method, only used if lon_lim and lat_lim are given
        save_nc (Optional[Annotated[Sequence[str], 2]]): [nc_file_path,varname] list containing the full path to the output L3 netcdf file and the name the variable will have in the file
        ax (Optional[plt.Axes]): if given, make the plot in the given matplotlib axes object, otherwise create a new one
        res (float): the resolution (in meters) of the grid with lon_lim and lat_lim are given
        scale (float): a factor with which the data will be scaled
        cmap (str): matplotlib named colormaps (https://matplotlib.org/stable/gallery/color/colormap_reference.html)
        set_nan (Optional[float]): this value will be replaced with nan after a pmesh_prep call
        use_valid_xtrack (bool): if True, only gets the data along the self.valid_xtrack slice
        """
        if ids is None:
            ids = self.ids
        if n is None:
            n = len(ids)
        if ax is None:
            self.init_plot(1)
            fig, ax = self.fig, self.ax
            fig.set_size_inches(8, 5)
        else:
            save_path = False  # when an input axis is given, don't try saving a figure

        if ylim:
            ax.set_ylim(*ylim)

        gridded = (lon_lim is not None) and (lat_lim is not None)

        if gridded and not self.use_dask:
            raise MSATError("/!\\ the gridded argument only works when self.use_dask is True")

        if latlon and not use_valid_xtrack:
            use_valid_xtrack = True

        if latlon and gridded:
            print("Note: latlon does nothing when gridded is True")
            latlon = False

        if not gridded:
            x = self.pmesh_prep(
                var,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                extra_id_dim=extra_id_dim,
                ids=ids,
                option=option,
                option_axis_dim=option_axis_dim,
                ratio=ratio,
                chunks=chunks,
                set_nan=set_nan,
                use_valid_xtrack=use_valid_xtrack,
            )
            x = x * scale
            if latlon:
                if self.is_l3:
                    lon_str = "lon"
                    lat_str = "lat"
                else:
                    lon_str = "Longitude"
                    lat_str = "Latitude"
                lat = self.pmesh_prep(
                    lat_str,
                    ids=ids,
                    chunks=chunks,
                    use_valid_xtrack=use_valid_xtrack,
                )
                lon = self.pmesh_prep(
                    lon_str,
                    ids=ids,
                    chunks=chunks,
                    use_valid_xtrack=use_valid_xtrack,
                )
            # end of if not gridded
        elif gridded:
            lon, lat, x = self.grid_prep(
                var,
                lon_lim,
                lat_lim,
                n=n,
                grp=grp,
                sv_var=sv_var,
                extra_id=extra_id,
                extra_id_dim=extra_id_dim,
                ids=ids,
                option=option,
                option_axis_dim=option_axis_dim,
                ratio=ratio,
                chunks=chunks,
                method=method,
                res=res,
                set_nan=set_nan,
            )
            x = x * scale

            if save_nc:
                with ncdf.Dataset(save_nc[0], "r+") as outfile:
                    if "atrack" not in save_nc.dimensions:
                        outfile.createDimension("atrack", lat.shape([0]))
                    if "xtrack" not in outfile.dimensions:
                        outfile.createDimension("atrack", lat.shape([1]))
                    if "latitude" not in outfile.variables:
                        outfile.createVariable("latitude", lat.shape, ("atrack", "xtrack"))
                        outfile["latitude"][:] = lat
                    if "longitude" not in outfile.variables:
                        outfile.createVariable("longitude", lat.shape, ("atrack", "xtrack"))
                        outfile["longitude"][:] = lon
                    if save_nc[1] not in outfile.variables:
                        outfile.createVariable(save_nc[1], lat.shape, ("atrack", "xtrack"))
                    outfile[save_nc[1]] = x
        # end of elif gridded

        # make the plot with pcolormesh
        if vminmax is None:
            vmin = None
            vmax = None
        else:
            vmin, vmax = vminmax

        if self.use_dask:
            x = x.compute()

        if latlon or gridded:
            if self.use_dask:
                lon = lon.compute()
                lat = lat.compute()
            try:
                m = ax.pcolormesh(
                    lon,
                    lat,
                    x,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
            except Exception:
                levels = np.linspace(vmin, vmax, 100) if vminmax else 100
                m = ax.contourf(
                    lon,
                    lat,
                    x,
                    levels,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extend="both",
                )
                ax.set_ylim(np.nanmin(lat), np.nanmax(lat))
                ax.set_xlim(np.nanmin(lon), np.nanmax(lon))
                fig.suptitle("Using contourf", color="red")
        else:
            m = ax.pcolormesh(x, cmap=cmap, vmin=vmin, vmax=vmax)
        m.cmap.set_over(over)
        m.cmap.set_under(under)

        # General plot layout
        if var == "dp":
            lab = r"$\Delta P$"
        elif sv_var:
            lab = sv_var
        else:
            lab = var

        units = self.fetch_units(var)
        if units:
            lab = f"{lab} ({units})"

        if option:
            lab = f"{option} {lab}"

        if len(plt.gcf().axes) == 1:
            if vminmax is not None:
                plt.colorbar(m, label=lab, ax=ax, extend="both")
            else:
                plt.colorbar(m, label=lab, ax=ax)

        if self.start_dates is not None:
            start_dates = sorted([self.start_dates[i] for i in ids])
            ax.set_title(
                f"{datetime.strftime(start_dates[0],'%Y%m%dT%H%M%S')} to {datetime.strftime(start_dates[-1],'%Y%m%dT%H%M%S')}"
            )

        if gridded or latlon:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            if self.is_l3:
                ax.set_ylabel("Latitude index")
                ax.set_xlabel("Longitude index")
            else:
                ax.set_ylabel("along-track index")
                ax.set_xlabel("cross-track index")

        # disable scientific notations in axis numbers
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_yaxis().get_major_formatter().set_useOffset(False)

        if save_path:
            fig.savefig(save_path)
            # also save a pickled version to be able to reopen the interactive matplotlib figure
            with open(f"{save_path}.pickle", "wb") as outfile:
                pickle.dump(fig, outfile)

    def search(self, key: str) -> None:
        """
        Use the msat_nc.search method on one of the files
        key: the string you would like to search for (included in groups or variables)
        """

        self.msat_files[self.ids[0]].search(key)

    def fetch(
        self, key: str, msat_id: int, chunks: Union[str, Tuple] = "auto"
    ) -> Union[np.ndarray, np.ma.masked_array]:
        """
        Use the msat_nc.fetch method on the file corresponding to the given id (from self.ids)
        key: the string you would like to search for (included in groups or variables)
        msat_id: id of the msat_file ()
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        """
        return self.msat_files[self.ids[msat_id]].fetch(key, chunks=chunks)

    def fetch_units(self, key: str) -> str:
        """
        Use the msat_mc.fetch_units method on the first file in the list to return the units of the variable that first matches key
        key: the string you would like to search for (included in groups or variables)
        """
        return self.msat_files[self.ids[0]].fetch_units(key)

    def fetch_varpath(self, key, grp: Optional[str] = None) -> str:
        """
        get the full path to the given variable such that the variable can be selected with self.nc_dset[varpath]
        key: the key you would like to search for
        grp: if given, searches in the given group only
        """
        return self.msat_files[self.ids[0]].fetch_varpath(key, grp=grp)

    def show_all(self) -> None:
        """
        Display all the groups, variables (+dimensions) in the netcdf files
        """
        self.msat_files[self.ids[0]].show_all()

    def show_group(self, grp: str) -> None:
        """
        Show all the variable names and dimensions of a given group
        grp: complete group name
        """
        self.msat_files[self.ids[0]].show_group(grp)

    def get_sv_slice(self, var: str) -> np.ndarray:
        """
        Get the state vector index for the given variable
        var: complete state vector variable name
        """
        return self.msat_files[self.ids[0]].get_sv_slice(var)

    def show_sv(self) -> None:
        """
        Display the state vector variable names
        """
        self.msat_files[self.ids[0]].show_sv()

    def get_var_paths(self) -> List[str]:
        """
        Get a list of all the full variable paths in the netcdf file
        """
        return self.msat_files[self.ids[0]].get_var_paths()

    def has_var(self, var: str) -> bool:
        """
        Check if the netcdf file has the given variable
        """
        return self.msat_files[self.ids[0]].has_var(var)

    def set_use_dask(self, use_dask: bool) -> None:
        self.use_dask = use_dask
        if hasattr(self, "msat_files"):
            for msat_file in self.msat_files.values():
                msat_file.use_dask = use_dask

    def read_dp(self) -> None:
        """
        Fills self.dp for all the files in the collection
        """
        if self.use_dask:
            dask.compute(*[dask.delayed(v.read_dp)() for f, v in self.msat_files.items()])
        else:
            for f, v in self.msat_files.items():
                v.read_dp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="full path to MethaneAIR file")
    args = parser.parse_args()

    msat = msat_file(args.infile)

    return msat


if __name__ == "__main__":
    msat = main()
