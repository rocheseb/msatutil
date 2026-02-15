from __future__ import annotations

import argparse
import glob
import os
import pickle
import re
import sys
import warnings
from collections import OrderedDict
from datetime import datetime
from typing import Annotated, Dict, List, Optional, Sequence, Tuple, Union

import contextily as ctx
import dask
import dask.array as da
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import numpy as np
from dask.diagnostics import ProgressBar
from matplotlib.collections import PolyCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from tqdm import tqdm

from msatutil.make_hist import make_hist
from msatutil.msat_dset import gs_list
from msatutil.msat_nc import MSATError, msat_nc
from msatutil.regrid import Regridder

GOOGLE_TILE_SOURCE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"


def set_clim(z: np.ndarray, n_std: int = 3) -> tuple[float, float]:
    """
    Define color limits as median +/- n_std standard deviations.
    Estimate std from IQR to eliminate outliers.

    Inputs:
        z (np.ndarray): input data
    Outputs:
        clim (tuple[float,float]): output limits
    """
    z = np.ma.MaskedArray(z).filled(np.nan)
    med_z = np.nanmedian(z)
    q25, q75 = np.nanpercentile(z, [25, 75])
    std_z = 0.74 * (q75 - q25)
    clim = (med_z - n_std * std_z, med_z + n_std * std_z)

    return clim


@dask.delayed
def get_msat_file(file_path: str):
    """
    Function to open msat_collection object faster when there are many files

    Inputs:
        file_path (str): full path to the input netcdf file
    """

    return msat_nc(file_path, use_dask=True)


def create_polygons(
    corner_lon: np.ndarray[float], corner_lat: np.ndarray[float]
) -> np.ndarray[Polygon]:
    """
    Create shapely polygons from the corner latitudes and longitudes

    Inputs:
        corner_lon (np.ndarray[float]): corner longitudes (corner,along-track,across-track)
        corner_lat (np.ndarray[float]): corner latitudes (corner,along-track,across-track)

    Outputs:
        polygons (np.ndarray[Polygon]): shapely polygons for each pixel
    """
    nalong, nacross = corner_lat.shape[1:]
    polygons = []
    for i in tqdm(range(nalong), desc="Build polygons", total=nalong):
        for j in range(nacross):
            clon = corner_lon[:, i, j]
            clat = corner_lat[:, i, j]
            if np.isnan(clon).any() or np.isnan(clat).any():
                polygons.append(np.nan)
                continue
            try:
                poly = Polygon([(clon[k], clat[k]) for k in range(4)])
            except Exception as e:
                print(i, j, clon, clat)
                raise e
            polygons.append(poly)
    return np.array(polygons)


def chunked(lst: List, n: int):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = msat_collection(
            flist, date_range=date_range, date_pattern=date_pattern, use_dask=True
        )
    return result


def pcolormesh_or_contourf(x: np.ndarray, y: np.ndarray, z: np.ndarray, ax: plt.Axes, **kwargs):
    """
    Make a 2D plot with x,y,z using pcolormesh. Switch to contourf if there is an error raised.

    Inputs:
        x (np.ndarray): horizontal coordinates
        x (np.ndarray): vertical coordinates
        z (np.ndarray): values
        ax (plt.Axes): matplotlib axis
        **kwargs: passed to pcolormesh and contourf
    Outputs:
        matplotlib.collections.QuadMesh or matplotlib.contour.QuadContourSet

    """
    vmin = kwargs.get("vmin")
    vmax = kwargs.get("vmax")

    try:
        # Use pcolormesh by default
        m = ax.pcolormesh(
            x,
            y,
            z,
            **kwargs,
        )
    except ValueError:
        # pcolormesh can't handle nans in lon and lat
        # fall back to contourf when that is the case
        levels = np.linspace(vmin, vmax, 100) if vmin and vmax else 100
        m = ax.contourf(
            x,
            y,
            z,
            levels,
            extend="both",
            **kwargs,
        )
        ax.figure.suptitle("Using contourf", color="red")

    return m


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
        use_dask: bool = True,
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
                [(file_path, msat_nc(file_path, use_dask=use_dask)) for file_path in self.file_list]
            )
        self.dsets = {key: val.nc_dset for key, val in self.msat_files.items()}

        # These would potentially generated multiple times with pmesh_prep
        # save them as class parameters to avoid re-reading / re-computing them
        self.vertices = None
        self.lon = None
        self.lat = None

        self.is_l1 = self.msat_files[self.ids[0]].is_l1
        self.is_l2 = self.msat_files[self.ids[0]].is_l2
        self.is_l2_met = self.msat_files[self.ids[0]].is_l2_met
        self.is_postproc = self.msat_files[self.ids[0]].is_postproc
        self.is_l3 = self.msat_files[self.ids[0]].is_l3
        self.set_lon_lat_vars()
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
        for k, v in globals().items():
            if v is self:
                # after closing the object won't be accessible anymore
                del globals()[k]

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

    def __iter__(self):
        return iter(self.msat_files.items())

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

        Keys are from self.common_dim_set
        Values are the corresponding dimension indices for the variable

        e.g. the index of the spectral_channel dimension for variable var is
        get_dim_map(var)["spectral_channel"]
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

        self.fig.set_size_inches(10, 8)

    def hist(
        self,
        ax: plt.Axes = None,
        scale: float = 1.0,
        var: str = None,
        sv_var: Optional[str] = None,
        extra_id: Optional[int] = None,
        extra_id_dim: Optional[str] = None,
        ids: Optional[List[int]] = None,
        ratio: bool = False,
        option: Optional[str] = None,
        option_axis_dim: str = "spectral_channel",
        chunks: Union[str, Tuple] = "auto",
        set_nan: Optional[float] = None,
        exp_fmt: bool = True,
        **kwargs,
    ) -> None:
        """
        Plot a histogram of the given variable

        ax: matplotlib axes object
        scale: quantity to multiply the variable with (can be useful to avoid overflow in the standard deviation of column amounts)
        ## msat_collection.pmesh_prep arguments:
        var: key contained in the variable to search (uses msat_nc fetch method)
        sv_var: sv_var must be one of SpecFitDiagnostics/APrioriState or SpecFitDiagnostics/APosterioriState, and var must be the exact SubStateName of the state vector variable
        extra_id: integer to slice a third index (e.g. along wmx_1 for Radiance_I (wmx_1,jmx,imx)) only does something for 3D variables
        extra_id_dim: name of the dimension along which extra_id will be selected
        ids: list of ids corresponding to the keys of self.ids, used to select which files are concatenated
        ratio: if True, return the variable divided by its median
        option: can be used to get stats from a 3d variable (any numpy method e.g. 'max' 'nanmax' 'std')
        option_axis_dim: the axis along which the stat is applied
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        set_nan (Optional[float]): this value will be replaced with nan after a pmesh_prep call
        ## make_hist arguments:
        exp_fmt: if True, use .3e format for stats in the histogram legend. If false use .2f format
        kwargs: passed to matplotlib.pyplot.hist
        """
        if ax is None:
            self.init_plot(1)
            fig, ax = self.fig, self.ax
            fig.set_size_inches(8, 5)

        x = self.pmesh_prep(
            var,
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

        x *= scale

        make_hist(ax, x, exp_fmt=exp_fmt, **kwargs)

    def pmesh_prep(
        self,
        var: str,
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
        var: key contained in the variable to search or full variable path
        sv_var: sv_var must be one of SpecFitDiagnostics/APrioriState or SpecFitDiagnostics/APosterioriState, and var must be the exact SubStateName of the state vector variable
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
        if sv_var is not None:
            sv_slice = self.get_sv_slice(sv_var)
        else:
            nc_slice = [slice(None)]

        if var == "dp" and self.msat_files[self.ids[0]].dp is None:
            self.read_dp()

        use_get_var = "/" in var
        if use_get_var:
            var_path = var
            var = var.split("/")[-1]
        else:
            var_path = self.fetch_varpath(var)

        if var_path is None:
            raise MSATError(f"Not variables match '{var}'")

        var_dim_map = self.get_dim_map(var_path)
        if sv_var is not None:
            if var_path not in [
                "SpecFitDiagnostics/APrioriState",
                "SpecFitDiagnostics/APosterioriState",
                "SpecFitDiagnostics/IntermediateState",
            ]:
                raise MSATError(
                    'var must be one of ["SpecFitDiagnostics/APrioriState","SpecFitDiagnostics/APosterioriState"] when sv_var is given'
                )
            nc_slice = [slice(None) for dim in var_dim_map]
            nc_slice[var_dim_map["xmx"]] = sv_slice
        if self.is_l3:
            atrack_axis = 0
        else:
            atrack_axis = var_dim_map["atrack"]
        x = []
        tqdm_disable = len(list(ids.values())) < 50
        for msat_id, msat_file_path in tqdm(
            ids.items(),
            total=len(ids),
            disable=tqdm_disable,
            leave=False,
            desc=var_path,
        ):
            if var_path == "dp":
                x.append(self.msat_files[msat_file_path].dp)
            else:
                if use_get_var:
                    x.append(self.get_var(var_path, msat_id, chunks=chunks)[tuple(nc_slice)])
                else:
                    x.append(self.fetch(var, msat_id, chunks=chunks)[tuple(nc_slice)])

        if self.use_dask:
            x = da.concatenate(x, axis=atrack_axis)
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float16)
            # x[da.greater(x, 1e29)] = np.nan
            if not self.is_l3:
                x = x.rechunk({atrack_axis: "auto"})
        else:
            x = np.concatenate(x, axis=atrack_axis)
            if np.issubdtype(x.dtype, np.integer):
                x = x.astype(np.float16)
            #x[np.greater(x, 1e29)] = np.nan

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
        sv_var: sv_var must be one of SpecFitDiagnostics/APrioriState or SpecFitDiagnostics/APosterioriState, and var must be the exact SubStateName of the state vector variable
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

        regridder = Regridder(lon_lim[0], lon_lim[1], lat_lim[0], lat_lim[1], (res, res))

        x_grid_list = []
        for i, ids_slice in enumerate(chunked_ids):
            sys.stdout.write(f"\rgrid_prep now doing chunk {i+1:>3}/{len(chunked_ids)}")
            sys.stdout.flush()

            x = self.pmesh_prep(
                var,
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
            ).compute()

            lat = self.pmesh_prep(
                self.lat_var, ids=ids_slice, chunks=chunks, use_valid_xtrack=use_valid_xtrack
            ).compute()
            lon = self.pmesh_prep(
                self.lon_var, ids=ids_slice, chunks=chunks, use_valid_xtrack=use_valid_xtrack
            ).compute()

            _, _, x_grid = regridder(lon, lat, x)

            x_grid_list.append(x_grid)

        stacked_grid = da.stack(x_grid_list, axis=0)
        x_grid_avg = da.nanmean(stacked_grid, axis=0)

        return regridder.lon_grid, regridder.lat_grid, x_grid_avg

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
            _ = self.heatmap(**kwargs)

    def set_lon_lat_vars(self):
        if self.is_l3:
            lon_str = "lon"
            lat_str = "lat"
        elif self.is_postproc:
            lon_str = "geolocation/longitude"
            lat_str = "geolocation/latitude"
        elif self.is_l1 or self.is_l2_met:
            lon_str = "Geolocation/Longitude"
            lat_str = "Geolocation/Latitude"
        elif self.is_l2:
            lon_str = "Level1/Longitude"
            lat_str = "Level1/Latitude"

        self.lon_var = lon_str
        self.lat_var = lat_str

    def save_geolocation(
        self,
        ids: Optional[List[int]] = None,
        chunks: Union[str, Tuple] = "auto",
        use_valid_xtrack: bool = False,
        use_corners: bool = False,
    ) -> str:
        """
        Since we may read longitude and latitude many times when using self.heatmap, handle them separately and keep the last set
        of ids saved to avoid having to recompute them
        """
        if ids is None:
            ids = self.ids
        else:
            ids = {i: self.ids[i] for i in ids}

        key = f"{list(ids.keys())}_{use_valid_xtrack}"

        if use_corners and self.vertices is not None and key in self.vertices:
            return key

        if not use_corners and self.lat is not None and key in self.lat:
            return key

        if use_corners:
            lon_var = "CornerLongitude" if not self.is_postproc else "longitude_bounds"
            lat_var = "CornerLatitude" if not self.is_postproc else "latitude_bounds"
        else:
            lon_var = self.lon_var
            lat_var = self.lat_var

        lat = self.pmesh_prep(
            lat_var,
            ids=ids,
            chunks=chunks,
            use_valid_xtrack=use_valid_xtrack,
        )

        lon = self.pmesh_prep(
            lon_var,
            ids=ids,
            chunks=chunks,
            use_valid_xtrack=use_valid_xtrack,
        )

        # when using corners, only save vertices
        # self.lon and self.lat are never the corners
        if not use_corners:
            if self.use_dask:
                self.lon = {key: lon.compute()}
                self.lat = {key: lat.compute()}
            else:
                self.lon = {key: lon}
                self.lat = {key: lat}
        else:
            if self.use_dask:
                lon = lon.compute()
                lat = lat.compute()
            if lon.shape.index(4) == 2:
                lon = lon.transpose((2, 0, 1))
                lat = lat.transpose((2, 0, 1))
            if self.vertices is None or key not in self.vertices:
                polygons = create_polygons(lon, lat)
                self.vertices = {
                    key: [
                        np.array(p.exterior.coords) if p is not np.nan else np.nan for p in polygons
                    ]
                }

        return key

    def heatmap(
        self,
        var: str,
        sv_var: Optional[str] = None,
        over: Optional[str] = "red",
        under: Optional[str] = "hotpink",
        latlon: bool = False,
        use_corners: bool = False,
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
        set_nan: Optional[float] = None,
        use_valid_xtrack: bool = False,
        colorbar_label: Optional[str] = None,
        cb_fraction: float = 0.04,
        mask: Optional[np.ndarray] = None,
        use_set_clim: bool = False,
        **kwargs,  # matplotlib pcolormesh arguments
    ) -> np.ndarray[float]:
        """
        Make a heatmap of the given variable
        var (str): key contained in the variable to search (uses msat_nc fetch method)
        sv_var: sv_var must be one of SpecFitDiagnostics/APrioriState or SpecFitDiagnostics/APosterioriState, and var must be the exact SubStateName of the state vector variable
        vminmax (Optional[Annotated[Sequence[float], 2]]): min and max value to be shown with the colorbar
        over (Optional[str]): color to use above the max of the color scale (only used if vminmax is not None)
        under (Optional[str]): color to use under the min of the color scale (only used if vminmax is not None)
        latlon (bool): if True, make the plot on latitude/longitude instead of xtrack/atrack
        use_corners (bool): if True, make the plot using polygons from the corner lat/lon
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
        colorbar_label (Optional[str]): if givem set as the colorbar label
        cb_fraction (float): controls the size of the colorbar
        mask (Optional[np.ndarray]): Boolean arrays where True will be set to nan before plotting
        use_set_clim (bool): if True, use the set_clim function to set the color range, overrides vmin and vmax
        kwargs: passed to make_heatmap, _make_heatmap_with_background_tile, and the pcolormesh call

        Outputs:
            the pmesh_prep (or grid_prep if lon_lim/lat_lim are given) output for the plotted variable
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

        if use_corners:
            latlon = True  # use_corners implies plotting in latitudes/longitudes

        if ylim:
            ax.set_ylim(*ylim)

        gridded = (lon_lim is not None) and (lat_lim is not None)

        if gridded and not self.use_dask:
            raise MSATError("/!\\ the gridded argument only works when self.use_dask is True")

        if gridded:
            if use_corners:
                print("use_corners has no effect when using grid_prep")
            use_corners = False
            latlon = True

        if latlon:
            use_valid_xtrack = True

        if not gridded:
            if latlon:
                geo_key = self.save_geolocation(
                    ids=ids,
                    chunks=chunks,
                    use_valid_xtrack=use_valid_xtrack,
                    use_corners=use_corners,
                )
                if not use_corners:
                    lon = self.lon[geo_key]
                    lat = self.lat[geo_key]
                else:
                    lon = None
                    lat = None

            x = self.pmesh_prep(
                var,
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
            if self.use_dask:
                x = x.compute()
        elif gridded:
            lon, lat, x = self.grid_prep(
                var,
                lon_lim,
                lat_lim,
                n=n,
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

        # Define some plot attributes
        if colorbar_label is not None:
            pass
        else:
            if var == "dp":
                colorbar_label = r"$\Delta P$"
            elif sv_var:
                colorbar_label = sv_var
            else:
                colorbar_label = var

            units = self.fetch_units(var)
            if units:
                colorbar_label = f"{colorbar_label} ({units})"

            if option:
                colorbar_label = f"{option} {colorbar_label}"

        if self.start_dates is not None:
            start_dates = sorted([self.start_dates[i] for i in ids])
            try:
                if len(start_dates) > 1:
                    title = f"{datetime.strftime(start_dates[0],'%Y%m%dT%H%M%S')} to {datetime.strftime(start_dates[-1],'%Y%m%dT%H%M%S')}"
                else:
                    title = f"{datetime.strftime(start_dates[0],'%Y%m%dT%H%M%S')}"
            except Exception:
                title = ""

        if latlon:
            xlabel = "Longitude"
            ylabel = "Latitude"
        else:
            if self.is_l3:
                xlabel = "Longitude index"
                ylabel = "Latitude index"
            else:
                xlabel = "cross-track index"
                ylabel = "along-track index"

        if mask is not None:
            x[mask] = np.nan

        # make the plot
        msat_collection.make_heatmap(
            ax,
            x,
            lon=lon if latlon else None,
            lat=lat if latlon else None,
            vertices=self.vertices.get(geo_key) if use_corners else None,
            over=over,
            under=under,
            colorbar_label=colorbar_label,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            cb_fraction=cb_fraction,
            use_set_clim=use_set_clim,
            **kwargs,
        )

        if save_path:
            fig.savefig(save_path)
            # also save a pickled version to be able to reopen the interactive matplotlib figure
            with open(f"{save_path}.pickle", "wb") as outfile:
                pickle.dump(fig, outfile)

        return x

    @staticmethod
    def _make_heatmap_with_background_tile(
        ax: plt.Axes,
        lon: np.ndarray,
        lat: np.ndarray,
        z: np.ndarray,
        latlon_padding: float = 0,
        latlon_step: float = 0.5,
        lab_prec: int = 1,
        tile_source: str = GOOGLE_TILE_SOURCE,
        gridlines: bool = True,
        add_basemap: bool = True,
        lon_extent: Optional[tuple[float, float]] = None,
        lat_extent: Optional[tuple[float, float]] = None,
        scalebar_km: float = 50,
        scalebar_thickness: float = 500,
        rounding_factor: int = 2,
        **kwargs,
    ):
        """
        ax: matplotlib axes
        lon (np.ndarray): longitude
        lat (np.ndarray): latitude
        z (np.ndarray): variable for the heatmap
        latlon_padding (float): add above/below the max/min lat and lon
        latlon_step (float): the spacing between lat/lon ticks
        lab_prec (int): number of decimals for the lat/lon tick labels
        tile_source (str): the WMTS URL for the background tiles, defaults to Google imagery
        gridlines (bool): if True, draw gridlines
        add_basemap (bool): if False, will not add the background tiles
        lon_extent (Optional[tuple[float, float]]): prescribed min/max longitude
        lat_extent (Optional[tuple[float, float]]): prescribed min/max latitude
        scalebar_km (float): the length of the scalebar in km
        rounding_factor (int): use for rounding the starting lat/lon label, 2 is nearest .5
        kwargs: passed to the pcolormesh call
        """
        vmin = kwargs.get("vmin")
        vmax = kwargs.get("vmax")

        if lon_extent is not None:
            lon_min, lon_max = lon_extent
        else:
            lon_min, lon_max = np.nanmin(lon) - latlon_padding, np.nanmax(lon) + latlon_padding
        if lat_extent is not None:
            lat_min, lat_max = lat_extent
        else:
            lat_min, lat_max = np.nanmin(lat) - latlon_padding, np.nanmax(lat) + latlon_padding

        lon_start = np.floor(lon_min * rounding_factor) / rounding_factor
        lat_start = np.floor(lat_min * rounding_factor) / rounding_factor

        delta_lat = lat_max - lat_min
        delta_lon = lon_max - lon_min
        n_ticks = np.max([int(delta_lat / latlon_step) + 2, int(delta_lon / latlon_step) + 2])
        lon_ticks = lon_start + np.arange(n_ticks) * latlon_step
        lat_ticks = lat_start + np.arange(n_ticks) * latlon_step

        # The background tiles are on Web Mercator
        # Transform coordinates to Web Mercator (EPSG:3857)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        # Transform lon/lat ticks to x/y in Web Mercator projection
        x_ticks, y_ticks = transformer.transform(lon_ticks, lat_ticks)

        if lon.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)

        # Now transform the main lon, lat arrays for plotting (after tick creation)
        x, y = transformer.transform(lon, lat)
        xmin, ymin = transformer.transform(lon_min, lat_min)
        xmax, ymax = transformer.transform(lon_max, lat_max)

        m = pcolormesh_or_contourf(x, y, z, ax, **kwargs)

        # Format longitude/latitude labels
        def format_latlon(value, is_lon):
            """
            Format the lat/lon to show degrees and direction (N, S, E, W).
            """
            if is_lon:
                direction = "E" if value >= 0 else "W"
                value = abs(value)
            else:
                direction = "N" if value >= 0 else "S"
                value = abs(value)

            return f"{value:.{lab_prec}f}°{direction}"

        ax.set_aspect("equal")

        # Set the ticks on the plot
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Apply formatted labels
        ax.set_xticklabels([format_latlon(lon, is_lon=True) for lon in lon_ticks])
        ax.set_yticklabels([format_latlon(lat, is_lon=False) for lat in lat_ticks])
        if gridlines:
            ax.grid(linestyle="--", alpha=0.4)

        if add_basemap:
            # Add Google Satellite tiles as the basemap
            ctx.add_basemap(ax, source=tile_source, crs="EPSG:3857")

        lat_mid = lat_min + (lat_max - lat_min) / 2
        scalebar_merc = 1e3 * scalebar_km * np.cos(np.deg2rad(lat_mid))

        # Add scalebar
        scalebar = AnchoredSizeBar(
            ax.transData,  # Transformation to use
            scalebar_merc,  # Length of the scalebar in data units (e.g., 100 km = 100,000 meters for Web Mercator)
            f"{scalebar_km} km",  # Label for the scalebar
            "lower left",  # Location of the scalebar
            pad=0.3,  # Padding around the scalebar
            color="white",  # Color of the scalebar
            frameon=False,  # Remove the surrounding frame
            size_vertical=scalebar_thickness,  # Thickness of the scalebar
            fontproperties=fm.FontProperties(size=12),  # Font size of the label
            label_top=True,  # Label above the scalebar
        )
        ax.add_artist(scalebar)

        return m

    @staticmethod
    def make_heatmap(
        ax: plt.Axes,
        x: np.ndarray[float],
        lon: Optional[np.ndarray[float]] = None,
        lat: Optional[np.ndarray[float]] = None,
        vertices: Optional[np.ndarray[float]] = None,
        over: Optional[str] = "red",
        under: Optional[str] = "hotpink",
        colorbar_label: str = "",
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        cb_fraction: float = 0.04,
        basic: bool = False,
        use_set_clim: bool = False,
        **kwargs,
    ):
        """
        Make a heatmap plot in ax

        Inputs:
            ax (plt.Axes): matplotlib axes
            x (np.ndarray[float]): 2D data array
            lon (Optional[np.ndarray[float]]): longitude array
            lat (Optional[np.ndarray[float]]): latitude array
            vertices (Optional[np.ndarray[float]]): if given, use matplotlib.collections.PolyCollection
            over (Optional[str]): color to use above the max of the color scale (only used if vminmax is not None)
            under (Optional[str]): color to use under the min of the color scale (only used if vminmax is not None)
            colorbar_label (str): label for the colorbar
            xlabel (str): horizontal axis label
            ylabel (str): vertical axis label
            title (str): plot title
            cb_fraction (float): controls the size of the colorbar as a fraction of the plot
            basic (bool): if True, don't add a background imagery tile to latlon plots
            use_set_clim (bool): if True, use the set_clim function to set the color range, overrides vmin and vmax
            **kwargs: matplotlib pcolormesh arguments
        """
        vmin = kwargs.get("vmin")
        vmax = kwargs.get("vmax")
        cmap = kwargs.get("cmap", "viridis")

        if use_set_clim:
            vmin, vmax = set_clim(x)
            kwargs["vmin"] = vmin
            kwargs["vmax"] = vmax

        if vertices is not None:
            # Make the plot with matplotlib.collections.PolyCollection
            # This is the most accurate representation of the geolocated data but also the slowest
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            flat_x = x.flatten()
            ids = np.isfinite(flat_x) & np.array([p is not np.nan for p in vertices], dtype=bool)
            color_list = matplotlib.colormaps[cmap](norm(flat_x[ids]))
            valid_vertices = [v for i, v in enumerate(vertices) if ids[i]]
            poly_collection = PolyCollection(
                valid_vertices,
                edgecolors=color_list,
                facecolors=color_list,
                cmap=cmap,
                norm=norm,
            )
            m = ax.add_collection(poly_collection)
            all_points = np.concatenate(valid_vertices)
            ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
            ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
        elif lat is not None and lon is not None:
            if basic:
                m = pcolormesh_or_contourf(lon, lat, x, ax, **kwargs)
            else:
                m = msat_collection._make_heatmap_with_background_tile(
                    ax,
                    lon,
                    lat,
                    x,
                    **kwargs,
                )
        else:
            # plotting against along-track and across-track indices
            m = ax.pcolormesh(x, **kwargs)
            # disable scientific notations in axis numbers
            ax.get_xaxis().get_major_formatter().set_useOffset(False)
            ax.get_yaxis().get_major_formatter().set_useOffset(False)
        if over is not None:
            m.cmap.set_over(over)
        if under is not None:
            m.cmap.set_under(under)

        # General plot layout
        if vmin is not None and vmax is not None:
            extend = "both"
        elif vmin is not None and vmax is None:
            extend = "min"
        elif vmax is not None and vmin is None:
            extend = "max"
        elif vmin is None and vmax is None:
            extend = "neither"

        has_cbar = any([hasattr(c, "colorbar") and c.colorbar is not None for c in ax.collections])
        if not has_cbar:
            plt.colorbar(
                m, label=colorbar_label, ax=ax, extend=extend, fraction=cb_fraction, pad=0.02
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def search(self, key: str) -> None:
        """
        Use the msat_nc.search method on one of the files
        key: the string you would like to search for (included in groups or variables)
        """

        self.msat_files[self.ids[0]].search(key)

    def get_var(self, varpath: str, msat_id: int, chunks: Union[str, Tuple] = "auto"):
        """
        Use the msat_nc.get_var method on the file corresponding to the given id (from self.ids)
        key: the string you would like to search for (included in groups or variables)
        msat_id: id of the msat_file ()
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        """

        return self.msat_files[self.ids[msat_id]].get_var(varpath, chunks=chunks)

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

    def fetch_varpath(self, key) -> str:
        """
        get the full path to the given variable such that the variable can be selected with self.nc_dset[varpath]
        key: the key you would like to search for
        """
        return self.msat_files[self.ids[0]].fetch_varpath(key)

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

    def show_var(self, varpath: str) -> ncdf.Variable:
        """
        Returns the given variable

        varpath (str): full path to the variable
        """
        return self.msat_files[self.ids[0]].show_var(varpath)

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
        if self.is_l2 and not self.is_postproc:
            self.msat_files[self.ids[0]].show_sv()
        else:
            print("No state vector to show")

    def show_err(self) -> None:
        """
        Display the error components
        """
        if self.is_l2 and not self.is_postproc:
            self.msat_files[self.ids[0]].show_err()
        else:
            print("No error components to show")

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

    def dimensions(self) -> dict[str, ncdf.Dimension]:
        """
        Get the netCDF dimensions
        """
        return self.msat_files[self.ids[0]].dimensions()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="full path to MethaneAIR file")
    args = parser.parse_args()

    msat = msat_nc(args.infile)

    return msat


if __name__ == "__main__":
    msat = main()
