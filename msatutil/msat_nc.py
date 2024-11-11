from __future__ import annotations
import os
import numpy as np
import netCDF4 as ncdf
from typing import Optional, Tuple, Union, List, Dict
import dask.array as da
from msatutil.msat_dset import msat_dset


class MSATError(Exception):
    pass


class msat_nc:
    """
    This class holds a netCDF.Dataset for a MethaneSAT/AIR L1B or L2 file.
    It contains methods that help navigate the dataset more quickly
    """

    def __init__(self, infile: str, use_dask: bool = False) -> None:
        self.use_dask = use_dask

        if infile.startswith("gs://"):
            # will need to figure out how to determine a gs path exists and then update this
            self.exists = True
        else:
            self.exists = os.path.exists(infile)
        if not self.exists:
            raise MSATError(f"Wrong path: {infile}")
        else:
            self.nc_dset = msat_dset(infile)

        self.inpath = infile
        self.dp = None
        self.datetimes = None
        self.is_l3 = "Provenance" in self.nc_dset.groups and hasattr(
            self.nc_dset["Provenance"], "msat_level3"
        )
        self.is_labels = "cloud_mask" in self.nc_dset.variables
        self.is_postproc = "product_co2proxy" in self.nc_dset.groups
        self.is_l2_met = "Surface_Band1" in self.nc_dset.groups
        self.is_l2 = not self.is_l2_met and (("Level1" in self.nc_dset.groups) or self.is_postproc)
        self.is_l1 = True not in [
            self.is_l2,
            self.is_l2_met,
            self.is_postproc,
            self.is_l3,
            self.is_labels,
        ]
        self.varpath_list = None

        """
        Dictionary that maps all the dimensions names across L1/L2/L3 file versions to a common set of names
        use the get_dim_map method to get a mapping of a given variables dimensions that use the dimensions names from the common set
        common set:
            [
                "one",
                "xtrack",
                "atrack",
                "xtrack_edge",
                "atrack_edge",
                "lev",
                "lev_edge",
                "corner",
                "spectral_channel",
                "xmx", # state vector dimension
                "nsubx", # when the per-iteration diagnostics are saved, the iterations are along this dimension
                "err_col", # error terms for the total columns
                "err_proxy", # error terms for the Proxy mole fraction
            ]
        """
        self.dim_name_map = {
            "one": "one",
            "o": "one",
            "imx": "xtrack",
            "xtrack": "xtrack",
            "across_track": "xtrack",
            "x": "xtrack",
            "imx_e": "xtrack_edge",
            "xtrack_edge": "xtrack_edge",
            "jmx": "atrack",
            "atrack": "atrack",
            "along_track": "atrack",
            "y": "atrack",
            "jmx_e": "atrack_edge",
            "atrack_edge": "atrack_edge",
            "lmx": "lev",
            "lev": "lev",
            "level": "lev",
            "levels": "lev",
            "ch4band_levels": "ch4_lev",
            "o2band_levels": "o2_lev",
            "z": "lev",
            "lmx_e": "lev_edge",
            "lev_edge": "lev_edge",
            "ze": "lev_edge",
            "vertices": "corner",
            "four": "corner",
            "c": "corner",
            "corner": "corner",
            "nv": "corner",
            "w1": "spectral_channel",
            "wmx_1": "spectral_channel",
            "spectral_channel": "spectral_channel",
            "xmx": "xmx",  # state vector dimension
            "nsubx": "nsubx",
            "p1": "one",
            "p2": "two",
            "p3": "three",
            "eci": "three",
            "w1_alb": "alb_wvl",
            "k1_alb": "alb_kernel",
            "p1_alb": "alb_poly",
            "w2_alb": "alb_wvl",
            "k2_alb": "alb_kernel",
            "p2_alb": "alb_poly",
            "iter_x": "iter_x",
            "iter_w": "iter_w",
            "err_col": "err_col",
            "err_proxy": "err_proxy",
            "lat": "lat",  # L3 dims
            "lon": "lon",  # L3 dims
        }

        self.dim_size_map = {
            self.dim_name_map[dim]: dim_item.size
            for dim, dim_item in self.nc_dset.dimensions.items()
        }

    def __enter__(self) -> None:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        self.nc_dset.close()

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"""msat_nc:
        use_dask: {self.use_dask}
        is_labels: {self.is_labels}
        is_l1: {self.is_l1}
        is_l2: {self.is_l2}
        is_l3: {self.is_l3}
        is_postproc: {self.is_postproc}
        is_l2_met: {self.is_l2_met}
        """

    def dimensions(self) -> dict[str, ncdf.Dimension]:
        """
        Return the netCDF dimensions
        """
        return self.nc_dset.dimensions

    def read_dp(self) -> None:
        """
        Getting the retrieved minus prior surface pressure in self.dp
        """
        if self.is_l2:
            if "o2dp_fit_diagnostics" in self.nc_dset.groups:
                self.dp = self.nc_dset["o2dp_fit_diagnostics/bias_corrected_delta_pressure"][
                    tuple()
                ]
            else:
                sp_slice = self.get_sv_slice("SurfacePressure")

                prior = "SpecFitDiagnostics/APrioriState"
                post = "SpecFitDiagnostics/APosterioriState"
                if not self.use_dask:
                    self.dp = self.nc_dset[post][sp_slice] - self.nc_dset[prior][sp_slice]
                else:
                    read_chunks = self.nc_dset[post].chunking()
                    if type(read_chunks) is str:
                        read_chunks = "auto"
                    self.dp = (
                        da.from_array(self.nc_dset[post], chunks=read_chunks)
                        - da.from_array(self.nc_dset[prior], chunks=read_chunks)
                    )[sp_slice]

    def get_var(
        self,
        varpath: str,
        chunks: Union[str, Tuple] = "auto",
    ) -> Union[np.ndarray, np.ma.masked_array, da.core.Array]:
        """
        return a variable array from the netcdf file
        varpath: complete variable path
        chunks: when self.use_dask is True, sets the chunk size for dask arrays
        """
        if varpath.lower() == "dp":
            return self.dp
        elif varpath.lower() in ["datetime", "datetimes"]:
            return self.datetimes
        else:
            if self.use_dask:
                read_chunks = self.nc_dset[varpath].chunking()
                if type(read_chunks) is str:
                    read_chunks = "auto"
                return (
                    da.from_array(self.nc_dset[varpath], chunks=read_chunks)
                    .rechunk(chunks)
                    .astype(float)
                )
            return self.nc_dset[varpath][tuple()].astype(float)

    def get_units(self, nc_var: ncdf.Variable) -> str:
        """
        get the units of the given variable
        varpath: complete variable path
        """
        units = ""
        if hasattr(nc_var, "units"):
            units = nc_var.units
        elif hasattr(nc_var, "unit"):
            units = nc_var.unit

        return units

    def show_var(self, varpath: str) -> ncdf.Variable:
        """
        display the given variable metadata
        varpath: complete variable path
        """
        return self.nc_dset[varpath]

    def show_all(self, start=None, indent="") -> None:
        """
        Show all the groups names, variable names, and variable dimensions
        """
        if start is None:
            start = self.nc_dset

        for var in start.variables:
            print(indent, var, start[var].dimensions)

        if start.groups:
            for grp in start.groups:
                print(indent, "Group:", grp)
                self.show_all(start=start[grp], indent=indent + "\t")

    def show_group(self, grp: str) -> None:
        """
        Show all the variable names and dimensions of a given group
        grp: complete group name
        """
        self.show_all(start=self.nc_dset[grp])

    def show_sv(self) -> None:
        """
        Display the state vector variable names
        """
        if not self.is_l2 or self.is_postproc:
            return
        sv_dict = self.nc_dset["SpecFitDiagnostics"]["APosterioriState"].__dict__
        for key, val in sv_dict.items():
            if type(val) == str:
                val = val.strip()
            print(f"{key.strip()}: {val}")

    def show_err(self) -> None:
        if self.is_postproc or not self.is_l2:
            return
        for err_var in ["GasColumnErrorComponents", "ProxyXGasErrorComponents"]:
            print(f"\n{err_var}")
            err_dict = self.show_var(self.fetch_varpath(err_var)).__dict__
            for key, val in err_dict.items():
                if type(val) == str:
                    val = val.strip()
                print(f"{key.strip()}: {val}")

    def get_sv_slice(self, var: str) -> np.ndarray:
        """
        Get the state vector index for the given variable
        var: complete state vector variable name
        """
        if not self.is_l2:
            return slice(None)

        sv_dict = self.nc_dset["SpecFitDiagnostics"]["APosterioriState"].__dict__

        for key, val in sv_dict.items():
            if key.startswith("SubStateName") and val.strip() == var:
                num = int(key.split("_")[-1]) - 1
                sv_slice = slice(sv_dict["SubState_Idx0"][num] - 1, sv_dict["SubState_Idxf"][num])
                break

        return sv_slice

    def search(self, key: str, start=None) -> None:
        """
        print out groups and variables that include the key (all lowercase checks)
        key: the string you would like to search for (included in groups or variables)
        """
        key = key.lower()

        if start is None:
            start = self.nc_dset

        for var in start.variables:
            varpath = _get_full_variable_path(start[var])
            if key in var.lower():
                print(f"{varpath} {start[var].dimensions}")
            if var in ["APosterioriState", "APrioriState"]:
                sv_dict = start[var].__dict__
                for sv_key, val in sv_dict.items():
                    if sv_key.startswith("SubStateName") and key in val.strip().lower():
                        sv_slice = self.get_sv_slice(val.strip())
                        print(
                            f"{varpath} {start[var].dimensions} \tSV_VAR: {val.strip()} \tSV_SLICE: {list(range(sv_slice.start,sv_slice.stop))}"
                        )

        for grp in start.groups:
            if key in grp.lower():
                print(f"GROUP: {grp}")
            self.search(key, start=start[grp])

    def fetch(
        self,
        key: str,
        chunks: Union[str, Tuple] = "auto",
        start: Optional[Union[ncdf.Datasat, ncdf.Group]] = None,
    ) -> Union[np.ndarray, np.ma.masked_array, da.core.Array]:
        """
        Recursively retrieves the first variable that matches the key (case-insensitive).
        key: The string you would like to search for (included in groups or variables).
        chunks: When self.use_dask is True, sets the chunk size for dask arrays.
        start: starting Dataset or Group, defaults to self.nc_dset
        """
        key = key.lower()

        if key == "dp":
            return self.dp
        elif key in ["datetime", "datetimes"]:
            return self.datetimes

        if start is None:
            start = self.nc_dset

        for var in start.variables:
            if key in var.lower():
                if self.use_dask:
                    read_chunks = start[var].chunking()
                    if type(read_chunks) is str:
                        read_chunks = "auto"
                    return (
                        da.from_array(start[var], chunks=read_chunks).rechunk(chunks).astype(float)
                    )
                return start[var][tuple()].astype(float)

        # Search in subgroups recursively
        for grp in start.groups:
            result = self.fetch(key, chunks=chunks, start=start[grp])
            if result is not None:
                return result

    def fetch_units(
        self,
        key: str,
        start: Optional[Union[ncdf.Datasat, ncdf.Group]] = None,
    ) -> str:
        """
        Recursively retrieves the units of the variable that matches the key.
        key: The string you would like to search for (included in groups or variables).
        start: starting Dataset or Group, defaults to self.nc_dset
        """
        key = key.lower()

        if key == "dp":
            return "hPa"
        elif key in ["datetime", "datetimes"]:
            return "hours since 1985-01-01"

        if start is None:
            start = self.nc_dset

        # Search in variables of the current group
        for var in start.variables:
            if key in var.lower():
                return self.get_units(start[var])

        # Search in subgroups recursively
        for grp in start.groups:
            units = self.fetch_units(key, start=start[grp])
            if units is not None:
                return units

    def fetch_varpath(
        self, key: str, start: Optional[Union[ncdf.Datasat, ncdf.Group]] = None
    ) -> Union[str, None]:
        """
        get the full path to the given variable such that the variable can be selected with self.nc_dset[varpath]
        key: the key you would like to search for
        grp: if given, searches in the given group only
        start: starting Dataset or Group, defaults to self.nc_dset
        """
        key = key.lower()
        if key == "dp":
            return "dp"

        if start is None:
            start = self.nc_dset

        for var in start.variables:
            if key in var.lower():
                return _get_full_variable_path(start[var])

        for grp in start.groups:
            result = self.fetch_varpath(key, start=start[grp])
            if result is not None:
                return result

    @staticmethod
    def nctime_to_pytime(
        nc_time_var: ncdf.Variable,
    ) -> Union[np.ndarray, np.ma.masked_array]:
        """
        Convert time in a netCDF variable to an array of Python datetime objects.
        """
        if hasattr(nc_time_var, "calendar"):
            calendar = nc_time_var.calendar
        else:
            calendar = "standard"
        if not hasattr(nc_time_var, "units"):
            print(
                "Time variable has no units for num2date conversion, assume hours since 1985-01-01"
            )
            units = "hours since 1985-01-01"
        else:
            units = nc_time_var.units

        return ncdf.num2date(
            nc_time_var[tuple()],
            units=units,
            calendar=calendar,
            only_use_cftime_datetimes=False,
        )

    def get_dim_map(self, var_path: str) -> Dict[str, int]:
        """
        For a given key, use fetch_varpath to inspect the corresponding variable and
        return a map of its dimension axes {'dim_name':dim_axis}
        """
        if var_path.lower() == "dp":
            if len(self.dp.shape) == 3:
                var_dim_map = {"one": 0, "atrack": 1, "xtrack": 2}
            else:
                var_dim_map = {"atrack": 0, "xtrack": 1}
        else:
            var_dims = self.nc_dset[var_path].dimensions
            var_dim_map = {self.dim_name_map[dim]: var_dims.index(dim) for dim in var_dims}

        return var_dim_map

    def get_valid_xtrack(self, varpath: Optional[str] = None) -> slice:
        """
        Get the valid cross track indices

        varpath (Optional[str]): full variable path. If given, use it to get the valid slice
        """
        if self.is_l3 or self.is_labels:
            return slice(None)

        is_msat = self.dim_size_map["xtrack"] == 2048

        if self.is_postproc:
            longitude_varpath = "geolocation/longitude"
        elif self.is_l2:
            longitude_varpath = "Level1/Longitude"
        elif self.is_l1 or self.is_l2_met:
            longitude_varpath = "Geolocation/Longitude"

        if varpath is not None:
            var_dim_map = self.get_dim_map(varpath)
            atrack_axis = var_dim_map["atrack"]
            valid_xtrack = np.where(
                ~np.isnan(np.nanmedian(self.nc_dset[varpath][tuple()], axis=atrack_axis).squeeze())
            )[0]
        elif self.is_l2 or is_msat:
            var_dim_map = self.get_dim_map(longitude_varpath)
            atrack_axis = var_dim_map["atrack"]
            valid_xtrack = np.where(
                ~np.isnan(
                    np.nanmedian(
                        self.nc_dset[longitude_varpath][tuple()], axis=atrack_axis
                    ).squeeze()
                )
            )[0]
        elif self.is_l1:
            var_dim_map = self.get_dim_map("Band1/Radiance")
            spec_axis = var_dim_map["spectral_channel"]
            atrack_axis = var_dim_map["atrack"]
            xtrack_axis = var_dim_map["xtrack"]
            rad = self.nc_dset["Band1/Radiance"][tuple()]
            rad = rad.transpose(atrack_axis, xtrack_axis, spec_axis)
            valid_xtrack = np.where(np.nanmedian(np.nansum(rad, axis=2), axis=0) > 0)[0]
        if len(valid_xtrack) == 0:
            print(self.nc_dset.filepath(), " has no valid xtrack")
            valid_xtrack_slice = slice(None)
        else:
            valid_xtrack_slice = slice(valid_xtrack[0], valid_xtrack[-1] + 1)

        return valid_xtrack_slice

    def get_valid_rad(self) -> Union[slice, None]:
        """
        Get the valid radiance indices
        """
        if not (self.is_l2 or self.is_l1):
            return None
        if self.is_l2 and self.has_var("RTM_Band1/Radiance_I"):
            rad_var_path = "RTM_Band1/Radiance_I"
        elif self.is_l2:
            rad_var_path = None
        elif not self.is_l2:
            rad_var_path = "Band1/Radiance"

        if rad_var_path is None:
            valid_rad_slice = None
        else:
            var_dim_map = self.get_dim_map(rad_var_path)
            xtrack_axis = var_dim_map["xtrack"]
            atrack_axis = var_dim_map["atrack"]
            valid_rad = np.where(
                np.nansum(self.nc_dset[rad_var_path][tuple()], axis=(xtrack_axis, atrack_axis)) > 0
            )[0]

            if valid_rad.size != 0:
                valid_rad_slice = slice(valid_rad[0], valid_rad[-1] + 1)
            else:
                print(self.nc_dset.filepath(), " has no valid radiances")
                valid_rad_slice = None

        return valid_rad_slice

    def get_var_paths(
        self, start: Optional[Union[ncdf.Datasat, ncdf.Group]] = None, varpath_list: list[str] = []
    ) -> List[str]:
        """
        Get a list of all the full variable paths in the netcdf file

        start: starting Dataset or Group, defaults to self.nc_dset
        """
        if start is None:
            start = self.nc_dset

        for var in start.variables:
            varpath_list.append(_get_full_variable_path(start[var]))

        for grp in start.groups:
            self.get_var_paths(start[grp], varpath_list)

        self.varpath_list = varpath_list

        return varpath_list

    def has_var(self, var: str) -> bool:
        """
        Check if the netcdf file has the given variable
        """
        varpath_list = self.get_var_paths()
        return var in varpath_list


def _get_full_variable_path(var: ncdf.Variable) -> str:
    """
    Get the full variable path for a given variable

    var (Variable): netCDF4 variable object
    """
    parts = []
    group = var.group()
    while group.name != "/":
        parts.append(group.name)
        group = group.parent
    parts.reverse()
    parts.append(var.name)

    return "/".join(parts)
