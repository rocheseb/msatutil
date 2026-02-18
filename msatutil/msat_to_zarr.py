"""
Convert an MSAT L1 file to a Zarr store with:

  - Band1/Radiance        -> zarr["Band1/Radiance"]        (3D)
  - Band1/Wavelength      -> zarr["Band1/Wavelength"]      (3D)
  - Band1/RadianceFlag    -> zarr["Band1/RadianceFlag"]    (3D)
  - MeanRadiance          -> zarr["MeanRadiance"]          (2D, atrack x xtrack)

Compute the MeanRadiance:
  - mask RadianceFlag > 0
  - mask wavelengths outside the inferred wvl_range
  - then nanmean over the spectral dimension

Usage
-----
python msat_to_zarr.py input.nc output.zarr
"""

import argparse
import os
import shutil

import numpy as np
import warnings
import zarr

from msatutil.msat_interface import msat_collection
from netCDF4 import Dataset
from typing import Optional

warnings.simplefilter("ignore")

CHUNKS_3D = (50, 50, 2048)  # (atrack, xtrack, spectral)
CHUNKS_2D = (64, 256)  # (atrack, xtrack)


def infer_wvl_range_from_name(name: str):
    if "_CH4_" in name or "_CO2_" in name:
        return (1597.7, 1682.1)
    elif "_O2_" in name:
        return (1249.0, 1289.0)
    elif "_H2O_" in name:
        return (1290.0, 1295.0)
    else:
        return None


def msat_netcdf_to_zarr(input_file: str, output_dir: str = "", overwrite: bool = False):
    if "_L1B_" in input_file:
        var_names = [
            "Band1/Radiance",
            "Band1/Wavelength",
            "Band1/RadianceFlag",
        ]
    elif "_L2_" in input_file:
        var_names = [
            "Posteriori_RTM_Band1/ResidualRadiance",
            "Posteriori_RTM_Band1/Radiance_I",
            "Posteriori_RTM_Band1/Wavelength",
            "OptProp_Band1/RefWvl_BRDF_KernelAmplitude_isotr",  # prior albedo
        ]
    else:
        raise NotImplementedError("Zarr conversion only implemented for L1B and L2 files")

    zarr_name = os.path.basename(input_file).replace(".nc", ".zarr")
    output_store = os.path.join(output_dir, zarr_name)

    convert_netcdf_to_zarr(input_file, output_store, var_names, overwrite)


def check_inputs(input_file, output_store, overwrite: bool = False):
    if not os.path.exists(input_file):
        raise ValueError(f"ERROR: input file does not exist: {input_file}")
    if os.path.exists(output_store):
        if not overwrite:
            raise ValueError(
                f"ERROR: output store exists: {output_store} (use --overwrite to replace)",
            )
        else:
            print(f"Removing existing Zarr store: {output_store}")
            shutil.rmtree(output_store)


def create_zarr_variables(
    dset: Dataset,
    var_names: list[str],
    zarr_root,
    compressor,
    nrow: int,
    ncol: int,
    nspec: int,
):
    zarr_dict = {}

    for v in var_names:
        try:
            _ = dset[v]
        except (IndexError, KeyError):
            print(f"Variable {v} not in file")
            continue

        if len(dset[v].dimensions) == 2:
            chunks = CHUNKS_2D
            shape = (nrow, ncol)
        else:
            chunks = CHUNKS_3D
            shape = (nrow, ncol, nspec)

        zarr_dict[v] = zarr_root.create_array(
            v,
            shape=shape,
            chunks=chunks,
            dtype=dset[v].dtype,
            compressors=[compressor],
        )

    return zarr_dict


def compute_mean_radiance(
    wvl,
    rad,
    flag,
    wvl_range: Optional[tuple[float, float]] = None,
):
    mask = np.zeros_like(rad, dtype=bool)
    mask |= flag > 0
    if wvl_range is not None:
        wmin, wmax = wvl_range
        mask |= (wvl < wmin) | (wvl > wmax)

    # Apply mask
    rad_valid = rad.astype(np.float32)
    rad_valid[mask] = np.nan

    mean_rad = np.nanmean(rad_valid, axis=2)

    return mean_rad


def convert_netcdf_to_zarr(
    input_file: str,
    output_store: str,
    var_names: list[str],
    overwrite: bool = False,
):
    check_inputs(input_file, output_store, overwrite)

    zarr_root = zarr.open_group(output_store, mode="w")
    compressor = zarr.codecs.Blosc(cname="zstd", clevel=5, shuffle=1)

    with msat_collection([input_file], use_dask=False) as c:
        if not (c.is_l2 or c.is_l1):
            raise Exception("Expected a L1B or L2 file")
        dset = c.dsets[c.ids[0]]

        ncol = c.dim_size_map["xtrack"]
        nrow = c.dim_size_map["atrack"]
        nspec = c.dim_size_map["spectral_channel"]

        print(f"Dimensions: (atrack, xtrack, spectral) = ({nrow}, {ncol}, {nspec})")

        fname = c.file_names[0]
        wvl_range = infer_wvl_range_from_name(fname)

        zarr_dict = create_zarr_variables(
            dset=dset,
            var_names=var_names,
            zarr_root=zarr_root,
            compressor=compressor,
            nrow=nrow,
            ncol=ncol,
            nspec=nspec,
        )
        if c.is_l1:
            zarr_dict["MeanRadiance"] = zarr_root.create_array(
                "MeanRadiance",
                shape=(nrow, ncol),
                chunks=CHUNKS_2D,
                dtype=np.float32,
                compressors=[compressor],
            )

        atrack_chunk = 20
        print("Starting read & conversion...")
        ichunk = 0
        while ichunk < nrow:
            iend = min(ichunk + atrack_chunk, nrow)
            print(f"  Processing nrow {ichunk}:{iend} ...")

            s = slice(ichunk, iend)

            for v in zarr_dict:
                if v == "MeanRadiance":
                    continue
                dim_map = c.get_dim_map(v)
                slices = [slice(None) for i in dim_map]
                slices[dim_map["atrack"]] = s
                var_chunk = dset[v][tuple(slices)]
                if isinstance(var_chunk, np.ma.MaskedArray):
                    fill_value = 0 if "RadianceFlag" in v else np.nan
                    var_chunk = var_chunk.filled(fill_value)
                if "spectral_channel" in dim_map and dim_map["spectral_channel"] == 0:
                    var_chunk = var_chunk.transpose(1, 2, 0)
                zarr_dict[v][s, ...] = var_chunk

            if c.is_l1:
                mean_rad_chunk = compute_mean_radiance(
                    wvl=zarr_dict["Band1/Wavelength"][s, ...],
                    rad=zarr_dict["Band1/Radiance"][s, ...],
                    flag=zarr_dict["Band1/RadianceFlag"][s, ...],
                    wvl_range=wvl_range,
                )

                # Store mean image into Zarr
                zarr_dict["MeanRadiance"][s, :] = mean_rad_chunk.astype(np.float32)

            ichunk = iend

        zarr_root.attrs["source_file"] = os.path.abspath(input_file)
        zarr_root.attrs["nrow"] = nrow
        zarr_root.attrs["ncol"] = ncol
        zarr_root.attrs["spectral_channels"] = nspec
        zarr_root.attrs["wavelength_range"] = wvl_range


def main():
    parser = argparse.ArgumentParser(description="Convert a MSAT L1B netCDF file to a Zarr store.")
    parser.add_argument("input_file", help="Path to input MSAT netcdf file")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Path to output directory under which the zarr data will be saved, if not given, save in current directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Zarr store if it exists",
    )
    args = parser.parse_args()

    msat_netcdf_to_zarr(args.input_file, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
