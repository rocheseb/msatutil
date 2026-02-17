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
import sys

import numpy as np
import warnings
import zarr

from msatutil.msat_interface import msat_collection

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


def convert_to_zarr(input_path: str, output_store: str, overwrite: bool = False):
    if "_L1B_" in input_path:
        convert_L1B_to_zarr(input_path, output_store, overwrite)
    elif "_L2_" in input_path:
        convert_L2_to_zarr(input_path, output_store, overwrite)
    else:
        raise NotImplementedError("Zarr conversion only implemented for L1B and L2 files")


def check_inputs(input_path, output_store, overwrite: bool = False):
    if not os.path.exists(input_path):
        raise ValueError(f"ERROR: input file does not exist: {input_path}")
    if os.path.exists(output_store):
        if not overwrite:
            raise ValueError(
                f"ERROR: output store exists: {output_store} (use --overwrite to replace)",
            )
        else:
            print(f"Removing existing Zarr store: {output_store}")
            shutil.rmtree(output_store)


def convert_L2_to_zarr(input_path: str, output_store: str, overwrite: bool = False):

    check_inputs(input_path, output_store, overwrite)

    zroot = zarr.open_group(output_store, mode="w")
    compressor = zarr.codecs.Blosc(cname="zstd", clevel=5, shuffle=1)

    var_names = [
        "Posteriori_RTM_Band1/ResidualRadiance",
        "Posteriori_RTM_Band1/Wavelength",
        "Posteriori_RTM_Band1/Radiance_I",
        "OptProp_Band1/RefWvl_BRDF_KernelAmplitude_isotr",  # prior albedo
    ]

    with msat_collection([input_path], use_dask=False) as c:
        if not c.is_L2:
            raise Exception("Expected a L2 file")
        dset = c.dsets[c.ids[0]]

        cols = c.dim_size_map["xtrack"]
        rows = c.dim_size_map["atrack"]
        nspec = c.dim_size_map["spectral_channel"]

        print(f"Dimensions: (atrack, xtrack, spectral) = ({rows}, {cols}, {nspec})")

        fname = c.file_names[0]
        wvl_range = infer_wvl_range_from_name(fname)

        zarr_dict = {}

        for v in var_names:
            try:
                _ = dset[v]
            except (IndexError, KeyError):
                print(f"Variable {v} not in file")
                continue

            if len(dset[v].dimensions) == 2:
                chunks = CHUNKS_2D
                shape = (rows, cols, nspec)
            else:
                chunks = CHUNKS_3D
                shape = (rows, cols)

            zarr_dict[v] = zroot.create_array(
                v,
                shape=shape,
                chunks=chunks,
                dtype=dset[v].dtype,
                compressors=[compressor],
            )

        atrack_chunk = 20
        print("Starting read & conversion...")
        ichunk = 0
        while ichunk < rows:
            iend = min(ichunk + atrack_chunk, rows)
            print(f"  Processing rows {ichunk}:{iend} ...")

            s = slice(ichunk, iend)

            for v in var_names:
                var_chunked = dset[v][s]
                if isinstance(var_chunked, np.ma.MaskedArray):
                    var_chunked = var_chunked.filled(np.nan)
                zarr_dict[v][s, :] = var_chunked

            ichunk = iend

        zroot.attrs["source_file"] = os.path.abspath(input_path)
        zroot.attrs["rows"] = rows
        zroot.attrs["cols"] = cols
        zroot.attrs["spectral_channels"] = nspec
        zroot.attrs["wavelength_range"] = wvl_range


def convert_L1B_to_zarr(input_path: str, output_store: str, overwrite: bool = False):

    check_inputs(input_path, output_store, overwrite)

    with msat_collection([input_path], use_dask=False) as c:
        if not c.is_l1:
            raise Exception("Expected a L1B file")

        cols = c.dim_size_map["xtrack"]
        rows = c.dim_size_map["atrack"]
        nspec = c.dim_size_map["spectral_channel"]

        print(f"Dimensions: (atrack, xtrack, spectral) = ({rows}, {cols}, {nspec})")

        fname = c.file_names[0]
        wvl_range = infer_wvl_range_from_name(fname)

        # Create Zarr group and datasets
        zroot = zarr.open_group(output_store, mode="w")
        compressor = zarr.codecs.Blosc(cname="zstd", clevel=5, shuffle=1)

        rad_z = zroot.create_array(
            "Band1/Radiance",
            shape=(rows, cols, nspec),
            chunks=CHUNKS_3D,
            dtype=c.dsets[c.ids[0]]["Band1/Radiance"].dtype,
            compressors=[compressor],
        )
        wvl_z = zroot.create_array(
            "Band1/Wavelength",
            shape=(rows, cols, nspec),
            chunks=CHUNKS_3D,
            dtype=c.dsets[c.ids[0]]["Band1/Wavelength"].dtype,
            compressors=[compressor],
        )
        flag_z = zroot.create_array(
            "Band1/RadianceFlag",
            shape=(rows, cols, nspec),
            chunks=CHUNKS_3D,
            dtype=c.dsets[c.ids[0]]["Band1/RadianceFlag"].dtype,
            compressors=[compressor],
        )

        # 2D mean radiance (atrack, xtrack)
        mean_z = zroot.create_array(
            "MeanRadiance",
            shape=(rows, cols),
            chunks=CHUNKS_2D,
            dtype=np.float32,
            compressors=[compressor],
        )

        rad_var = c.dsets[c.ids[0]]["Band1/Radiance"]
        wvl_var = c.dsets[c.ids[0]]["Band1/Wavelength"]
        flag_var = c.dsets[c.ids[0]]["Band1/RadianceFlag"]

        atrack_chunk = 20
        print("Starting read & conversion...")
        ichunk = 0
        while ichunk < rows:
            iend = min(ichunk + atrack_chunk, rows)
            print(f"  Processing rows {ichunk}:{iend} ...")

            s = slice(ichunk, iend)

            rad_chunk = rad_var[s, :, :]
            wvl_chunk = wvl_var[s, :, :]
            flag_chunk = flag_var[s, :, :]

            if isinstance(rad_chunk, np.ma.MaskedArray):
                rad_chunk = rad_chunk.filled(np.nan)
            if isinstance(wvl_chunk, np.ma.MaskedArray):
                wvl_chunk = wvl_chunk.filled(np.nan)
            if isinstance(flag_chunk, np.ma.MaskedArray):
                flag_chunk = flag_chunk.filled(0)

            # Write to Zarr
            rad_z[s, :, :] = rad_chunk
            wvl_z[s, :, :] = wvl_chunk
            flag_z[s, :, :] = flag_chunk

            mask = np.zeros_like(rad_chunk, dtype=bool)
            mask |= flag_chunk > 0
            if wvl_range is not None:
                wmin, wmax = wvl_range
                mask |= (wvl_chunk < wmin) | (wvl_chunk > wmax)

            # Apply mask
            rad_valid = rad_chunk.astype(np.float32)
            rad_valid[mask] = np.nan

            mean_rad_chunk = np.nanmean(rad_valid, axis=2)

            # Store mean image into Zarr
            mean_z[s, :] = mean_rad_chunk.astype(np.float32)

            ichunk = iend

        zroot.attrs["source_file"] = os.path.abspath(input_path)
        zroot.attrs["rows"] = rows
        zroot.attrs["cols"] = cols
        zroot.attrs["spectral_channels"] = nspec
        zroot.attrs["wavelength_range"] = wvl_range

    print(f"Done. Zarr store written to: {output_store}")


def main():
    parser = argparse.ArgumentParser(description="Convert a MSAT L1B netCDF file to a Zarr store.")
    parser.add_argument("input", help="Path to input MSAT file")
    parser.add_argument("output", help="Path to output Zarr directory (e.g., file.zarr)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Zarr store if it exists",
    )
    args = parser.parse_args()

    convert_to_zarr(args.input, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
