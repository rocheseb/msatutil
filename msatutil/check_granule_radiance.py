import argparse
import os
import warnings
from typing import Optional

import numpy as np
import pylab as plt
from msat_nc import msat_nc


def check_granule_radiance(
    l1_infile: str, threshold: float = 0.04, save_path: Optional[str] = None
) -> None:
    """
    Compute and plot the standard deviation of normalized radiances in the granule at l1_infile
    Also plot the radiances of the spectra that fall below a given threshold and include the pixel index in the legend

    l1_infile: full path to MethaneSAT/AIR L1B file
    threshold: value of the standard deviation of normalized radiances below which spectra are considered "bad"
    save_path: full path to the output directory where figures will be saved
    """

    with warnings.catch_warnings(), msat_nc(l1_infile, use_dask=True) as l1:
        warnings.simplefilter("ignore")

        radiance = l1.get_var("Radiance", grp="Band1").compute()
        rad_dims = l1.get_dim_map("Band1/Radiance")
        atrack_axis = rad_dims["atrack"]
        xtrack_axis = rad_dims["xtrack"]
        spec_axis = rad_dims["spectral_channel"]

        rad_slice = [slice(None) for i in range(3)]
        rad_slice[xtrack_axis] = l1.get_valid_xtrack()
        rad_slice[atrack_axis] = slice(0, 101)
        radiance = radiance[tuple(rad_slice)]

        std_norm_rad = np.nanstd(radiance, axis=spec_axis, ddof=1) / np.nanmax(
            radiance, axis=spec_axis
        )

    bad_ids = np.where(std_norm_rad < threshold)

    flat_std_norm_rad = std_norm_rad.flatten()

    plt.plot(flat_std_norm_rad, marker="o", linewidth=0)
    plt.title("Standard deviation of normalized radiances")
    plt.xlabel("pixel index")
    if save_path is not None:
        save_name = os.path.join(
            save_path, os.path.basename(l1_infile).replace(".nc", "_std_norm_rad.png")
        )
        plt.gcf().savefig(save_name)
    else:
        plt.show()
    plt.clf()
    if len(bad_ids):
        for i, j in zip(bad_ids[0], bad_ids[1]):
            plt.plot(radiance[i, j], label=f"along_index: {i}; across_index: {j}")
        plt.legend()
        plt.xlabel("spectral index")
        plt.ylabel("Radiance")
        if save_path is not None:
            save_name = os.path.join(
                save_path,
                os.path.basename(l1_infile).replace(".nc", "_bad_spectra.png"),
            )
            plt.gcf().savefig(save_name)
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compute and plot the standard deviation of normalized radiances in the granule at l1_infile."
        " Also plot the radiances of the spectra that fall below a given threshold and include the pixel index in the legend"
    )
    parser.add_argument("l1_infile")
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.04,
        help="threshold for the identifying bad standard deviation of normalized radiances",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        default=None,
        help="full path to the DIRECTORY of the output figures, does not save figures by default",
    )
    args = parser.parse_args()

    check_granule_radiance(args.l1_infile, threshold=args.threshold, save_path=args.save_path)


if __name__ == "__main__":
    main()
