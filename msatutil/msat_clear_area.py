import argparse
import os
from pathlib import Path

import pandas as pd

from msatutil.msat_targets import get_target_dict, gs_posixpath_to_str


def derive_L2_qc_path(l2pp_file_path: str) -> str:
    """
    Return the qaqc html path corresponding to a L2 post-processed file

    Input:
        l2pp_file_path (str): path to the L2 post-processed file
    Ouputs:
        html_file_path (str): path to the L2 qaqc file
    """
    l2pp_file_path = Path(l2pp_file_path)
    qc_file_path = (
        l2pp_file_path.parent
        / "qaqc"
        / l2pp_file_path.name.replace("_L2_", "_L2_QAQC_").replace(".nc", ".csv")
    )

    return gs_posixpath_to_str(qc_file_path)


def msat_clear_area(
    l2_file_list: str,
    l3_file_list: str,
    qc_dir: str,
    flag_fraction_threshold: float,
    out_file: str,
    download: bool,
):
    """
    Download L2 QAQC csv files.
    List the collections that have less than flag_fraction_threshold flagged data.
    Write a list of bucket paths for corresponding L3 collections, to be ingested by msat_targets
    """

    d = get_target_dict(l2_file_list, derive_L2_qc_path)
    dl3 = get_target_dict(l3_file_list)

    l3_collections = [c for t in dl3 for c in dl3[t]]

    if download:
        qc_files = [
            d[t][c][p] + "\n" for t in d for c in d[t] for p in d[t][c] if c in l3_collections
        ]
        with open("l2_qc_file_list.txt", "w") as f:
            f.writelines(qc_files)
        os.system(f"cat l2_qc_file_list.txt | gsutil -m cp -n -I {qc_dir}/")

    l3_clear_list = []
    for t in d:
        for c in d[t]:
            if c not in l3_collections:
                continue
            for p in d[t][c]:
                local_qc_path = Path(qc_dir) / Path(d[t][c][p]).name
                if not local_qc_path.exists():
                    continue
                qc = (
                    pd.read_csv(
                        local_qc_path,
                        names=["var", "status", "value"],
                        skiprows=1,
                    )[["var", "value"]]
                    .set_index("var")
                    .T
                )
                if float(qc["CH4 flagged fraction"].value) < flag_fraction_threshold:
                    l3_pid = list(dl3[t][c].keys())[0]
                    l3_clear_list += [dl3[t][c][l3_pid] + "\n"]

    with open(out_file, "w") as f:
        f.writelines(l3_clear_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("l2_file_list", help="full path to input file with list of L2 bucket paths")
    parser.add_argument("l3_file_list", help="full path to input file with list of L3 bucket paths")
    parser.add_argument(
        "qc_dir",
        help="full path to output directory where the L2 qaqc csv files will be downloaded",
    )
    parser.add_argument(
        "-f",
        "--flag-fraction-threshold",
        required=True,
        type=float,
        help="Only collections with less than this fraction of data flagged will be",
    )
    parser.add_argument(
        "-o",
        "--out-file",
        default="msat_l3_clear_file_list.txt",
        help="full path to output list of files",
    )
    parser.add_argument(
        "--download", action="store_true", help="if given, download the qc csv files"
    )
    args = parser.parse_args()

    msat_clear_area(
        args.l2_file_list,
        args.l3_file_list,
        args.qc_dir,
        args.flag_fraction_threshold,
        args.out_file,
        args.download,
    )


if __name__ == "__main__":
    main()
