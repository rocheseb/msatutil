import argparse
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
from typing import Optional, Union
import subprocess


def parse_dark_file_path_star(args):
    """
    For use with multiprocessing
    """
    return parse_dark_file_path(*args)


def parse_l0_file_path_star(args):
    """
    For use with multiprocessing
    """
    return parse_l0_file_path(*args)


def parse_l1_file_path_star(args):
    """
    For use with multiprocessing
    """
    return parse_l1_file_path(*args)


def parse_l2_or_l3_file_path_star(args):
    """
    For use with multiprocessing
    """
    return parse_l2_or_l3_file_path(*args)


def split_str(s: str, i: int, sep: str = "_") -> Optional[str]:
    """
    Inputs:
        s (str): input string to split
        i (int): index of element to return from the split list
        sep (str): separation string for the splitting
    Outputs:
        x (Union[str,None]):
    """
    try:
        x = s.split(sep)[i]
    except IndexError:
        pass
    else:
        return x


def parse_dark_file_path(path: str, timestamp: str) -> Optional[dict[str, str]]:
    path = path.strip()
    if "dark_file" not in path or "PreRadFlag" in path or "MSI" in path:
        return None
    gs_path = path.startswith("gs://")
    if gs_path:
        path = path[5:]
    spath = path.split("/")[2:]
    columns = ["campaign", "year", "month", "day", "flight_name", "file_name"]

    d = {k: [None] for k in columns}
    for i, v in enumerate(spath):
        try:
            d[columns[i]] = v
        except (IndexError, KeyError):
            continue

    if gs_path:
        d["uri"] = f"gs://{path}"
    else:
        d["uri"] = path

    d["molecule"] = "CH4" if "CH4" in os.path.basename(path) else "O2"

    return d


def parse_l0_file_path(path: str, timestamp: str) -> Optional[dict[str, str]]:
    path = path.strip()
    if (
        "test" in path
        or ".orig" in path
        or ".sav" in path
        or ("ch4_camera" not in path and "o2_camera" not in path)
    ):
        return None
    gs_path = path.startswith("gs://")
    if gs_path:
        path = path[5:]
    spath = path.split("/")[1:]
    columns = ["flight_date", "molecule", "uri"]

    d = {k: [None] for k in columns}
    for i, v in enumerate(spath):
        try:
            d[columns[i]] = v
        except (IndexError, KeyError):
            continue

    if gs_path:
        d["uri"] = f"gs://{path}"
    else:
        d["uri"] = path

    return d


def parse_l1_file_path(path: str, timestamp: str) -> Optional[dict[str, str]]:
    path = path.strip()

    if not path.endswith(".nc") or "level1a" in path or "moved_files" in path:
        return None

    gs_path = path.startswith("gs://")
    if gs_path:
        path = path[5:]
    spath = path.split("/")[1:]

    columns = [
        "campaign",
        "year",
        "month",
        "day",
        "flight_name",
        "production_operation",
        "collection",
        "aggregation",
        "molecule",
        "file_name",
    ]

    d = {k: [None] for k in columns}
    for i, v in enumerate(spath):
        try:
            d[columns[i]] = v
        except (IndexError, KeyError):
            continue

    if gs_path:
        d["uri"] = f"gs://{path}"
    else:
        d["uri"] = path
    d["production_timestamp"] = timestamp
    d["time_start"] = split_str(d["file_name"], 3)
    d["time_end"] = split_str(d["file_name"], 4)

    return d


def parse_l2_or_l3_file_path(path: str, timestamp: str) -> Optional[dict[str, str]]:
    """
    Used to parse both L2 and L3 file paths
    """
    is_msat = "data-methanesat" in path
    path = path.strip()
    gs_path = path.startswith("gs://")
    if gs_path:
        path = path[5:]
    spath = path.split("/")[1:]

    if (
        (not path.endswith(".nc"))
        or ("subgranule-retrievals" in path)
        or ("qaqc" in path)
        or ("bias-model" in path)
        or ("moved_files" in path)
        or (is_msat and not spath[0].isdigit())
    ):
        return None

    columns = [
        "campaign",
        "year",
        "month",
        "day",
        "flight_name",
        "production_operation",
        "collection",
        "aggregation",
    ]
    if is_msat:
        columns = columns[:-2]

    if "granule-retrievals" in path:
        columns += ["type", "molecule"]
    elif "level2-results" in path:
        columns += ["type"] if is_msat else ["type", "post_product"]
    elif "interpolated_granule_retrievals" in path:
        columns += ["type", "molecule"] if is_msat else ["type", "post_product", "molecule"]
    elif "level2-apriori" in path:
        columns += ["type"]
    elif "level1b" in path:
        columns += ["molecule"]
    elif "mosaic" in path or "segment" in path:
        columns += ["type", "run", "level3_target_name", "resolution"]
    elif "regrid" in path:
        columns += ["type", "resolution"]
    elif "segment" in path:
        columns += ["type"]
    columns += ["file_name"]

    d = {k: [None] for k in columns}
    for i, v in enumerate(spath):
        try:
            d[columns[i]] = v
        except (IndexError, KeyError):
            continue

    if gs_path:
        d["uri"] = f"gs://{path}"
    else:
        d["uri"] = path
    d["production_timestamp"] = timestamp
    try:
        d["time_start"] = split_str(d["file_name"], 3)
    except Exception:
        return None
    d["time_end"] = split_str(d["file_name"], 4)

    if "level2-apriori" in path:
        d["molecule"] = split_str(d["file_name"], 2)
        if d["molecule"] == "CH4":
            d["molecule"] == "CO2"

    return d


def mair_gs_db(
    file_list: list[str],
    timestamp_list: list[str],
    outfile: str,
    ncores: int = int(len(os.sched_getaffinity(0)) / 2),
    level: int = 2,
) -> None:
    """
    file_list (list[str]): list of MethaneSAT bucket files
    timestamp_list (list[str]): timestamps corresponding to files in file_list
    outfile (str): the output csv file
    ncores (int): number of cores for parallel processing
    level (int): data level from -1 to 3, -1 is for darks
    """

    args = tuple((file_path, timestamp) for file_path, timestamp in zip(file_list, timestamp_list))
    func_dict = {
        -1: parse_dark_file_path_star,
        0: parse_l0_file_path_star,
        1: parse_l1_file_path_star,
        2: parse_l2_or_l3_file_path_star,
        3: parse_l2_or_l3_file_path_star,
    }
    with multiprocessing.Pool(ncores) as pp:
        file_metadata = list(tqdm(pp.imap_unordered(func_dict[level], args), total=len(file_list)))

    if level in [2, 3]:
        columns = [
            "campaign",
            "year",
            "month",
            "day",
            "flight_name",
            "production_operation",
            "collection",
            "aggregation",
            "type",
            "molecule",
            "post_product",
            "run",
            "target",
            "resolution",
            "file_name",
            "uri",
            "production_timestamp",
            "time_start",
            "time_end",
        ]
    elif level == 1:
        columns = [
            "campaign",
            "year",
            "month",
            "day",
            "flight_name",
            "production_operation",
            "collection",
            "aggregation",
            "molecule",
            "file_name",
            "uri",
            "production_timestamp",
            "time_start",
            "time_end",
        ]
    elif level == 0:
        columns = ["flight_date", "molecule", "uri"]
    elif level == -1:
        columns = [
            "campaign",
            "year",
            "month",
            "day",
            "flight_name",
            "file_name",
            "molecule",
            "uri",
        ]

    d = {k: [None] * len(file_metadata) for k in columns}

    for i, row in enumerate(file_metadata):
        if row is None:
            continue
        for k in columns:
            if k in row:
                d[k][i] = row[k]

    df = pd.DataFrame.from_dict(d)

    df = df.dropna(how="all")

    if level != 0:
        df["flight_date"] = df[["year", "month", "day"]].agg("-".join, axis=1)
        drop_list = ["index", "year", "month", "day", "file_name"]
        if level in [2, 3]:
            drop_list += ["post_product"]

        df = df.reset_index().drop(columns=drop_list)

    df.to_csv(outfile, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Create a csv file with parsed MethaneAIR gs bucket file paths that can be used as input for mair_ls"
    )
    parser.add_argument(
        "gs_bucket_name",
        help="gs bucket name where the file paths will be parse (don't include gs://)",
    )
    parser.add_argument("outfile", help="full path to output csv file")
    parser.add_argument(
        "--level",
        default=2,
        choices=[-1, 0, 1, 2, 3],
        type=int,
        help="data level, -1 is for darks",
    )
    parser.add_argument(
        "--out-gs",
        default=None,
        help="full gs path where the output csv file will be copied (include gs:// and file name)",
    )
    args = parser.parse_args()

    if "data-methanesat" in args.gs_bucket_name:
        gs_command = f"gsutil -m ls -l gs://{args.gs_bucket_name}/**"
    else:
        gs_command = f"gsutil -m ls -l gs://{args.gs_bucket_name}/MAIR*/**"
    grep_command = 'grep "\.nc$"'
    gs_data_file = f"{os.path.join(os.path.dirname(args.outfile),args.gs_bucket_name)}.txt"

    command = f"{gs_command} | {grep_command} > {gs_data_file}"
    print(f"Now running following command, may take several minutes to complete:\n {command}")
    subprocess.run(command, shell=True)

    with open(gs_data_file, "r") as f:
        c = [i.strip().split() for i in f.readlines()]
    file_list = [i[-1] for i in c]
    timestamp_list = [i[1] for i in c]

    mair_gs_db(file_list, timestamp_list, args.outfile, level=args.level)

    if args.out_gs is not None:
        subprocess.run(f"gsutil cp {args.outfile} {args.out_gs}", shell=True)


if __name__ == "__main__":
    main()
