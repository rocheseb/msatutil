import argparse
import os
import pandas as pd
import multiprocessing
from tqdm import tqdm
from typing import Optional
import subprocess


def parse_file_path_star(args):
    """
    For use with multiprocessing
    """
    return parse_file_path(*args)


def split_str(s: str, i: int, sep: str = "_") -> str:
    try:
        x = s.split(sep)[i]
    except IndexError:
        x = None
    return x


def parse_file_path(path: str, timestamp: str) -> Optional[dict[str, str]]:
    path = path.strip()
    if (not path.endswith(".nc")) or any(
        [i in path for i in ["subgranule-retrievals", "qaqc", "bias-model", "tmp"]]
    ):
        return None

    gs_path = path.startswith("gs://")
    if gs_path:
        path = path[5:]
    spath = path.split("/")[1:]

    columns = [
        "target",
        "year",
        "month",
        "day",
        "flight_name",
        "production_operation",
    ]
    if "manual" in path:
        columns = ["manual"] + columns

    if "granule-retrievals" in path:
        columns += ["type", "molecule"]
    elif any([i in path for i in ["level2-results", "level2-apriori"]]):
        columns += ["type"]
    elif "qmask" in path:
        columns += ["aggregation", "type"]
    elif "regrid" in path:
        columns += ["aggregation", "type", "resolution"]
    columns += ["file_name"]

    if len(spath) != len(columns):
        return None

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
    d["target"] = d["target"].strip("t")
    d["production_timestamp"] = timestamp
    d["time_start"] = split_str(d["file_name"], 3)
    d["time_end"] = split_str(d["file_name"], 4)

    if "level2-apriori" in path or "_L1B_" in path:
        d["molecule"] = split_str(d["file_name"], 2)
        if d["molecule"] == "CO2":
            d["molecule"] = "CH4"

    return d


def msat_gs_db(
    file_list: list[str],
    timestamp_list: list[str],
    outfile: str,
    ncores: int = int(len(os.sched_getaffinity(0)) / 2),
) -> None:
    """
    file_list (List[str]): list of MethaneSAT bucket files
    """

    args = tuple((file_path, timestamp) for file_path, timestamp in zip(file_list, timestamp_list))
    with multiprocessing.Pool(ncores) as pp:
        file_metadata = list(
            tqdm(pp.imap_unordered(parse_file_path_star, args), total=len(file_list))
        )

    columns = [
        "target",
        "year",
        "month",
        "day",
        "flight_name",
        "production_operation",
        "aggregation",
        "type",
        "molecule",
        "resolution",
        "file_name",
        "uri",
        "production_timestamp",
        "time_start",
        "time_end",
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

    df["flight_date"] = df[["year", "month", "day"]].agg("-".join, axis=1)

    drop_list = ["index", "year", "month", "day", "file_name"]
    for c in df.columns:
        if df[c].isna().all():
            drop_list += [c]

    df = df.reset_index().drop(columns=drop_list)

    df.to_csv(outfile, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Create a csv file with parsed MethaneSAT gs bucket file paths that can be used as input for mair_ls"
    )
    parser.add_argument(
        "gs_bucket_name",
        help="gs bucket name where the file paths will be parse (don't include gs://)",
    )
    parser.add_argument("outfile", help="full path to output csv file")
    parser.add_argument(
        "--out-gs",
        default=None,
        help="full gs path where the output csv file will be copied (include gs:// and file name)",
    )
    parser.add_argument(
        "-i",
        "--infile",
        default=None,
        help="input text file listing google storage paths, if not given it will be created",
    )
    args = parser.parse_args()

    if args.infile is None:
        gs_command = f"gsutil -m ls -l gs://{args.gs_bucket_name}/**/*.nc"
        gs_data_file = f"{os.path.join(os.path.dirname(args.outfile),args.gs_bucket_name)}.txt"

        command = f"{gs_command} > {gs_data_file}"
        print(f"Now running following command, may take several minutes to complete:\n {command}")
        subprocess.run(command, shell=True)
    else:
        gs_data_file = args.infile

    with open(gs_data_file, "r") as f:
        c = [i.strip().split() for i in f.readlines()]
    file_list = [i[-1] for i in c]
    timestamp_list = [i[1] for i in c]

    msat_gs_db(file_list, timestamp_list, args.outfile)

    if args.out_gs is not None:
        subprocess.run(f"gsutil cp {args.outfile} {args.out_gs}", shell=True)


if __name__ == "__main__":
    main()
