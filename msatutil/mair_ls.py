import argparse
from typing import Optional
from msatutil.msat_dset import cloud_file
import pandas as pd
import re


def mair_ls(
    in_path: str,
    flight_name: Optional[str] = None,
    uri: Optional[str] = None,
    aggregation: Optional[str] = None,
    resolution: Optional[str] = None,
    production_operation: Optional[str] = None,
    production_environment: Optional[str] = None,
    flight_date: Optional[str] = None,
    timestamp: Optional[str] = None,
    target_name: Optional[str] = None,
    time_start: Optional[str] = None,
    time_end: Optional[str] = None,
    molecule: Optional[str] = None,
    latest: bool = False,
    show: bool = True,
):
    with cloud_file(in_path) as csv_in:
        df = pd.read_csv(csv_in)

    for k in ["production_timestamp", "time_start", "time_end", "flight_date"]:
        if (k is not None) and (k in df.columns):
            df[k] = pd.to_datetime(df[k])

    # string equality checks
    for k, v in {
        "flight_name": flight_name,
        "aggregation": aggregation,
        "level3_resolution": resolution,
        "production_operation": production_operation,
        "production_environment": production_environment,
        "molecule": molecule,
    }.items():
        if (v is not None) and (k in df.columns):
            df = df.loc[df[k].str.lower() == v.lower()]

    # string contains checks
    for k, v in {"level3_target_name": target_name, "uri": uri}.items():
        if (v is not None) and (k in df.columns):
            df = df.loc[df[k].str.contains(v, na=False, case=False)]

    # dates and timestamp filters
    if (flight_date is not None) and ("flight_date" in df.columns):
        df = df.loc[df["flight_date"] == pd.to_datetime(flight_date)]

    if latest and "production_timestamp" in df.columns:
        sorted_production_operation = list(
            df.sort_values(by="production_timestamp")
            .groupby("production_operation")
            .first()
            .sort_values(by="production_timestamp")
            .reset_index()["production_operation"]
        )
        production_operation = sorted_production_operation[-1]
        df = df.loc[df["production_operation"] == production_operation]

    if timestamp is not None:
        timestamp = pd.to_datetime(timestamp)

    if timestamp is not None and "production_timestamp" in df.columns:
        df = df.loc[df["production_timestamp"] == timestamp]

    time_filter_pattern = r"([<>]=?|[=!]=)\s*(.*)"
    for k, v in {"time_start": time_start, "time_end": time_end}.items():
        if (v is not None) and (k in df.columns):
            try:
                operator, timestamp = re.match(time_filter_pattern, v).groups()
                timestamp = pd.to_datetime(timestamp)
            except (AttributeError, pd.errors.ParserError):
                print(f"Could not parse {v} for {k}")
                continue
            operator_mapping = {
                "!=": lambda x: x != timestamp,
                "==": lambda x: x == timestamp,
                ">=": lambda x: x >= timestamp,
                "<=": lambda x: x <= timestamp,
                ">": lambda x: x > timestamp,
                "<": lambda x: x < timestamp,
            }
            if operator in operator_mapping:
                df = df.loc[operator_mapping[operator](df[k])]

    if show:
        for i in df.uri.tolist():
            print(i)

    return df.reset_index().drop(columns=["index"])


def create_parser(**kwargs):
    parser = argparse.ArgumentParser(**kwargs)
    parser.add_argument(
        "-f",
        "--flight-name",
        type=str,
        default=None,
        help="flight name e.g. (RF06, MX011)",
    )
    parser.add_argument(
        "--uri",
        type=str,
        default=None,
        help="Pattern to look for in the google storage path",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default=None,
        help="level3 aggregation, e.g. 5x1 or 1x1",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=str,
        default=None,
        help="level3 resolution e.g. 10m or 30m",
    )
    parser.add_argument(
        "--production-operation",
        type=str,
        default=None,
        help="Jira production ticket number e.g. PO-444a",
    )
    parser.add_argument(
        "--flight-date",
        type=str,
        default=None,
        help="flight date as YYYYMMDD or any pandas parsable date formats",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="production timestamp as YYYY-MM-DD HH:MM:SS or any pandas parsable date formats",
    )
    parser.add_argument(
        "--time-start",
        type=str,
        default=None,
        help="operation on start times in the data files e.g. >= 2022-01-02 00:00:00",
    )
    parser.add_argument(
        "--time-end",
        type=str,
        default=None,
        help="operation on end times in the data files e.g. >= 2022-01-02 00:00:00",
    )
    parser.add_argument(
        "-n",
        "--target-name",
        type=str,
        default=None,
        help="Pattern to look for in the target name",
    )
    parser.add_argument(
        "--production-environment",
        type=str,
        default=None,
        help="production environment e.g. prod or stag or dev",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="if given, only show matches with the latest timestamp",
    )
    parser.add_argument(
        "-m",
        "--molecule",
        type=str,
        default=None,
        help="L2_granret only, molecule name: CO2, H2O, or O2",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="if given, print the matching list of files",
    )
    return parser


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("in_path", help="Full path to the input csv table")
    parser = create_parser(
        parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()

    mair_ls(**vars(args))


if __name__ == "__main__":
    main()
