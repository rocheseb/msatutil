import argparse
import re
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from msatutil.msat_dset import cloud_file


def load_dataframe(in_path: Union[str, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(in_path, str):
        with cloud_file(in_path) as csv_in:
            df = pd.read_csv(csv_in)  # load from path
    elif isinstance(in_path, pd.DataFrame):
        df = in_path  # use pre-loaded catalogue for efficiency
    else:
        raise TypeError("in_path must be either a string representing a path or a pandas DataFrame")

    return df


def convert_to_datetime(df: pd.DataFrame, col: str):
    """
    Modifies df in place by converting df[col] to pandas timestamps

    Inputs:
        df (pd.DataFrame): input dataframe
        col (str): column name to convert to datetime
    """

    try:
        transformed = pd.to_datetime(df[col], errors="coerce")
    except (ValueError, TypeError):
        return

    if set(transformed) != set([pd.NaT]):
        df[col] = transformed


def get_latest(df: pd.DataFrame) -> pd.DataFrame:

    df_out = df.copy()

    is_msat = len(df.iloc[0]["flight_name"]) == 8

    if not is_msat and "production_timestamp" in df.columns:
        sorted_production_operation = list(
            df.sort_values(by="production_timestamp")
            .groupby("production_operation")
            .first()
            .sort_values(by="production_timestamp")
            .reset_index()["production_operation"]
        )
        production_operation = sorted_production_operation[-1]
        df_out = df.loc[df["production_operation"] == production_operation]
    elif is_msat and "production_timestamp" in df.columns:
        df_out = (
            df.sort_values(by="production_timestamp").groupby("flight_name").last().reset_index()
        )

    return df_out


def mair_ls(
    in_path: Union[str, pd.DataFrame],
    target: Optional[str] = None,
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
    type: Optional[str] = None,
    latest: bool = False,
    show: bool = True,
):
    df = load_dataframe(in_path)

    for k in ["production_timestamp", "time_start", "time_end", "flight_date"]:
        if (k is not None) and (k in df.columns):
            convert_to_datetime(df, k)

    # string equality checks
    for k, v in {
        "target": target,
        "flight_name": flight_name,
        "aggregation": aggregation,
        "resolution": resolution,
        "production_operation": production_operation,
        "production_environment": production_environment,
        "molecule": molecule,
        "type": type,
    }.items():
        if (v is not None) and (k in df.columns):
            df = df.loc[df[k].astype(str).str.lower() == v.lower()]

    # string contains checks
    for k, v in {"target": target_name, "uri": uri}.items():
        if (v is not None) and (k in df.columns):
            df = df.loc[df[k].astype(str).str.contains(v, na=False, case=False)]

    # dates and timestamp filters
    if (flight_date is not None) and ("flight_date" in df.columns):
        df = df.loc[df["flight_date"] == pd.to_datetime(flight_date)]

    if df.index.size == 0:
        return df.reset_index().drop(columns=["index"])

    if latest:
        df = get_latest(df)

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


def mair_ls_serial(catalogue, latest=False, **kwargs) -> pd.DataFrame:
    """
    Calls mair_ls in a loop over flight names, if latest==True. Otherwise, simply returns the unfiltered catalogue. Useful for specifying
    latest=True, when the latest version *for each flight* is desired.

    kwargs are passed to mair_ls. Don't specifiy `flight_name` or `show` in kwargs

    Parameters
    ----------
    catalogue : str or pd.DataFrame
        _description_
    latest : bool, optional
        Passes `latest=True` to mair_ls, by default False

    Returns
    -------
    pd.DataFrame
        Output of mair_ls

    Raises
    ------
    TypeError
        if catalogue is not str or pd.DataFrame
    """
    if isinstance(catalogue, str):
        catalogue_pth = catalogue
        catalogue = mair_ls(catalogue_pth, show=False, **kwargs)  # load from path
    elif isinstance(catalogue, pd.DataFrame):
        pass  # use pre-loaded catalogue
    else:
        raise TypeError(
            "catalogue must be either a string representing a path or a pandas DataFrame"
        )
    if latest == False:
        return catalogue
    flight_name_key = "flight_name"
    unq_flights = np.unique(catalogue[flight_name_key])
    latest_segments_concat = []  # init
    for flight in unq_flights:
        latest_segments = mair_ls(catalogue, flight_name=flight, show=False, latest=True, **kwargs)
        latest_segments_concat.append(latest_segments)
    catalogue_filtered = pd.concat(latest_segments_concat, ignore_index=True)
    assert len(catalogue_filtered) <= len(
        catalogue
    ), "The catalogue appears not to be filtered. Did you specify any keyword arguments?"
    return catalogue_filtered


def create_parser(**kwargs):
    parser = argparse.ArgumentParser(**kwargs)
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default=None,
        help="MSAT target number",
    )
    parser.add_argument(
        "-f",
        "--flight-name",
        type=str,
        default=None,
        help="flight name e.g. RF06, MX011 for MAIR or unique 8 character collection ID for MSAT",
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
        help="Pattern to look for in the MAIR level3 target name",
    )
    parser.add_argument(
        "--production-environment",
        type=str,
        default=None,
        help="MAIR production environment e.g. prod or stag or dev",
    )
    parser.add_argument(
        "--type",
        type=str,
        default=None,
        help="product type e.g. mosaic or regrid",
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
        help="molecule name: CH4, H2O, or O2",
    )

    return parser


def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("in_path", help="Full path to the input csv table")
    parser = create_parser(
        parents=[parent_parser], formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()
    args.show = True

    mair_ls(**vars(args))


if __name__ == "__main__":
    main()
