import argparse
import re
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Callable, Optional

import geopandas as gpd
import geoviews as gv
import holoviews as hv
import pandas as pd
from bokeh.embed import file_html
from bokeh.models import (
    BoxSelectTool,
    Button,
    Column,
    ColumnDataSource,
    CustomJS,
    DateRangeSlider,
    Div,
    GlyphRenderer,
    HoverTool,
    InlineStyleSheet,
    NumericInput,
    Row,
    Select,
    TabPanel,
    Tabs,
    TapTool,
    TextInput,
)
from bokeh.plotting import figure
from bokeh.resources import CDN
from geoviews.element import WMTS

from msatutil.msat_targets import GOOGLE_IMAGERY, extract_timestamp, gs_posixpath_to_str

gv.extension("bokeh")


class PONumber:
    """
    Object representing a MAIR/SAT processing ID (PID)
    adds a > operator to check if a PID is larger (newer) than another
    """

    def __init__(self, val: str):
        match = re.search(r"(?i)po[-_](\d+)([A-Za-z]?)", val)
        self.str = val
        self.value = int(match.group(1)) if match else None
        self.extra = ord(match.group(2).upper()) - ord("A") + 1 if match and match.group(2) else 0

    def __gt__(self, other):
        if isinstance(other, PONumber):
            if self.value == other.value:
                return self.extra > other.extra
            else:
                return self.value > other.value
        return NotImplemented

    def __eq__(self, other):
        if isinstance(other, PONumber):
            return self.value == other.value and self.extra == other.extra
        return NotImplemented

    def __str__(self):
        return self.str

    def __repr__(self):
        return self.__str__()


def derive_L3_html_path(
    l3_mosaic_file_path: PosixPath, html_bucket: str, campaign: str, flight: str, area: str
) -> str:

    fname = l3_mosaic_file_path.name.replace(".nc", ".html")

    html_path = Path(f"gs://{html_bucket}") / campaign / flight / area / "30m" / fname

    return gs_posixpath_to_str(html_path)


def get_target_dict(file_list: str, func: Callable = gs_posixpath_to_str, **kwargs) -> dict:
    with open(file_list, "r") as fin:
        file_list = [Path(i.strip()) for i in fin.readlines()]

    is_L2 = "level2" in str(file_list[0])
    ids_L2 = {
        "campaign": 2,
        "flight": 6,
        "pid": 7,
    }
    ids_L3 = ids_L2.copy()
    ids_L3["area"] = 12
    ids_L3_b = {
        "campaign": 2,
        "flight": 4,
        "pid": 5,
        "area": 11,
    }
    ids_L3_c = ids_L3_b.copy()
    ids_L3_c["area"] = 13

    d = {}
    for i in file_list:
        if is_L2:
            ids = ids_L2
        elif len(i.parts[3]) == 4:
            ids = ids_L3
        elif "mosaic/t" in str(i):
            ids = ids_L3_c
        else:
            ids = ids_L3_b
        params = {k: i.parts[v] for k, v in ids.items()}

        pid = PONumber(params["pid"])

        if pid.value is None:
            print(f"Could not parse pid for {i}")
            continue

        c = params["campaign"]
        f = params["flight"]
        p = params["pid"]

        if c not in d:
            d[c] = {}
        if f not in d[c]:
            d[c][f] = {}

        # only keep the highest pid
        # if a processing exists, overwrite with the higher pid
        if d[c][f] != {}:
            old_pid = PONumber(list(d[c][f].keys())[0])
            if old_pid > pid:
                continue
            elif pid > old_pid:
                del d[c][f][str(old_pid)]

        if is_L2:
            d[c][f] = {p: func(i, **kwargs)}
        else:
            if p not in d[c][f]:
                d[c][f][p] = {}
            if "html_bucket" in kwargs:
                for k in ["campaign", "flight", "area"]:
                    kwargs[k] = params[k]
            d[c][f][p][params["area"]] = func(i, **kwargs)

    return d
