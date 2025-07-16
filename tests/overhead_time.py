import os
import json
import numpy as np
from astropy.table import Table
from astropy.time import Time
from datetime import datetime
from pathlib import Path

def compute_overheads_from_jsons(plan_dir):
    """
    Read daily *_plan.json files and compute per-exposure overheads.

    Parameters:
        plan_dir : str or Path — directory containing *_plan.json files

    Returns:
        Astropy Table with columns ['day', 'start_time', 'overhead_s']
    """
    plan_dir = Path(plan_dir)
    json_files = sorted(plan_dir.glob("*_plan.json"))
    rows = []

    for day_idx, json_file in enumerate(json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if len(data) < 2:
            continue  # skip days with too few exposures

        # Sort exposures by start_utc
        data.sort(key=lambda x: x["start_utc"])
        times = [d["start_utc"] for d in data]
        t = Time(times, format="iso", scale="utc")
        t_sec = t.unix  # convert to seconds

        exptimes = [d.get("exptime_s", 30.0) for d in data]

        for i in range(len(t_sec) - 1):
            dt = t_sec[i + 1] - t_sec[i]
            overhead = dt - exptimes[i]
            rows.append((day_idx, times[i + 1], overhead))

    return Table(rows=rows, names=["day", "start_time", "overhead_s"])

overheads = compute_overheads_from_jsons("../plans")
overheads.write("./daily_exposure_overheads.fits", overwrite=True)
