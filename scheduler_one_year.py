# -*- coding: utf-8 -*-
"""
survey_scheduler.py — Adaptive 45-Field Clustering Scheduler
=============================================================

This module provides a real-time, dynamically adaptive scheduler for astronomical surveys. 
It clusters fields into 45-field observation groups and revisits each group three times 
during a night. The selection process accounts for visibility constraints, moon/planet 
proximity, field altitude, slew overheads, and exposure history.

Key Features
------------
- Clustering:
    - At each decision step, selects the field with the highest weight.
    - Builds a cluster with its 44 nearest eligible neighbors (by great-circle distance).

- Repetition:
    - Each cluster is observed three times per night (with revisit spacing).
    - Sequence: center → neighbors → back to center (same for each pass).

- Weight Model:
    - Based on: (1) visibility, (2) altitude at local midnight, 
      (3) Moon/planet angular distance, (4) revisit history, 
      and (5) proximity to previous pointing.

- Real-time logic:
    - Weights are updated between clusters to reflect changing sky conditions.
    - Observations continue until astronomical dawn.

Requirements
------------
- Python 3.7+
- `astropy`, `dataclasses`, `datetime`, `json`, `numpy`, `pandas`, `pathlib`, `typing`, `warnings`
- Input FITS file with field coordinates and visibility data.

Outputs
-------
- JSON: nightly exposure plan per date (`YYYYMMDD_plan.json`)
- FITS: final summary table with per-field exposure counts

"""

from __future__ import annotations

import json
from datetime import date as dt_date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import random
import astropy.units as u
from astropy.coordinates import (
    AltAz, EarthLocation, SkyCoord, get_body, get_sun
)
from astropy.time import Time
from astroplan import moon
from sklearn.cluster import KMeans
from dataclasses import dataclass
import argparse
import warnings
warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration class and defaults
# -----------------------------------------------------------------------------

@dataclass
class Config:
    exposure_time: float                 # Exposure duration [s]
    slew_speed: float                    # Slew rate [deg/s]
    airmass_max: float                   # Maximum airmass for visibility
    revisit_interval: float              # Time between first and second/third pass [s]
    cluster_size_max: int                # Number of fields in each cluster (max 45)
    w_visible_nights: float              # Weight: fields with fewer visible nights
    w_altitude: float                    # Weight: favors higher-altitude fields
    w_unvisited: float                   # Weight: favors under-observed fields
    w_unvisited_alpha: float             # Unvisited reward scale (below target)
    w_unvisited_beta: float              # Visit penalty scale (above target)
    w_d0_dist: float                     # Characteristic distance for slew penalty
    w_gam_dist: float                    # Slew distance exponent
    expected_total_exposures: int        # Target exposures per field across full campaign
    total_nights: Optional[int]          # Number of nights in campaign
    current_day_idx: int                 # Index of current night (0-based)

def default_config():
    return Config(
        exposure_time=30.0,
        slew_speed=0.5,
        airmass_max=2.0,
        revisit_interval=1800.0,
        cluster_size_max=45,
        w_visible_nights=2.0,
        w_altitude=1.0,
        w_unvisited=1.0,
        w_unvisited_alpha=5.0,
        w_unvisited_beta=2.0,
        w_d0_dist=15.0,
        w_gam_dist=2.0,
        expected_total_exposures=20,
        total_nights=None,
        current_day_idx=0,
    )

# -----------------------------------------------------------------------------
# Slew time util
# -----------------------------------------------------------------------------

def slew_sec(a: SkyCoord, b: SkyCoord, cfg: Config) -> float:
    """
    Compute the telescope slew time between two sky positions.

    The slew time is calculated assuming a constant angular speed.

    Parameters:
        a   -- Starting sky position (SkyCoord).
        b   -- Target sky position (SkyCoord).
        cfg -- Configuration object containing the slew speed in deg/sec.

    Returns:
        Slew time in seconds as a float.
    """
    ra_sep = abs((a.ra - b.ra).to(u.deg).value)
    dec_sep = abs((a.dec - b.dec).to(u.deg).value)
    total_sep = max(ra_sep, dec_sep)
    
    return total_sep / cfg.slew_speed

# -----------------------------------------------------------------------------
# Twilight window util (same as v2 but compact)
# -----------------------------------------------------------------------------

def night_window(night: dt_date, site: EarthLocation) -> tuple[Time, Time]:
    """
    Compute the UTC start and end times of the astronomical night based on sun altitude.

    Parameters:
        night -- Local calendar date.
        site  -- Observatory location.

    Returns:
        Tuple of (Time, Time) in UTC: (evening_twilight_end, morning_twilight_start)
    """
    # Define time grid for the full UTC day (start of night to next morning)
    dt_start = datetime.combine(night, datetime.min.time(), tzinfo=timezone.utc)
    time_grid = Time(dt_start) + np.linspace(0, 24, 24 * 60) * u.hour  # 1-min resolution

    # Compute solar altitudes over this day
    sun_alt = get_sun(time_grid).transform_to(AltAz(obstime=time_grid, location=site)).alt

    # Astronomical twilight = Sun below -12 deg
    below_twilight = sun_alt < -12 * u.deg

    # Find the first long period below -12 deg (i.e., the night)
    indices = np.where(below_twilight)[0]
    if len(indices) == 0:
        raise RuntimeError("No astronomical night found for this date at this location.")

    i_start = indices[0]
    i_end = indices[-1]

    return time_grid[i_start], time_grid[i_end]

# -----------------------------------------------------------------------------
# Filter out fields based on altitude
# -----------------------------------------------------------------------------

def get_active_fields(coords: SkyCoord, site: EarthLocation, obstime: Time, cfg: Config) -> np.ndarray:
    """
    Determine which fields are observable during a 1.5-hour window starting at `obstime`.

    For each field, checks whether:
        - altitude is above the minimum threshold (airmass constraint)
        - moon separation ≥ 15 ~ 30 deg
        - planet (Venus, Jupiter) separation ≥ 0 deg
    are satisfied throughout the 1.5-hour window, sampled every 10 minutes.

    Parameters:
        coords   -- SkyCoord array of field centers (ICRS).
        site     -- Observatory location.
        obstime  -- Start time of observation window.
        cfg      -- Config object containing airmass_max.

    Returns:
        Boolean array (N,) where True indicates the field is active over the full window.
    """
    # Generate time grid: 10-minute steps over 1.5 hours
    times = obstime + np.arange(0, 91, 10) * u.min  # [0,10,20,...,90] minutes

    n_fields = len(coords)
    visible = np.ones(n_fields, dtype=bool)

    for t in times:
        altaz = coords.transform_to(AltAz(obstime=t, location=site))
        alt_ok = altaz.secz <= cfg.airmass_max

        moon_pos = get_body("moon", t, location=site)
        phase = moon.moon_illumination(t)
        moon_ok = coords.separation(moon_pos) >= 15 * u.deg + 15 * phase * u.deg

        planets = [get_body(p, t, site) for p in ("venus", "jupiter")]
        separation_matrix = u.Quantity([coords.separation(p) for p in planets])
        min_planet_sep = separation_matrix.min(axis=0)
        planet_ok = min_planet_sep >= 0 * u.deg

        visible &= alt_ok & moon_ok & planet_ok
        if not np.any(visible):
            break  # no need to check further if all eliminated

    return visible

# -----------------------------------------------------------------------------
# Weight Functions
# -----------------------------------------------------------------------------

def weight_visible_nights(tbl, cfg: Config) -> np.ndarray:
    """
    Compute weight based on the number of annually visible nights for each field.
    Fields with fewer visible nights are prioritized.

    Parameters:
        tbl -- DataFrame containing `visible_days` for each field.
        cfg -- Configuration object.

    Returns:
        Normalized 1D weight array of shape (N,), where N is the number of fields.
    """
    k = cfg.w_visible_nights
    days = np.clip(tbl.visible_days, 1, 365)
    weights = 1.0 - ((days - 1) / 364.0) * 0.9
    return k * weights


def weight_alt(coords: SkyCoord, frame: AltAz, cfg: Config) -> np.ndarray:
    """
    Compute altitude-based weight at the given time.

    Fields above 30° altitude receive a linearly increasing weight up to 60°,
    and zero below 30°.

    Parameters:
        coords -- SkyCoord array of field positions.
        frame  -- AltAz frame at the target time/location.
        cfg    -- Configuration object.

    Returns:
        1D weight array of shape (N,).
    """
    k = cfg.w_altitude
    altitudes = coords.transform_to(frame).alt
    norm_alt = np.clip((altitudes - 30 * u.deg) / (60 * u.deg), 0, 1)
    return k * norm_alt


def weight_distance(coords: SkyCoord, current_pt: Optional[SkyCoord], cfg: Config) -> np.ndarray:
    """
    Apply distance-based penalty to reduce telescope slew times.

    Prioritizes fields close to the current pointing to minimize telescope motion.

    Parameters:
        coords     -- SkyCoord array of field positions.
        current_pt -- Current telescope pointing (None if undefined).
        cfg        -- Configuration object.

    Returns:
        1D array of multiplicative weights (penalties ≤ 1.0).
    """
    if current_pt is None:
        return np.ones(len(coords))
    dist_deg = coords.separation(current_pt).deg
    return np.exp(-(dist_deg / cfg.w_d0_dist) ** cfg.w_gam_dist)


def weight_unvisited(history: np.ndarray, cfg: Config) -> np.ndarray: # use daily maximum exposures instead of total exposures ??
    """
    Compute dynamic weighting based on visit history.

    - Fields with fewer visits than the campaign target are rewarded linearly.
    - Fields with excess visits are exponentially suppressed.

    Parameters:
        history -- Array of current visit counts per field.
        cfg     -- Configuration object with campaign metadata.

    Returns:
        1D weight array of shape (N,).
    """
    k = cfg.w_unvisited

    # If campaign info is missing, fallback to inverse history
    if cfg.total_nights is None or cfg.total_nights == 0:
        inv = 1 / np.clip(history + 1, 1, None)
        return k * inv / inv.max()

    # Compute time-proportional exposure target
    day_fraction = (cfg.current_day_idx + 1) / cfg.total_nights
    target = max(day_fraction * cfg.expected_total_exposures, 1)
    alpha = cfg.w_unvisited_alpha
    beta = cfg.w_unvisited_beta

    below_target = history <= target
    penalty = np.ones_like(history, dtype=float)

    # Linear reward for under-visited fields
    penalty[below_target] += alpha * (target - history[below_target]) / target
    # Exponential suppression for over-visited fields
    penalty[~below_target] *= np.exp(-beta * (history[~below_target] - target) / target)

    inv = 1 / np.clip(history + 1, 1, None)
    weighted = inv * penalty
    return k * weighted / weighted.max()

# -----------------------------------------------------------------------------
#   dynamic weights  ‑‑  compute total weight for all fields at given time
# -----------------------------------------------------------------------------

def weights(
    time: Time,
    coords: SkyCoord,
    tbl: pd.DataFrame,
    history: np.ndarray,
    active: np.ndarray,
    current_pt: Optional[SkyCoord],
    site: EarthLocation,
    cfg: Optional[Config] = None,
) -> np.ndarray:
    """
    Compute total weight for all sky fields at given time.

    Includes:
    - Visibility (annual nights)
    - Altitude at local midnight
    - Separation from Moon and planets
    - Visit history balance
    - Distance penalty (if `current_pt` is provided)

    Parameters:
        time        -- observation time (local midnight)
        coords      -- field coordinates (SkyCoord, shape: N)
        tbl         -- table containing `visible_days` for each field
        history     -- array of visit counts per field (length N)
        active      -- boolean array of valid/active fields (length N)
        current_pt  -- current telescope pointing (None if no prior)
        site        -- EarthLocation of observatory
        cfg         -- configuration (uses `default_config()` if None)

    Returns:
        Weight array (shape: N), with 0.0 for inactive fields.
    """
    if cfg is None:
        cfg = default_config()

    frame = AltAz(obstime=time, location=site)

    w = (
        weight_visible_nights(tbl, cfg)
        + weight_alt(coords, frame, cfg)
    )

    w *= weight_unvisited(history, cfg)
    w *= weight_distance(coords, current_pt, cfg)
    w[~active] = 0.0

    return w

# -----------------------------------------------------------------------------
# group_fields_by_chunks  ‑‑  cluster fields into chunks of 45 based on
# ------------------------------------------------------------------------------

def group_fields_by_chunks(coords: SkyCoord, chunk_size=45) -> np.ndarray:
    """
    Cluster fields into groups of approximately `chunk_size` using KMeans.
    This function uses KMeans clustering to partition the fields into
    `N_chunks` groups, where `N_chunks` is determined by the total number of
    fields divided by `chunk_size`.
    Parameters:
        coords      -- SkyCoord array of field positions.
        chunk_size  -- Desired size of each cluster (default: 45).
    Returns:
        1D array of cluster labels (shape: N,), where N is the number of fields.
        Each label corresponds to a cluster index, indicating which group
        the field belongs to.
    Note:
        - The number of clusters is determined by the total number of fields
          divided by `chunk_size`.
        - The clustering is performed in Cartesian coordinates (RA, Dec).
        - The function uses KMeans from scikit-learn with `n_init="auto"` for better stability.
    """
    N_chunks = len(coords) // chunk_size
    xyz = coords.cartesian.xyz.value.T
    kmeans = KMeans(n_clusters=N_chunks, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(xyz)
    return labels

# -----------------------------------------------------------------------------
# schedule_night  ‑‑  fully revised to account for inter‑cluster slews
# -----------------------------------------------------------------------------

def schedule_night(night: dt_date | str,
                   site: EarthLocation,
                   tbl: pd.DataFrame,
                   cfg: Config | None = None,
                   history: np.ndarray | None = None) -> pd.DataFrame:
    """
    Generate an exposure timeline for a single night at a given site.

    Returns a DataFrame with columns:
        field_id · exptime_s · start_utc · ra · dec · repeat (0, 1, 2)

    Parameters:
        night   -- Observation date (as date or string).
        site    -- Observatory location (EarthLocation).
        tbl     -- DataFrame of candidate fields (must include center_ra/dec).
        cfg     -- Observation configuration (exposure time, slew speed, etc.).
        history -- Array of exposure counts per field from previous nights.
    """
    if cfg is None:
        cfg = default_config()

    night = pd.to_datetime(night).date()
    t0, t_end = night_window(night, site)
    t_now = t0

    if history is None:
        history = np.zeros(len(tbl), dtype=int)

    # --- Precompute sky positions and configs ---
    coords  = SkyCoord(tbl.center_ra.values * u.deg,
                       tbl.center_dec.values * u.deg)
    exp     = cfg.exposure_time
    revisit = cfg.revisit_interval

    # --- Initialize ---
    rows = []
    observed_this_night = np.zeros(len(tbl), int)
    current_pointing: SkyCoord | None = None

    while t_now < t_end:
        # Check if any fields are active
        active = get_active_fields(coords, site, t_now, cfg)
        active_idx = np.where(active)[0]
        if len(active_idx) == 0:
            t_now += 60 * u.s
            continue

        coords_active = coords[active_idx]

        # --- 1. Choose new cluster center ---
        w = weights(
            time=t_now + 900 * u.s,
            coords=coords,
            tbl=tbl,
            history=history,
            active=active,
            current_pt=current_pointing,
            site=site,
            cfg=cfg
        )
        w_active = w[active_idx]
        
        labels = group_fields_by_chunks(coords_active, chunk_size=45)
        n_groups = labels.max() + 1
        group_weights = np.zeros(n_groups)
        for i in range(n_groups):
            group_weights[i] = w_active[labels == i].sum()

        best_group = group_weights.argmax()
        in_best_group = np.where(labels == best_group)[0]
        in_best_group_global = active_idx[in_best_group]
        local_weights = w_active[in_best_group_global]
        local_indices = active_idx[in_best_group]

        center_idx = local_indices[local_weights.argmax()]
        
        # Slew to center
        if current_pointing is not None:
            t_now += slew_sec(current_pointing, coords[center_idx], cfg) * u.s
            if t_now >= t_end:
                break
        current_pointing = coords[center_idx]

        # --- 2. Build cluster (neighbors) ---
        idx_active = np.where(active & (observed_this_night < 3))[0]
        dists = coords[idx_active].separation(coords[center_idx]).deg
        order = idx_active[np.argsort(dists)]

        neighbours = []
        time_est = exp
        prev_coord = coords[center_idx]

        for idx in order:
            if idx == center_idx or len(neighbours) >= cfg.cluster_size_max - 1:
                continue
            slew_to = slew_sec(prev_coord, coords[idx], cfg)
            slew_back = slew_sec(coords[idx], coords[center_idx], cfg)
            total_time = time_est + slew_to + exp + slew_back + exp
            if total_time > revisit:
                break
            neighbours.append(idx)
            time_est += slew_to + exp
            prev_coord = coords[idx]

        sequence = [center_idx] + neighbours + [center_idx]
        seq_coords = [coords[i] for i in sequence]

        # --- 3. Do three passes ---
        first_pass_start = t_now
        cluster_observed = np.zeros(len(tbl), int)

        for rep in range(3):
            if rep > 0:
                t_now = first_pass_start + rep * revisit * u.s
            if t_now >= t_end:
                break

            prev_c = None
            for idx, coord in zip(sequence, seq_coords):
                if prev_c is not None:
                    t_now += slew_sec(prev_c, coord, cfg) * u.s
                    if t_now >= t_end:
                        break
                rows.append(dict(
                    field_id=tbl.field_id.iloc[idx].decode("utf-8"),
                    repeat=rep,
                    start_utc=t_now.iso,
                    ra=float(coord.ra.deg),
                    dec=float(coord.dec.deg),
                    exptime_s=exp
                ))
                cluster_observed[idx] += 1
                observed_this_night[idx] += 1
                t_now += exp * u.s
                prev_c = coord
                current_pointing = coord
                if t_now >= t_end:
                    break
            if t_now >= t_end:
                break

        # --- 4. Update history ---
        history += cluster_observed
        if t_now >= t_end:
            break

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# schedule_campaign
# -----------------------------------------------------------------------------

def schedule_campaign(start_date: str, n: int, site: EarthLocation,
                      fits_path: str, out_dir: str,
                      cfg: Config | None = None):
    """
    Generate observation plans for *n* consecutive nights starting from *start_date*.

    Outputs:
        - One JSON file per night: <out_dir>/YYYYMMDD_plan.json
        - One FITS file summarizing total exposures across all fields.

    Parameters:
        start_date -- Campaign start date in "YYYY-MM-DD" (UTC).
        n          -- Number of nights.
        site       -- Observatory location.
        fits_path  -- Path to input FITS file containing field table.
        out_dir    -- Directory for output plans and coverage summary.
        cfg        -- Optional Config object (default will be used if None).
    """
    if cfg is None:
        cfg = default_config()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from astropy.table import Table
    tbl = Table.read(fits_path, format='fits').to_pandas()
    tbl.columns = [c.lower() for c in tbl.columns]
    history = np.zeros(len(tbl), int)

    cfg.total_nights = n
    for i in range(n):
        cfg.current_day_idx = i
        night = pd.to_datetime(start_date).date() + timedelta(days=i)
        if random.random() > 1:
            print(f"{night}: cloudy night, skipped.")
            continue
        plan = schedule_night(night, site, tbl, cfg, history)
        if plan.empty:
            print(f"{night}: no schedule")
            continue
        with open(out / f"{night:%Y%m%d}_plan.json", "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(orient="records"), f, indent=2)
        print(f"{night}: wrote {len(plan)} exposures over {plan.field_id.nunique()} fields")

    # --- Save total exposure coverage summary ---
    summary_tbl = tbl.copy()
    summary_tbl["exposures"] = history
    from astropy.table import Table as _Table
    cov_name = out / f"coverage_{start_date.replace('-', '')}_{n:03d}nights.fits"
    _Table.from_pandas(summary_tbl).write(cov_name, overwrite=True)
    print(f"Coverage table written → {cov_name}")

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Main entry point for the scheduling script with configurable arguments.
    """
    parser = argparse.ArgumentParser(description="Run scheduling campaign.")
    parser.add_argument("--start", type=str, default="2025-07-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=365, help="Number of days to schedule")
    parser.add_argument("--lat", type=float, default=40.393, help="Site latitude in degrees")
    parser.add_argument("--lon", type=float, default=117.575, help="Site longitude in degrees")
    parser.add_argument("--height", type=float, default=960.0, help="Site height in meters")
    parser.add_argument("--fits", type=str, default="survey_fov_footprints.fits", help="Path to input FITS file")
    parser.add_argument("--out", type=str, default="plans", help="Output directory")

    args = parser.parse_args()

    site = EarthLocation(lat=args.lat * u.deg, lon=args.lon * u.deg, height=args.height * u.m)
    schedule_campaign(args.start, args.days, site, args.fits, args.out)

