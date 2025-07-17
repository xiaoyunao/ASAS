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
- pandas DataFrame with scheduled observations.

"""

from __future__ import annotations

import re
from datetime import date as dt_date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import (
    AltAz, EarthLocation, SkyCoord, get_body, get_sun
)
from astropy.time import Time
from astroplan import moon
from sklearn.cluster import KMeans
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# Configuration class and defaults
# -----------------------------------------------------------------------------

@dataclass
class Config:
    exposure_time: float                 # Exposure duration [s]
    slew_speed_ra: float                 # Slew speed in RA direction [deg/s]
    slew_speed_dec: float                # Slew speed in Dec direction [deg/s]
    airmass_max: float                   # Maximum airmass for visibility
    revisit_interval: float              # Time between first and second/third pass [s]
    w_visible_nights: float              # Weight: fields with fewer visible nights
    w_altitude: float                    # Weight: favors higher-altitude fields
    w_unvisited: float                   # Weight: favors under-observed fields
    w_d0_dist: float                     # Characteristic distance for slew penalty
    w_gam_dist: float                    # Slew distance exponent

def default_config():
    return Config(
        exposure_time=30.0,
        slew_speed_ra=0.2,
        slew_speed_dec=0.2,
        airmass_max=2.0,
        revisit_interval=1800.0,
        w_visible_nights=2.0,
        w_altitude=1.0,
        w_unvisited=1.0,
        w_d0_dist=15.0,
        w_gam_dist=2.0,
    )
    
# -----------------------------------------------------------------------------
# Slew time util
# -----------------------------------------------------------------------------

def slew_sec(a: SkyCoord, b: SkyCoord, cfg: Config) -> float:
    """
    Compute the telescope slew time required to move from position `a` to position `b`.

    Assumes constant angular slew speeds in RA and Dec directions independently,
    with the final slew time determined by the slower axis movement.
    A minimum slew overhead of 10 seconds (e.g., for readout or settling) is enforced.

    Args:
        a (SkyCoord):
            Starting sky coordinate of the telescope pointing.

        b (SkyCoord):
            Target sky coordinate to slew to.

        cfg (Config):
            Scheduler configuration containing slew speed parameters.

    Returns:
        float:
            Estimated slew time in seconds (minimum 10 seconds).
    """
    # Calculate angular separations in degrees for RA and Dec axes
    ra_sep = abs((a.ra - b.ra).to(u.deg).value) / cfg.slew_speed_ra
    dec_sep = abs((a.dec - b.dec).to(u.deg).value) / cfg.slew_speed_dec

    # Total slew time is the maximum of RA and Dec slew times (serial movement)
    total_sep = max(ra_sep, dec_sep) + 10.0

    # Enforce a minimum slew overhead time (e.g., CCD readout, settling)
    return max(10.0, total_sep)

# -----------------------------------------------------------------------------
# Twilight window util
# -----------------------------------------------------------------------------

def night_window(night: dt_date, site: EarthLocation) -> Tuple[Time, Time]:
    """
    Calculate the start and end times of the astronomical night for a given
    date and observatory location.

    The astronomical night is defined as the time interval when the sun is
    more than 12 degrees below the horizon (nautical twilight end).

    Args:
        night (datetime.date):
            The local calendar date for the night.

        site (EarthLocation):
            The observatory location.

    Returns:
        Tuple[Time, Time]:
            Start and end times of the astronomical night as astropy Time
            objects in UTC.

    Raises:
        RuntimeError:
            If no period with sun altitude below -12 degrees is found on the date.
    """
    # Generate a time grid with 1-minute resolution over 24 hours starting at midnight UTC
    dt_start = datetime.combine(night, datetime.min.time(), tzinfo=timezone.utc)
    time_grid = Time(dt_start) + np.linspace(0, 24 * 60, 24 * 60) * u.min

    # Calculate sun altitude for each time in the grid at the observatory site
    sun_alt = get_sun(time_grid).transform_to(AltAz(obstime=time_grid, location=site)).alt

    # Identify time indices where sun is below -12 degrees altitude (astronomical night)
    below_twilight = sun_alt < -12 * u.deg
    indices = np.where(below_twilight)[0]

    if len(indices) == 0:
        raise RuntimeError("No astronomical night found for given date and site.")

    # Return the first and last times when sun is below -12 deg
    return time_grid[indices[0]], time_grid[indices[-1]]

# -----------------------------------------------------------------------------
# FieldWeighter class for computing observational weights
# -----------------------------------------------------------------------------

class FieldWeighter:
    """
    Encapsulates the logic to compute observational priority weights for fields
    based on a combination of factors:
    - Annual visibility duration (fields with fewer visible nights favored)
    - Altitude at local midnight (higher altitude preferred)
    - Visit history (fields with fewer prior visits favored)
    - Slew distance penalty (fields closer to current pointing favored)

    All weight components are combined multiplicatively or additively
    and scaled by configurable parameters from the scheduler config.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the FieldWeighter with scheduler configuration.

        Args:
            cfg (Config): Configuration with weight scaling factors.
        """
        self.cfg = cfg

    def weight_visible_nights(self, tbl: pd.DataFrame) -> np.ndarray:
        """
        Compute a weight penalizing fields visible on many nights,
        so that fields with fewer available nights are favored.

        The weight scales roughly from 1.0 (fields visible only 1 night/year)
        down to 0.1 (fields visible all year).

        Args:
            tbl (pd.DataFrame):
                DataFrame containing a `visible_days` column indicating
                the number of visible nights per year for each field.

        Returns:
            np.ndarray:
                Weight values scaled by the configured visibility weight.
        """
        k = self.cfg.w_visible_nights
        days = np.clip(tbl.visible_days.values, 1, 365)
        # Linear scaling: fewer visible days → higher weight
        weights = 1.0 - ((days - 1) / 364.0) * 0.9  # from 1.0 down to 0.1
        
        # return scaled weights
        return k * weights

    def weight_altitude(self, coords: SkyCoord, frame: AltAz) -> np.ndarray:
        """
        Compute weights based on the altitude of fields at the specified time and location.

        - Fields below 30 degrees altitude receive zero weight.
        - Weight increases linearly between 30 and 60 degrees altitude.
        - Fields above 60 degrees receive full weight.

        Args:
            coords (SkyCoord):
                Sky coordinates of fields.

            frame (AltAz):
                AltAz coordinate frame at observation time and location.

        Returns:
            np.ndarray:
                Altitude-based weights scaled by configured altitude weight.
        """
        k = self.cfg.w_altitude
        altitudes = coords.transform_to(frame).alt
        # Normalize altitude from 30° (0 weight) to 60° (full weight)
        norm_alt = np.clip((altitudes - 30 * u.deg) / (60 * u.deg), 0, 1)
        
        # return scaled weights
        return k * norm_alt.value

    def weight_distance(self, coords: SkyCoord, current_pt: Optional[SkyCoord]) -> np.ndarray:
        """
        Apply a penalty based on angular distance to the current telescope pointing.

        Closer fields receive a weight closer to 1, while distant fields
        are exponentially downweighted to minimize slew overheads.

        Args:
            coords (SkyCoord):
                Sky coordinates of all candidate fields.

            current_pt (Optional[SkyCoord]):
                Current telescope pointing coordinate; if None, no penalty applied.

        Returns:
            np.ndarray:
                Slew distance weights in (0, 1], with 1 indicating no penalty.
        """
        if current_pt is None:
            return np.ones(len(coords))

        dist_deg = coords.separation(current_pt).deg
        
        # Exponential decay penalty with characteristic scale and power-law exponent
        return np.exp(-(dist_deg / self.cfg.w_d0_dist) ** self.cfg.w_gam_dist)

    def weight_unvisited(self, history: np.ndarray, active: np.ndarray) -> np.ndarray:
        """
        Compute a weight that favors fields with fewer prior visits.
        This weight is scaled by the configured unvisited weight factor.
        
        Args:
            history (np.ndarray):
                Array of visit counts for each field (length N).
            active (np.ndarray):
                Boolean array indicating which fields are currently active.
        
        Returns:
            np.ndarray:
                Weight values scaled by the configured unvisited weight.
                Fields with no visits get a weight of 1.0, while those with
                more visits are downweighted proportionally.
        """
        k = self.cfg.w_unvisited
        weighted = np.zeros_like(history, dtype=float)

        if np.any(active):
            max_obs = history[active].max()
            if max_obs > 0:
                weighted[active] = 1.0 - (history[active] / max_obs)
            else:
                weighted[active] = 1.0  # all zero-visits case

        # Scale the weights by the configured unvisited weight factor
        return k * weighted

    def compute(
        self,
        time: Time,
        coords: SkyCoord,
        tbl: pd.DataFrame,
        history: np.ndarray,
        active: np.ndarray,
        current_pt: Optional[SkyCoord],
        site: EarthLocation,
    ) -> np.ndarray:
        """
        Compute the combined weights for scheduling fields based on multiple factors.
        This method combines visibility, altitude, unvisited history, and distance penalties
        into a single weight array.
        Args:
            time (Time):
                Current observation time as an astropy Time object.
            coords (SkyCoord):
                Sky coordinates of all candidate fields.
            tbl (pd.DataFrame):
                DataFrame containing field metadata, including visibility days.
            history (np.ndarray):
                Array of visit counts for each field.
            active (np.ndarray):
                Boolean array indicating which fields are currently active.
            current_pt (Optional[SkyCoord]):
                Current telescope pointing coordinate; if None, no distance penalty applied.
            site (EarthLocation):
                Observatory location for altitude calculations.
        
        Returns:
            np.ndarray:
                Combined weight values for each field, scaled by the configured weights.
        """
        frame = AltAz(obstime=time, location=site)

        # Additive weights: visibility & altitude
        w = (
            self.weight_visible_nights(tbl)
            + self.weight_altitude(coords, frame)
        )

        # Multiplicative weights: unvisited & distance penalty
        w *= self.weight_unvisited(history, active)
        w *= self.weight_distance(coords, current_pt)
        w[~active] = 0.0

        return w

# -----------------------------------------------------------------------------
# Scheduler class for managing field selection and sequencing
# -----------------------------------------------------------------------------

class Scheduler:
    """
    Scheduler class manages the selection and sequencing of astronomical fields
    for observation based on site conditions, field visibility, observational
    constraints, and scheduling configuration.

    It supports computing active fields, selecting fields by combined weights and
    spatial clustering, optimizing observation sequences by minimizing slew time,
    and producing detailed nightly schedules including fallback observations.
    
    Usage:
        scheduler = Scheduler(site, tbl, cfg, history)
        active_fields = scheduler.get_active_fields(obstime)
        weights = scheduler.compute_weights(time, coords, tbl, history, active, current_pt, site)
    """

    def __init__(self, site, tbl, history):
        """
        Initialize the Scheduler with site location, field table, configuration,
        and historical observation data.

        Args:
            site (EarthLocation): Observatory location for coordinate transformations.
            tbl (pd.DataFrame): Table containing field metadata including RA/Dec.
            history (np.ndarray): Array or structure with past visit counts per field.
        """
        self.site = site
        self.tbl = tbl.copy()
        self.cfg = default_config()
        self.history = history
        # SkyCoord array for all fields based on center RA/Dec in degrees
        self.coords = SkyCoord(
            ra=self.tbl['center_ra'].values * u.deg,
            dec=self.tbl['center_dec'].values * u.deg
        )
        # Normalize column names to lowercase for consistent access
        self.tbl.columns = [c.lower() for c in self.tbl.columns]

    def get_active_fields(self, obstime: Time) -> np.ndarray:
        """
        Determine which fields are currently observable based on multiple criteria:
        - Altitude cut based on max airmass threshold
        - Minimum angular separation from the Moon, accounting for lunar phase
        - Minimum angular separation from bright planets (Venus, Jupiter)
        
        Checks visibility in 10-minute intervals over the next 90 minutes and
        returns fields that remain visible throughout.

        Args:
            obstime (Time): Current observation time as an astropy Time object.

        Returns:
            np.ndarray: Boolean mask array indicating active (observable) fields.
        """
        times = obstime + np.arange(0, 91, 10) * u.min
        n_fields = len(self.coords)
        visible = np.ones(n_fields, dtype=bool)

        for t in times:
            # Compute altitude and apply airmass threshold
            altaz = self.coords.transform_to(AltAz(obstime=t, location=self.site))
            alt_ok = altaz.secz <= self.cfg.airmass_max

            # Moon position and illumination phase for exclusion zone calculation
            moon_pos = get_body("moon", t, location=self.site)
            phase = moon.moon_illumination(t)
            moon_ok = self.coords.separation(moon_pos) >= 15 * u.deg + 15 * phase * u.deg

            # Compute separations from Venus and Jupiter, no exclusion radius currently applied
            planets = [get_body(p, t, self.site) for p in ("venus", "jupiter")]
            separation_matrix = u.Quantity([self.coords.separation(p) for p in planets])
            min_planet_sep = separation_matrix.min(axis=0)
            planet_ok = min_planet_sep >= 0 * u.deg

            # Update visibility mask
            visible &= alt_ok & moon_ok & planet_ok
            if not np.any(visible):
                break

        return visible

    def select_fields_by_weight_and_proximity(self, time, active, chunk_size=45) -> list[int]:
        """
        Select a subset of active fields to observe, prioritizing by computed weights
        and spatial clustering.

        - If fewer active fields than chunk_size, return all active indices.
        - Otherwise, cluster active fields via KMeans in 3D Cartesian space.
        - Select the cluster with highest total weight.
        - Within the cluster, sort fields by angular distance from the cluster's
          highest-weighted field center.
        - Return up to chunk_size field indices in order.

        Args:
            time (Time): Current observation time.
            active (np.ndarray): Boolean array of active fields.
            chunk_size (int): Maximum number of fields to select.

        Returns:
            list[int]: List of selected field indices.
        """
        active_idx = np.where(active)[0]
        if len(active_idx) < chunk_size:
            return active_idx.tolist()

        weighter = FieldWeighter(self.cfg)
        weights = weighter.compute(
            time=time,
            coords=self.coords,
            tbl=self.tbl,
            history=self.history,
            active=active,
            current_pt=None,
            site=self.site
        )
        w_active = weights[active_idx]

        # Convert coordinates of active fields to Cartesian XYZ for clustering
        xyz = self.coords[active_idx].cartesian.xyz.value.T
        N_chunks = len(active_idx) // chunk_size
        kmeans = KMeans(n_clusters=N_chunks, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(xyz)

        # Sum weights per cluster to find the most promising cluster
        group_weights = np.array([w_active[labels == i].sum() for i in range(N_chunks)])
        best_label = group_weights.argmax()

        best_idx = active_idx[labels == best_label]
        # Find highest-weight field in selected cluster to serve as cluster center
        center_idx = best_idx[np.argmax(weights[best_idx])]
        center_coord = self.coords[center_idx]

        # Sort cluster fields by angular distance to cluster center
        dists = self.coords.separation(center_coord).deg
        sorted_idx = np.argsort(dists)

        # Return up to chunk_size fields from the cluster, preserving order by proximity
        return [i for i in sorted_idx if i in best_idx][:chunk_size]

    def optimize_sequence_by_slew_time(self, field_indices, current_pt=None) -> list[int]:
        """
        Order a sequence of field indices to minimize total telescope slew time.

        Uses a greedy nearest neighbor heuristic starting from current pointing
        or nearest field if current_pt is None.

        Args:
            field_indices (list[int]): List of candidate field indices to sequence.
            current_pt (SkyCoord, optional): Current telescope pointing coordinate.

        Returns:
            list[int]: Ordered list of field indices optimized for minimal slew.
        """
        if not field_indices:
            return []

        remaining = set(field_indices)
        sequence = []

        if current_pt is None:
            # Start from the field closest to arbitrary current pointing
            current_pt = self.coords[next(iter(remaining))]

        # Find initial field closest to current pointing
        current_idx = min(remaining, key=lambda i: self.coords[i].separation(current_pt).deg)
        sequence.append(current_idx)
        remaining.remove(current_idx)

        # Greedily select the next closest field until none remain
        while remaining:
            current_coord = self.coords[current_idx]
            next_idx = min(remaining, key=lambda i: slew_sec(current_coord, self.coords[i], self.cfg))
            sequence.append(next_idx)
            remaining.remove(next_idx)
            current_idx = next_idx

        return sequence

    def main_observation(self, sequence, start_time, tonight_history) -> tuple[list, Time, SkyCoord]:
        """
        Simulate the main observation sequence for a given list of fields.

        - Slews between fields incur time penalties based on slew_sec.
        - Observations have fixed exposure time from configuration.
        - Revisit interval is enforced with breaks to avoid overly rapid repeats.
        - Tracks which fields have been observed tonight to avoid duplicates.

        Args:
            sequence (list[int]): Ordered list of field indices to observe.
            start_time (Time): Observation start time.
            tonight_history (list): List of dicts recording fields observed tonight.

        Returns:
            tuple:
                - list of dicts: Observation logs with timing and pointing info.
                - Time: End time after completing observations.
                - SkyCoord: Final telescope pointing after sequence.
        """
        rows = []
        exp = self.cfg.exposure_time
        revisit = self.cfg.revisit_interval
        t_now = start_time
        current_pointing = None
        seen_ids = set(entry["field_id"] for entry in tonight_history)

        # Repeat the sequence up to 3 times, enforcing revisit intervals
        for rep in range(3):
            for idx in sequence:
                coord = self.coords[idx]

                # Add slew time if moving from previous pointing, otherwise add nominal delay
                if current_pointing:
                    t_now += slew_sec(current_pointing, coord, self.cfg) * u.s
                else:
                    t_now += 60 * u.s

                # Stop if revisit interval exceeded for the current repetition
                if t_now - start_time >= revisit * (rep + 1) * u.s:
                    break

                # Record observation info
                field_id = self.tbl.field_id.iloc[idx].decode("utf-8")
                rows.append(dict(
                    field_id=field_id,
                    repeat=rep + 1,
                    start_utc=t_now.iso,
                    ra=float(coord.ra.deg),
                    dec=float(coord.dec.deg),
                    exptime_s=exp
                ))

                # Add to tonight's history if not previously observed
                if field_id not in seen_ids:
                    tonight_history.append(dict(field_id=field_id, coord=coord))
                    seen_ids.add(field_id)

                t_now += exp * u.s
                current_pointing = coord

        return rows, t_now, current_pointing

    def fallback_observation(self, tonight_history, current_pointing, t_now, t_end) -> list[dict]:
        """
        Generate a fallback observation schedule for the remainder of the night.

        - Select active fields at current time.
        - Rank by computed weights, pick top cluster center.
        - Optimize observation sequence by slew time.
        - Schedule observations until night end or no more time remains.

        Args:
            tonight_history (list): List of dicts of fields observed tonight.
            current_pointing (SkyCoord): Current telescope pointing coordinate.
            t_now (Time): Current observation time.
            t_end (Time): Night end time.

        Returns:
            list[dict]: Observation logs for fallback fields.
        """
        exp = self.cfg.exposure_time
        active = self.get_active_fields(t_now)
        if not np.any(active):
            return []

        weights = FieldWeighter(self.cfg).compute(
            time=t_now,
            coords=self.coords,
            tbl=self.tbl,
            history=self.history,
            active=active,
            current_pt=current_pointing,
            site=self.site
        )

        active_idx = np.where(active)[0]
        top_idx = np.argmax(weights)
        top_coord = self.coords[top_idx]

        # Sort active fields by distance from highest weighted field
        dists = self.coords[active_idx].separation(top_coord).deg
        sorted_idx = np.argsort(dists)
        fallback_idx = [i for i in active_idx[sorted_idx]][:179]

        # Optimize fallback sequence by slew time
        sequence = self.optimize_sequence_by_slew_time(fallback_idx, current_pointing)

        rows = []
        for idx in sequence:
            coord = self.coords[idx]
            if current_pointing:
                t_now += slew_sec(current_pointing, coord, self.cfg) * u.s
            else:
                t_now += 60 * u.s

            if t_now + exp * u.s > t_end:
                break

            field_id = self.tbl.field_id.iloc[idx].decode("utf-8")
            rows.append(dict(
                field_id=field_id,
                repeat=-1,
                start_utc=t_now.iso,
                ra=float(coord.ra.deg),
                dec=float(coord.dec.deg),
                exptime_s=exp
            ))
            t_now += exp * u.s
            current_pointing = coord

        return rows

    def schedule_night(self, night: dt_date) -> pd.DataFrame:
        """
        Produce a full observing schedule for a given night.

        - Defines night start and end times via night_window.
        - Loops until near night end or no fields left.
        - Generates main observation blocks and fallback observations.
        - Returns complete schedule as a pandas DataFrame.

        Args:
            night (date): Date object representing the night to schedule.

        Returns:
            pd.DataFrame: DataFrame of scheduled observations with timing and field info.
        """
        t0, t_end = night_window(night, self.site)
        t_now = t0
        rows_all, tonight_history = [], []

        while t_now + timedelta(minutes=90) < t_end:
            active = self.get_active_fields(t_now)
            sequence = self.select_fields_by_weight_and_proximity(t_now, active)
            if not sequence:
                break
            sequence = self.optimize_sequence_by_slew_time(sequence)
            rows_main, t_now, current_pointing = self.main_observation(
                sequence, t_now, tonight_history
            )
            rows_all.extend(rows_main)
            if t_now + timedelta(minutes=30) >= t_end:
                break

        rows_fallback = self.fallback_observation(
            tonight_history, current_pointing, t_now, t_end
        )
        rows_all.extend(rows_fallback)
        
        # return as DataFrame
        return pd.DataFrame(rows_all)

# -----------------------------------------------------------------------------
# HistoryManager class for exposure history management
# -----------------------------------------------------------------------------
class HistoryManager:
    """
    Manages exposure history for fields, storing daily cumulative updates
    as versioned FITS files in a dedicated history directory.

    Example:
        history = HistoryManager(
            history_dir="./exposure_history",
            template_path="./survey_fov_footprints.fits"
        )
        history.update_from_directory("/processed/20250713/L1")
    """

    def __init__(self, history_dir: str, template_path: str = "./survey_fov_footprints.fits"):
        self.history_dir = Path(history_dir)
        self.template_path = Path(template_path)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._table = None  # will be loaded during update

    def _get_latest_history_file(self) -> Path | None:
        """
        Return the latest history FITS file from the directory.

        Returns:
            Path or None
        """
        # make sure the directory exists
        if not self.history_dir.exists():
            self.history_dir.mkdir(parents=True, exist_ok=True)
            
        fits_files = sorted(self.history_dir.glob("????????_exposure_history.fits"))
        return fits_files[-1] if fits_files else None

    def _load_or_initialize_history(self) -> Table:
        """
        Load the latest history file, or initialize from template if none exists.

        Returns:
            Table
        """
        latest_file = self._get_latest_history_file()
        if latest_file:
            return Table.read(latest_file, format='fits')
        else:
            base_table = Table.read(self.template_path, format='fits')
            if 'field_id' not in base_table.colnames:
                raise ValueError("Template FITS must contain 'field_id' column.")
            if base_table['field_id'].dtype.kind in {'S', 'U'}:
                base_table['field_id'] = base_table['field_id'].astype(str)
            base_table['exposure_count'] = np.zeros(len(base_table), dtype=int)
            return base_table

    def update_from_directory(self, directory: str, initialize_only: bool = False, force: bool = False) -> None:
        """
        Update exposure history from a given night's directory and save it.

        Args:
            directory (str): Path like "/processed/yyyymmdd/L1"
            initialize_only (bool): If True, skip scanning directory and just
                                    create a new history file with the given date.
            force (bool): If True, allows overwriting existing history for the same date.
        """
        dir_path = Path(directory)
        match = re.search(r'/(\d{8})/', str(dir_path) + "/")
        if not match:
            raise ValueError("Directory path must contain a date in 'yyyymmdd' format.")
        obs_date = match.group(1)

        save_path = self.history_dir / f"{obs_date}_exposure_history.fits"
        if save_path.exists() and not force:
            raise FileExistsError(f"History file already exists for date {obs_date}: {save_path}. Use force=True to overwrite.")

        # Load previous history (not current day's!) or initialize
        self._table = self._load_or_initialize_history()

        if not initialize_only:
            if not dir_path.exists():
                raise FileNotFoundError(f"Directory does not exist: {directory}")

            pattern = re.compile(r"OBJ_SV(\d{5})_\d{4}\.fits\.gz$")
            field_id_counter = {}

            for file in dir_path.glob("*.fits.gz"):
                match = pattern.search(file.name)
                if match:
                    field_id = f"SV{match.group(1)}"
                    field_id_counter[field_id] = field_id_counter.get(field_id, 0) + 1

            for i, fid in enumerate(self._table['field_id']):
                if fid in field_id_counter:
                    self._table['exposure_count'][i] += field_id_counter[fid]

        self._table.write(save_path, overwrite=True)
        print(f"[INFO] History saved to {save_path}")

