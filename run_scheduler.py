import argparse
import json
import re
from pathlib import Path
from datetime import datetime
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.table import Table
from main import Scheduler, HistoryManager
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="astropy")

OBJ_PATTERN = re.compile(r"OBJ_SV(\d{5})_\d{4}\.fits\.gz$")

def parse_args():
    parser = argparse.ArgumentParser(description="Run scheduling campaign.")

    parser.add_argument("--date", type=str, required=True,
                        help="Date in YYYY-MM-DD format (e.g., 2025-07-13)")
    parser.add_argument("--lat", type=float, default=40.393,
                        help="Observatory latitude in degrees (default: 40.393)")
    parser.add_argument("--lon", type=float, default=117.575,
                        help="Observatory longitude in degrees (default: 117.575)")
    parser.add_argument("--height", type=float, default=960.0,
                        help="Observatory height in meters (default: 960.0)")
    parser.add_argument("--fov-fits", type=str, default="./survey_fov_footprints.fits",
                        help="Path to input FOV FITS file (default: ./survey_fov_footprints.fits)")
    parser.add_argument("--output-dir", type=str, default="./plans",
                        help="Output directory for the plan (default: ./plans)")
    parser.add_argument("--processed-root", type=str, default="./processed",
                        help="Root path to processed observation directories (default: ./processed)")
    parser.add_argument("--history-dir", type=str, default="./exposure_history",
                        help="Directory to store exposure history files (default: ./exposure_history)")
    parser.add_argument("--initialize-only", action="store_true",
                        help="Only initialize exposure history for the date without scanning observation files")
    parser.add_argument("--force", action="store_true",
                        help="Force overwrite of today's history file if it already exists")

    return parser.parse_args()

def find_latest_observation_date(root: Path, before_date: datetime.date) -> str | None:
    date_dirs = []
    for subdir in root.iterdir():
        if subdir.is_dir() and subdir.name.isdigit() and len(subdir.name) == 8:
            try:
                date = datetime.strptime(subdir.name, "%Y%m%d").date()
                if date < before_date:
                    date_dirs.append((date, subdir))
            except ValueError:
                continue

    for date, d in sorted(date_dirs, key=lambda x: x[0], reverse=True):
        l1_dir = d / "L1"
        if not l1_dir.exists():
            continue
        for file in l1_dir.glob("*.fits.gz"):
            if OBJ_PATTERN.match(file.name):
                return date.strftime("%Y-%m-%d")

    print("[INFO] No previous observation directory with matching files found.")
    return None


def main():
    args = parse_args()

    night = datetime.strptime(args.date, "%Y-%m-%d").date()
    obs_date_str = night.strftime("%Y%m%d")
    last_obs_date_str = find_latest_observation_date(Path(args.processed_root), night).replace("-", "")
    
    if last_obs_date_str is None:
        last_obs_date_str = obs_date_str  # Fallback to today if no previous date found
        
    obs_dir = Path(args.processed_root) / last_obs_date_str / "L1"

    site = EarthLocation(lat=args.lat * u.deg, lon=args.lon * u.deg, height=args.height * u.m)

    tbl = Table.read(args.fov_fits, format="fits").to_pandas()
    tbl.columns = [c.lower() for c in tbl.columns]

    history_mgr = HistoryManager(history_dir=args.history_dir, template_path=args.fov_fits)
    history_mgr.update_from_directory(directory=str(obs_dir),
                                      initialize_only=args.initialize_only,
                                      force=args.force)

    history_path = Path(args.history_dir) / f"{last_obs_date_str}_exposure_history.fits"
    if not history_path.exists():
        raise FileNotFoundError(f"Expected history file not found: {history_path}")
    
    history_tbl = Table.read(history_path, format="fits")
    history = history_tbl['exposure_count'].data

    scheduler = Scheduler(site, tbl, history)
    plan = scheduler.schedule_night(night)

    if plan.empty:
        print(f"[INFO] {night}: no schedule generated.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{obs_date_str}_plan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(orient="records"), f, indent=2)

    print(f"[INFO] {night}: wrote {len(plan)} exposures over {plan.field_id.nunique()} fields to {out_path}")


if __name__ == "__main__":
    main()
