import os
import json
import numpy as np
from astropy.time import Time
from astropy.io import fits
from astropy.coordinates import SkyCoord, AltAz, EarthLocation
import astropy.units as u
from tqdm import tqdm

# 观测站点
site = EarthLocation(lat=40.393*u.deg, lon=117.575*u.deg, height=960*u.m)

# 目录
json_dir = "../plans"
json_files = sorted([f for f in os.listdir(json_dir) if f.endswith("_plan.json")])

# 存储结果
records = []

for day_idx, file in tqdm(enumerate(json_files), total=len(json_files), desc="Processing plans"):
    with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
        plan = json.load(f)

    if not plan:
        continue

    # 拆解字段
    times = Time([p["start_utc"] for p in plan])
    ra = np.array([p["ra"] for p in plan]) * u.deg
    dec = np.array([p["dec"] for p in plan]) * u.deg
    field_ids = [p["field_id"] for p in plan]
    exptimes = [p["exptime_s"] for p in plan]
    date = file.split("_")[0]

    coords = SkyCoord(ra=ra, dec=dec, frame="icrs")
    altaz = coords.transform_to(AltAz(obstime=times, location=site))
    altitudes = altaz.alt.deg

    for fid, t, r, d, alt, exp in zip(field_ids, times.iso, ra, dec, altitudes, exptimes):
        records.append((date, day_idx, fid, t, r.value, d.value, alt, exp))

# 转为 numpy structured array
dtype = [
    ("date", "U10"),
    ("day", "i4"),
    ("field_id", "U10"),
    ("time_utc", "U25"),
    ("ra", "f8"),
    ("dec", "f8"),
    ("altitude", "f8"),
    ("exptime", "f8")
]
table = np.array(records, dtype=dtype)

# 写入 FITS 文件
hdu = fits.BinTableHDU(data=table)
hdu.writeto("./daily_exposure_altitudes.fits", overwrite=True)

print("Saved to daily_exposure_altitudes.fits")
