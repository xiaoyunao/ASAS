from astropy.coordinates import SkyCoord, AltAz, EarthLocation
from astropy.time import Time
from astroplan import Observer
import astropy.units as u
from datetime import datetime, timezone, timedelta
from astropy.utils import iers
iers.conf.auto_download = False
iers.conf.iers_auto_url = None
iers.conf.iers_degraded_accuracy = 'warn'

# 设置观测台站
lon, lat, height = 117.575, 40.393, 960
observer_location = EarthLocation(lon=lon*u.deg, lat=lat*u.deg, height=height*u.m)
observer = Observer(location=observer_location, name="Xinglong")

# ===== 用户输入区域 =====
# 输入本地时间（北京时间）
local_time_str = "20250714 01:40"  # 输入格式为 YYYYMMDD HH:MM
ra_input = "20:57:25.892"          # 支持 h:m:s 格式或 float（小时或度）
dec_input = "+00:47:26.799"        # 支持 d:m:s 格式或 float

# ===== 时间转换 =====
# 将本地时间（北京时间，UTC+8）转换为 UTC
local_dt = datetime.strptime(local_time_str, "%Y%m%d %H:%M")
utc_dt = local_dt - timedelta(hours=8)
time = Time(utc_dt, scale='utc')

# ===== RA/DEC 处理 =====
try:
    # 如果是字符串，按 hms/dms 格式解析
    coord = SkyCoord(ra=ra_input, dec=dec_input, unit=(u.hourangle, u.deg), frame='icrs')
except ValueError:
    # 否则按 float（度）解析
    ra_deg = float(ra_input)
    dec_deg = float(dec_input)
    coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')

# ===== 坐标转换 AltAz =====
altaz = coord.transform_to(AltAz(obstime=time, location=observer_location))

# ===== 输出结果 =====
print(f"UTC Time     : {time.iso}")
print(f"RA (deg)     : {coord.ra.deg:.6f}")
print(f"Dec (deg)    : {coord.dec.deg:.6f}")
print(f"Altitude     : {altaz.alt.deg:.2f} deg")
print(f"Azimuth      : {altaz.az.deg:.2f} deg")
