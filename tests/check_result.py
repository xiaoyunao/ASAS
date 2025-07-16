import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
from astropy import units as u

from zoneinfo import ZoneInfo

local_tz = ZoneInfo("Asia/Shanghai")  # 北京时间

# 观测点
site = EarthLocation(lat=40.393*u.deg, lon=117.575*u.deg, height=960*u.m)

# 读取json文件
json_file = "./plans/20260628_plan.json"
with open(json_file, "r", encoding="utf-8") as f:
    exposures = json.load(f)

# 解析曝光时间和场中心坐标（RA, Dec）
times = [Time(exp["start_utc"]) for exp in exposures]
ras = np.array([exp["ra"] for exp in exposures]) * u.deg
decs = np.array([exp["dec"] for exp in exposures]) * u.deg
fields = SkyCoord(ras, decs)

fig, ax = plt.subplots(figsize=(10, 6))
plt.title("Nightly exposures (Azimuth vs Altitude)", pad=20)

ax.set_xlim(0, 360)
ax.set_ylim(0, 90)
ax.set_xlabel("Azimuth (deg, N=0°, E=90°)")
ax.set_ylabel("Altitude (deg)")

ax.grid(True)

# 画点的函数
def plot_object_no_filter(ax, coord_altaz, color, marker, label):
    az = coord_altaz.az.degree
    alt = coord_altaz.alt.degree
    ax.scatter(az, alt, color=color, marker=marker, s=100, label=label)
    ax.text(az, alt, f"{label}\n{alt:.1f}°", fontsize=8)

def update(frame):
    ax.clear()
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Azimuth (deg, N=0°, E=90°)")
    ax.set_ylabel("Altitude (deg)")
    ax.grid(True)
    local_dt = times[frame].to_datetime(timezone=local_tz)
    # 格式化显示本地时间字符串
    local_time_str = local_dt.strftime("%Y-%m-%d %H:%M:%S")
    plt.title(f"Exposure {frame+1} at {local_time_str}", pad=20)

    altaz_frame = AltAz(obstime=times[frame], location=site)

    # 天体地平坐标
    moon_altaz = get_body('moon', times[frame], location=site).transform_to(altaz_frame)
    venus_altaz = get_body('venus', times[frame], location=site).transform_to(altaz_frame)
    jupiter_altaz = get_body('jupiter', times[frame], location=site).transform_to(altaz_frame)

    # 曝光场地平坐标
    field_altaz = fields[frame].transform_to(altaz_frame)

    plot_object_no_filter(ax, field_altaz, color="red", marker="*", label="Field")
    plot_object_no_filter(ax, moon_altaz, color="gray", marker="o", label="Moon")
    plot_object_no_filter(ax, venus_altaz, color="yellow", marker="o", label="Venus")
    plot_object_no_filter(ax, jupiter_altaz, color="orange", marker="o", label="Jupiter")

    # 避免重复label导致图例冗余
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

ani = animation.FuncAnimation(fig, update, frames=len(exposures), interval=5, repeat=False)
date = json_file.split("/")[-1].split("_")[0]
output_file = f"exposures_{date}.gif"
ani.save(output_file, writer='ffmpeg', fps=10, dpi=300)
plt.show()
