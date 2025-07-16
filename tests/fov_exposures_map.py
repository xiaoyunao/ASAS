import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from astropy.coordinates import Angle
from astropy.time import Time
import astropy.units as u
import glob
from astropy.table import Table
from zoneinfo import ZoneInfo

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

# ==== 安全绘制跨180°的 fill 多边形 ====
def plot_aitoff_segments(ax, az, alt, **kwargs):
    az = np.mod(az + np.pi, 2 * np.pi) - np.pi  # wrap to [-π, π]
    diff = np.abs(np.diff(az))
    brk = np.where(diff > np.pi)[0] + 1
    idx = np.concatenate(([0], brk, [len(az)]))
    for s, e in zip(idx[:-1], idx[1:]):
        ax.fill(az[s:e], alt[s:e], **kwargs)

# ==== 设置 ====
local_tz = ZoneInfo("Asia/Shanghai")
json_files = sorted(glob.glob("../plans/*_plan.json"))
exposures = []

for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        exposures.extend(json.load(f))

# ==== 读取天区 footprints ====
fov_table = Table.read("../survey_fov_footprints.fits", format="fits")
field_ids = np.array([row["field_id"] for row in fov_table])

# 构建 footprint 多边形映射
field_polygons = {}
for row in fov_table:
    ra_corners = np.array([
        row["corner_ra_1"],
        row["corner_ra_2"],
        row["corner_ra_3"],
        row["corner_ra_4"],
        row["corner_ra_1"],  # 闭合
    ])
    dec_corners = np.array([
        row["corner_dec_1"],
        row["corner_dec_2"],
        row["corner_dec_3"],
        row["corner_dec_4"],
        row["corner_dec_1"],
    ])
    ra_rad = Angle(ra_corners * u.deg).wrap_at(180 * u.deg).radian
    dec_rad = Angle(dec_corners * u.deg).radian
    field_polygons[row["field_id"]] = (ra_rad, dec_rad)

# ==== 准备时间与 field_id ====
times = [Time(exp["start_utc"]) for exp in exposures]
field_seq = [exp["field_id"] for exp in exposures]

# ==== 时间帧构建 ====
N = len(field_seq)
n_frames = 100
deltas = np.array([(t - times[0]).to('day').value for t in times])

cut_ratio = 0.6
cut_frame = 70
cut_N = int(N * cut_ratio)

log_raw = np.unique(np.logspace(0, np.log10(cut_N), cut_frame, dtype=int)).tolist()
log_bins = list(sorted(set(log_raw)))

# 线性补充
day_start = deltas[log_bins[-1]]
day_end = deltas[-1]
linear_days_per_frame = 5
remaining_frames = int(np.ceil((day_end - day_start) / linear_days_per_frame))
day_steps = np.linspace(day_start, day_end, remaining_frames + 1)

for d in day_steps[1:]:
    idx = np.searchsorted(deltas, d)
    log_bins.append(min(idx, N - 1))
log_bins[-1] = N - 1
n_frames = len(log_bins)

# ==== 初始化图形 ====
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(111, projection="aitoff")
plt.subplots_adjust(top=0.82, bottom=0.22)

# 设置 colorbar（使用空 scatter 占位）
dummy = ax.scatter([], [], c=[], cmap="rainbow", s=5, marker="s", vmin=0, vmax=50)
cbar_ax = fig.add_axes([0.25, 0.10, 0.5, 0.02])
cb = fig.colorbar(dummy, cax=cbar_ax, orientation='horizontal')
cb.set_label("Number of Exposures")

# ==== 曝光次数计数器 ====
field_counts = {fid: 0 for fid in field_ids}

# ==== 更新函数 ====
def update(frame):
    ax.cla()
    ax.grid(True)

    now_idx = log_bins[frame]
    current_fields = field_seq[:now_idx]
    current_times = times[:now_idx]
    last_time = current_times[-1].to_datetime(timezone=local_tz)
    time_str = last_time.strftime("%Y-%m-%d %H:%M")

    ax.set_title(
        f"Cumulative exposures: {now_idx+1} / {N} — {time_str} (local)",
        pad=60
    )

    # 更新计数
    field_counts.update({fid: 0 for fid in field_ids})  # reset
    for fid in current_fields:
        field_counts[fid] += 1

    # 遍历所有天区并画多边形
    for fid, (ra, dec) in field_polygons.items():
        count = field_counts[fid]
        if count > 0:
            plot_aitoff_segments(ax, ra, dec, color=plt.cm.rainbow(min(count, 50) / 50), alpha=0.8)

# ==== 动画 ====
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=150, blit=False, repeat=False)

# ==== 显示或保存 ====
ani.save("./fov_exposure_map.gif", writer="pillow", dpi=150)
plt.show()
