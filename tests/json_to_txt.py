import json
from astropy.coordinates import SkyCoord
import astropy.units as u
from datetime import datetime

# 读取 JSON 文件
with open("../20250713_plan.json", "r") as f:
    data = json.load(f)

# 起止时间
start_time = datetime.fromisoformat("2025-07-13 14:21:12.311")
end_time = datetime.fromisoformat("2025-07-13 17:18:14.189")

output_lines = []

for entry in data:
    entry_time = datetime.fromisoformat(entry["start_utc"])
    if start_time <= entry_time <= end_time:
        field_id = entry["field_id"]
        ra_deg = entry["ra"]
        dec_deg = entry["dec"]
        exptime = int(entry["exptime_s"])

        # 转换 RA/DEC 为字符串格式（保留秒的三位小数）
        coord = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg, frame='icrs')
        ra_str = coord.ra.to_string(unit=u.hour, sep=':', precision=3, pad=True)
        dec_str = coord.dec.to_string(unit=u.deg, sep=':', precision=3, alwayssign=True, pad=True)

        # 格式化输出行
        line = f"SV{field_id}    {ra_str}    {dec_str}    V    1    {exptime}    0"
        output_lines.append(line)

# 写入 TXT 文件
output_filename = "SV_fields_observing_script.txt"
with open(output_filename, "w") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"Saved script to {output_filename}")
