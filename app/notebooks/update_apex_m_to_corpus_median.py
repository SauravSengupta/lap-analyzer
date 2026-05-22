"""Update ridge.json apex_m for each corner to the corpus-median speed-min position
within that corner's range. Replaces May-1-derived apex_m values that reflected one
specific lap's speed-min point.

For T6 and T8: secondary_apex_m is updated to the centerline position of the Maps pin
(T6b, T8b) since the secondary apex is by definition the racing-line / visual apex.

Outputs a proposed new ridge.json; user reviews before applying.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_LAT, M_LON = 111_132.0, 111_132.0 * np.cos(np.radians(47.255))

ROOT = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
TRACK_PATH = Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.json")
PINS_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge_apex_pins.json")
CENTERLINE_PATH = Path(r"D:\Projects\lap-analyzer\data\corpus\ridge_centerline.parquet")

track = json.loads(TRACK_PATH.read_text(encoding="utf-8"))
pins = {k: v for k, v in json.loads(PINS_PATH.read_text(encoding="utf-8")).items() if not k.startswith("_")}
centerline = pd.read_parquet(CENTERLINE_PATH)


def centerline_td_at_pin(pin_lat: float, pin_lon: float) -> float:
    dlat = (centerline.lat - pin_lat) * M_LAT
    dlon = (centerline["long"] - pin_lon) * M_LON
    d = np.sqrt(dlat ** 2 + dlon ** 2)
    return float(centerline.iloc[int(d.idxmin())].track_dist_m)


# Pull all clean-lap transits, compute corpus median min_speed_dist per corner
print("Loading corner transits...")
corners_dfs = []
for sd in [d for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")]:
    cp = sd / "corners.parquet"
    if cp.exists():
        corners_dfs.append(pd.read_parquet(cp))
df = pd.concat(corners_dfs, ignore_index=True)
print(f"  {len(df)} transits across {df.session_id.nunique()} sessions, {df.lap.nunique()} unique lap numbers")


# For each corner, compute median speed-min position. Also report variance.
print(f"\n=== Proposed apex_m updates ===")
print(f"  {'corner':<5s} {'old apex_m':>11s} {'median min_speed_dist':>21s} {'p25':>7s} {'p75':>7s}  {'shift':>7s}  {'corner range':<22s}")
new_corners = []
for c in track["corners"]:
    cid = c["id"]
    sub = df[df.corner_id == cid]
    if len(sub) == 0:
        new_corners.append(c)
        continue
    median_sm = float(sub.min_speed_dist_m.median())
    p25 = float(sub.min_speed_dist_m.quantile(0.25))
    p75 = float(sub.min_speed_dist_m.quantile(0.75))
    # Clip to corner range — speed-min should always be inside the corner
    if not (c["start_m"] <= median_sm <= c["end_m"]):
        print(f"  {cid:<5s}  WARNING: median_sm={median_sm:.1f} outside corner range [{c['start_m']}, {c['end_m']}]")
        median_sm = max(c["start_m"], min(c["end_m"], median_sm))
    new_c = dict(c)
    old_apex = c["apex_m"]
    new_c["apex_m"] = round(median_sm, 1)
    print(f"  {cid:<5s} {old_apex:>9.1f}m {median_sm:>19.1f}m {p25:>5.1f}m {p75:>5.1f}m  {median_sm - old_apex:>+5.1f}m  [{c['start_m']}, {c['end_m']}]")

    # Update secondary_apex_m for T6 and T8 using their Maps pin's centerline position
    if c.get("secondary_apex_m") is not None:
        if cid == "T6" and "T6b" in pins:
            new_sec = centerline_td_at_pin(pins["T6b"]["lat"], pins["T6b"]["long"])
        elif cid == "T8" and "T8b" in pins:
            new_sec = centerline_td_at_pin(pins["T8b"]["lat"], pins["T8b"]["long"])
        else:
            new_sec = c["secondary_apex_m"]
        new_c["secondary_apex_m"] = round(new_sec, 1)
        if new_sec != c["secondary_apex_m"]:
            print(f"  {cid:<5s} secondary_apex_m: {c['secondary_apex_m']:.1f}m -> {new_sec:.1f}m (from {cid}b Maps pin)")

    new_corners.append(new_c)

# Sanity: every apex_m within its corner range, ordered
print("\n=== Sanity ===")
for c in new_corners:
    if not (c["start_m"] <= c["apex_m"] <= c["end_m"]):
        print(f"  ! {c['id']}: apex_m={c['apex_m']} outside [{c['start_m']}, {c['end_m']}]")
    if c.get("secondary_apex_m") is not None:
        if not (c["start_m"] <= c["secondary_apex_m"] <= c["end_m"]):
            print(f"  ! {c['id']}: secondary_apex_m={c['secondary_apex_m']} outside [{c['start_m']}, {c['end_m']}]")

# Write
new_track = dict(track)
new_track["corners"] = new_corners
out_path = Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.proposed.json")
out_path.write_text(json.dumps(new_track, indent=2), encoding="utf-8")
print(f"\nWrote proposed track to {out_path}")
