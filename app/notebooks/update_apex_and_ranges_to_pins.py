"""Update ridge.json so that every corner's apex_m is the visual apex (Maps pin's
position on the centerline), and corner ranges are adjusted to (a) always contain
their apex and (b) not overlap neighbors.

Logic:
  1. For each corner, set apex_m = centerline track_dist_m at the closest centerline
     point to the corner's Maps pin.
  2. For T6 and T8, set secondary_apex_m similarly using the T6b / T8b pins.
  3. Compute new start_m / end_m:
     - Try to preserve the corner's original pre-apex and post-apex widths
       (start = new_apex - old_pre_width, end = new_apex + old_post_width)
     - Where this overlaps the previous corner, place boundary at the midpoint
       between this corner's apex and the previous corner's apex
     - Same logic for overlapping the next corner
  4. Verify monotonicity and apex-in-range.

Output: ridge.proposed.json for review.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_LAT, M_LON = 111_132.0, 111_132.0 * np.cos(np.radians(47.255))

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


# Step 1: set new apex_m and secondary_apex_m for each corner from Maps pins
print("=== Step 1: New apex_m from Maps pins ===")
print(f"  {'corner':<5s} {'old apex_m':>11s} {'new apex_m':>11s} {'shift':>7s}")
new_corners = []
for c in track["corners"]:
    cid = c["id"]
    new_c = dict(c)
    if cid in pins:
        new_apex = centerline_td_at_pin(pins[cid]["lat"], pins[cid]["long"])
        new_c["apex_m"] = round(new_apex, 1)
        print(f"  {cid:<5s} {c['apex_m']:>9.1f}m {new_apex:>9.1f}m  {new_apex - c['apex_m']:>+5.1f}m")
    else:
        print(f"  {cid:<5s} {c['apex_m']:>9.1f}m  (no Maps pin; keeping old apex_m)")
    if c.get("secondary_apex_m") is not None:
        sec_pin_id = f"{cid}b"
        if sec_pin_id in pins:
            new_sec = centerline_td_at_pin(pins[sec_pin_id]["lat"], pins[sec_pin_id]["long"])
            new_c["secondary_apex_m"] = round(new_sec, 1)
            print(f"  {cid:<5s} secondary_apex_m: {c['secondary_apex_m']:.1f} -> {new_sec:.1f}m (from {sec_pin_id} pin)")
    new_corners.append(new_c)

# Step 2: adjust corner ranges
print("\n=== Step 2: Adjust corner ranges ===")
# Compute desired ranges based on preserved pre/post-apex widths
desired_ranges = []
for i, (c_old, c_new) in enumerate(zip(track["corners"], new_corners)):
    pre_width = c_old["apex_m"] - c_old["start_m"]
    post_width = c_old["end_m"] - c_old["apex_m"]
    desired_start = c_new["apex_m"] - pre_width
    desired_end = c_new["apex_m"] + post_width
    desired_ranges.append((desired_start, desired_end))

# Resolve overlaps: for each pair of adjacent corners, if they overlap,
# place the boundary at the midpoint between their apexes.
final_starts = [r[0] for r in desired_ranges]
final_ends = [r[1] for r in desired_ranges]
for i in range(len(new_corners) - 1):
    cur_end = final_ends[i]
    next_start = final_starts[i + 1]
    if cur_end > next_start:
        # Overlap: place boundary at midpoint between apexes
        cur_apex = new_corners[i]["apex_m"]
        next_apex = new_corners[i + 1]["apex_m"]
        boundary = (cur_apex + next_apex) / 2
        final_ends[i] = round(boundary, 1)
        final_starts[i + 1] = round(boundary, 1)
        print(f"  Overlap between {new_corners[i]['id']} and {new_corners[i+1]['id']}: "
              f"placed boundary at midpoint of apexes = {boundary:.1f}m")
    elif cur_end > next_start - 0.5:
        # touching, no overlap — keep as-is
        pass

# Apply final ranges
print("\n=== Final corner ranges ===")
print(f"  {'corner':<5s} {'old start':>10s} {'new start':>10s} {'old apex':>10s} {'new apex':>10s} {'old end':>10s} {'new end':>10s}")
for i, (c_old, c_new) in enumerate(zip(track["corners"], new_corners)):
    c_new["start_m"] = round(final_starts[i], 1)
    c_new["end_m"] = round(final_ends[i], 1)
    print(f"  {c_new['id']:<5s} {c_old['start_m']:>8.1f}m {c_new['start_m']:>8.1f}m "
          f"{c_old['apex_m']:>8.1f}m {c_new['apex_m']:>8.1f}m "
          f"{c_old['end_m']:>8.1f}m {c_new['end_m']:>8.1f}m")

# Step 3: validate
print("\n=== Validation ===")
problems = 0
prev_end = -1.0
for c in new_corners:
    if not (c["start_m"] <= c["apex_m"] <= c["end_m"]):
        print(f"  ! {c['id']}: apex_m={c['apex_m']} outside [{c['start_m']}, {c['end_m']}]")
        problems += 1
    if c.get("secondary_apex_m") is not None:
        if not (c["start_m"] <= c["secondary_apex_m"] <= c["end_m"]):
            print(f"  ! {c['id']}: secondary_apex_m={c['secondary_apex_m']} outside [{c['start_m']}, {c['end_m']}]")
            problems += 1
    if c["start_m"] < prev_end - 0.5:
        print(f"  ! {c['id']}: starts {prev_end - c['start_m']:.1f}m before previous corner ends")
        problems += 1
    prev_end = c["end_m"]
if problems == 0:
    print("  All corners valid, no overlaps, all apexes within range.")

# Write
new_track = dict(track)
new_track["corners"] = new_corners
out_path = Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.proposed.json")
out_path.write_text(json.dumps(new_track, indent=2), encoding="utf-8")
print(f"\nWrote proposed track to {out_path}")
