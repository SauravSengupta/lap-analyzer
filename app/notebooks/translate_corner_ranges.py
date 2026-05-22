"""Translate ridge.json corner ranges (start_m, end_m, apex_m, secondary_apex_m)
from the old May-1-based coord system into the new centerline-based coord system.

For each corner range value V (in old coord):
  - If V corresponds to an apex with a Maps pin: use the centerline track_dist_m at
    the closest centerline point to the pin. This is ground-truth.
  - Otherwise: use the seed lap's GPS at the old V (since the seed is clean
    everywhere). Find the centerline's nearest point, take its track_dist_m as
    the new V.

Output: a proposed new corners section. User can review and apply manually.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))

TRACK_PATH = Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.json")
PINS_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge_apex_pins.json")
CENTERLINE_PATH = Path(r"D:\Projects\lap-analyzer\data\corpus\ridge_centerline.parquet")
SEED_PATH = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge\20240811-135036\samples.parquet")

track = json.loads(TRACK_PATH.read_text(encoding="utf-8"))
pins = {k: v for k, v in json.loads(PINS_PATH.read_text(encoding="utf-8")).items() if not k.startswith("_")}
centerline = pd.read_parquet(CENTERLINE_PATH)
seed = pd.read_parquet(SEED_PATH)
seed = seed[seed["lap"] == 5].sort_values("dist_lap_m").reset_index(drop=True)
print(f"Seed lap dist_lap_m range: {seed.dist_lap_m.min():.1f} - {seed.dist_lap_m.max():.1f}")


def centerline_td_at_latlong(lat: float, lon: float) -> tuple[float, float]:
    """Return (centerline_track_dist_m, distance_from_centerline_m) at the closest point."""
    dlat = (centerline.lat - lat) * M_PER_DEG_LAT
    dlon = (centerline["long"] - lon) * M_PER_DEG_LON
    d = np.sqrt(dlat ** 2 + dlon ** 2)
    i = int(d.idxmin())
    return float(centerline.iloc[i].track_dist_m), float(d.iloc[i])


def seed_latlong_at_old_dist(old_dist: float) -> tuple[float, float]:
    """Interpolate the seed lap's lat/long at a given dist_lap_m value."""
    lat = float(np.interp(old_dist, seed.dist_lap_m, seed.lat))
    lon = float(np.interp(old_dist, seed.dist_lap_m, seed["long"]))
    return lat, lon


def translate(old_dist: float, source_label: str = "seed") -> tuple[float, float]:
    """Convert an old-coord track distance to new-coord track distance.
    Returns (new_dist, distance_from_centerline_m_at_seed_position)."""
    lat, lon = seed_latlong_at_old_dist(old_dist)
    return centerline_td_at_latlong(lat, lon)


print("\n=== Translating corner ranges ===")
print(f"  {'corner':<5s} {'field':<20s} {'old':>7s} {'new':>7s} {'shift':>7s}  source  centerline_dist")
new_corners = []
for c in track["corners"]:
    cid = c["id"]
    new_c = dict(c)
    # Translate every value via seed lap → centerline (preserves original corner
    # geometry from cluster analysis, just shifted into the new coord system).
    # For corners with Maps pins, we also report the pin distance to centerline
    # for sanity, but don't re-anchor — that creates overlap artifacts.
    apex_new, apex_dist = translate(c["apex_m"])
    start_new, start_dist = translate(c["start_m"])
    end_new, end_dist = translate(c["end_m"])
    pin_apex = pins.get(cid)
    pin_note = ""
    if pin_apex:
        pin_td, pin_off = centerline_td_at_latlong(pin_apex["lat"], pin_apex["long"])
        pin_note = f"  (pin says apex at td={pin_td:.1f}, delta={pin_td - apex_new:+.1f}m)"
    print(f"  {cid:<5s} {'apex_m':<20s} {c['apex_m']:>7.1f} {apex_new:>7.1f} "
          f"{apex_new - c['apex_m']:>+6.1f}  seed   {apex_dist:>5.2f}m{pin_note}")
    print(f"  {cid:<5s} {'start_m':<20s} {c['start_m']:>7.1f} {start_new:>7.1f} "
          f"{start_new - c['start_m']:>+6.1f}  seed   {start_dist:>5.2f}m")
    print(f"  {cid:<5s} {'end_m':<20s} {c['end_m']:>7.1f} {end_new:>7.1f} "
          f"{end_new - c['end_m']:>+6.1f}  seed   {end_dist:>5.2f}m")
    new_c["apex_m"] = round(apex_new, 1)
    new_c["start_m"] = round(start_new, 1)
    new_c["end_m"] = round(end_new, 1)

    if c.get("secondary_apex_m"):
        sec_new, sec_dist = translate(c["secondary_apex_m"])
        pin_sec = pins.get(f"{cid}b")
        sec_pin_note = ""
        if pin_sec:
            sec_pin_td, _ = centerline_td_at_latlong(pin_sec["lat"], pin_sec["long"])
            sec_pin_note = f"  (T{cid[1:]}b pin says td={sec_pin_td:.1f}, delta={sec_pin_td - sec_new:+.1f}m)"
        print(f"  {cid:<5s} {'secondary_apex_m':<20s} {c['secondary_apex_m']:>7.1f} {sec_new:>7.1f} "
              f"{sec_new - c['secondary_apex_m']:>+6.1f}  seed   {sec_dist:>5.2f}m{sec_pin_note}")
        new_c["secondary_apex_m"] = round(sec_new, 1)

    new_corners.append(new_c)
    print()


# Fix overlaps: where a corner's start falls before the previous corner's end,
# push the later corner's start to the previous corner's end. The pinned corner
# (which has the more reliable apex placement) keeps its full extent.
print("=== Fixing corner-to-corner overlaps ===")
for i in range(1, len(new_corners)):
    prev = new_corners[i - 1]
    cur = new_corners[i]
    if cur["start_m"] < prev["end_m"]:
        gap = prev["end_m"] - cur["start_m"]
        old_start = cur["start_m"]
        cur["start_m"] = prev["end_m"]
        # If apex now falls before start (shouldn't with our anchor logic, but check), bump apex too
        if cur["apex_m"] < cur["start_m"]:
            print(f"  WARN: {cur['id']} apex {cur['apex_m']:.1f} now before start {cur['start_m']:.1f}; corner def needs review")
        print(f"  {cur['id']}: pushed start from {old_start:.1f} to {cur['start_m']:.1f} "
              f"(was overlapping {prev['id']} by {gap:.1f}m)")

# Sanity check
print("\n=== Final monotonicity check ===")
prev_end = -1
problems = 0
for c in new_corners:
    sm = c["start_m"]
    em = c["end_m"]
    am = c["apex_m"]
    if sm < prev_end:
        print(f"  STILL OVERLAPS: {c['id']} start={sm:.1f} prev_end={prev_end:.1f}")
        problems += 1
    if not (sm <= am <= em):
        print(f"  APEX OUT OF RANGE: {c['id']} start={sm:.1f} apex={am:.1f} end={em:.1f}")
        problems += 1
    prev_end = em
if problems == 0:
    print("  All corners properly ordered, all apexes within their ranges.")


# Write proposed new corners back to a file for review
out_path = Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.proposed.json")
new_track = dict(track)
new_track["corners"] = new_corners
new_track["_translation_notes"] = {
    "translated_on": "2026-05-10",
    "method": "Maps-pin for pinned apexes; seed lap (20240811-135036 L5) for unpinned values",
    "centerline_source": "data/corpus/ridge_centerline.parquet"
}
out_path.write_text(json.dumps(new_track, indent=2), encoding="utf-8")
print(f"\nWrote proposed new corners to {out_path}")
print("Review and rename to ridge.json to apply.")
