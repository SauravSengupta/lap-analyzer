"""For each Maps-pinned apex, find the closest sample on the May-1 reference lap
(across the entire lap, no corner range constraint).

Output:
  - Closest distance: tells us if the reference lap is glitched at this corner.
    Clean = ~3m. Glitched = 30-50m+. T12 was 2m, T7 was 50m.
  - track_dist_m of the closest sample: tells us which corner the pin is actually
    at, so we can cross-check the user's label against the corner definition.

Then: for the corners flagged as glitched on the reference lap, scan the corpus
for clean laps (where some other lap's GPS does pass close to the Maps pin).
Those would be splice candidates.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))

PINS_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge_apex_pins.json")
TRACK_PATH = Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.json")
SESSIONS = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
NOTES_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge.json")

pins = {k: v for k, v in json.loads(PINS_PATH.read_text(encoding="utf-8")).items() if not k.startswith("_")}
track = json.loads(TRACK_PATH.read_text(encoding="utf-8"))
corner_ranges = {c["id"]: (c["start_m"], c["end_m"], c["apex_m"]) for c in track["corners"]}


def m_dist(lat1, lon1, lat2, lon2):
    return np.sqrt(((lat2 - lat1) * M_PER_DEG_LAT) ** 2
                 + ((lon2 - lon1) * M_PER_DEG_LON) ** 2)


def closest_in_lap(samples: pd.DataFrame, lap: int, pin_lat: float, pin_lon: float):
    sub = samples[samples["lap"] == lap]
    if len(sub) == 0:
        return None, None, None
    d = m_dist(sub["lat"].values, sub["long"].values, pin_lat, pin_lon)
    idx = int(d.argmin())
    s = sub.iloc[idx]
    return float(d[idx]), float(s["track_dist_m"]), s


def which_corner(track_dist_m: float) -> str:
    for cid, (s, e, _) in corner_ranges.items():
        if s <= track_dist_m <= e:
            return cid
    return "(between corners)"


# Step 1: reference-lap check
ref = pd.read_parquet(SESSIONS / "20260501-101355" / "samples.parquet")
print("=== May-1 reference lap (L3) closest sample to each Maps pin ===")
print(f"  {'pin':<5s} {'pin (lat, long)':<28s} "
      f"{'dist':>7s}  {'ref track_dist_m':>16s}  {'lands in':<8s} {'expected':<8s}  status")
results = {}
for pin_id, pin in pins.items():
    dist, td, _ = closest_in_lap(ref, 3, pin["lat"], pin["long"])
    if dist is None:
        print(f"  {pin_id}: no data"); continue
    lands_in = which_corner(td)
    status = "CLEAN" if dist < 10 else ("BORDERLINE" if dist < 25 else "GLITCHED")
    label_match = "yes" if lands_in == pin_id else "no!"
    print(f"  {pin_id:<5s} ({pin['lat']:.5f}, {pin['long']:.5f})  "
          f"{dist:>5.2f}m  {td:>14.1f}m   {lands_in:<8s} {pin_id:<8s} {label_match}  {status}")
    results[pin_id] = {"ref_dist": dist, "ref_track_dist_m": td, "lands_in": lands_in, "status": status}


# Step 2: for glitched pins, scan the corpus for laps with clean GPS at that pin
print("\n=== Splice candidate scan ===")
print("For each glitched pin, find the laps with the closest recorded GPS to truth.")
print("Laps with <10m distance to the pin = clean recording = splice candidates.\n")

notes = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
excluded = {sid for sid, m in notes.items() if "exclude" in m}

laps_index = pd.concat(
    [pd.read_csv(p) for p in SESSIONS.rglob("laps.csv")], ignore_index=True
)
clean_laps = laps_index[laps_index.is_clean.astype(bool) & ~laps_index.session_id.isin(excluded)]
print(f"Scanning {len(clean_laps)} clean laps from non-excluded sessions...")

glitched_pins = [pid for pid, r in results.items() if r["status"] == "GLITCHED"]
print(f"Glitched pins to scan: {glitched_pins}\n")


def scan_pin(pin_id: str, pin_lat: float, pin_lon: float) -> pd.DataFrame:
    rows = []
    for sid in clean_laps.session_id.unique():
        sp = SESSIONS / sid / "samples.parquet"
        if not sp.exists():
            continue
        s = pd.read_parquet(sp)
        for ln in clean_laps[clean_laps.session_id == sid].lap.unique():
            d, td, _ = closest_in_lap(s, int(ln), pin_lat, pin_lon)
            if d is None:
                continue
            rows.append({"session_id": sid, "lap": int(ln), "closest_m": d, "closest_td_m": td})
    return pd.DataFrame(rows).sort_values("closest_m")


for pid in glitched_pins:
    p = pins[pid]
    df = scan_pin(pid, p["lat"], p["long"])
    print(f"--- {pid} (Maps pin {p['lat']}, {p['long']}) ---")
    n_clean = (df.closest_m < 10).sum()
    n_borderline = ((df.closest_m >= 10) & (df.closest_m < 25)).sum()
    n_glitched = (df.closest_m >= 25).sum()
    print(f"  Total laps: {len(df)}  |  clean (<10m): {n_clean}  borderline (10-25m): {n_borderline}  glitched (>=25m): {n_glitched}")
    print(f"  Top 10 closest laps:")
    for _, r in df.head(10).iterrows():
        print(f"    {r.session_id} lap {int(r.lap):>2d}: {r.closest_m:>5.2f}m at track_dist_m={r.closest_td_m:.0f}")
    print(f"  Distribution: median {df.closest_m.median():.1f}m  p10 {df.closest_m.quantile(0.1):.1f}m  "
          f"p25 {df.closest_m.quantile(0.25):.1f}m  p75 {df.closest_m.quantile(0.75):.1f}m  max {df.closest_m.max():.1f}m\n")
