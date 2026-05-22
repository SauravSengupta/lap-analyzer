"""Refresh ridge.json corner notes: replace May-1-derived quantitative claims
with corpus-median values measured at the new visual_apex_m positions. Preserve
qualitative driving context. Strip historical _* fields.

For each corner, computes from clean transits:
  - apex_speed_mph (speed at apex_m, i.e. visual apex now)
  - max_lat_g
  - latg_peak_offset_m (relative to visual apex)
  - apex_dist_offset_m (speed-min relative to visual apex; same as in transits)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRACK_PATH = Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.json")
ROOT = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")

track = json.loads(TRACK_PATH.read_text(encoding="utf-8"))

# Pull all clean-lap transits
dfs = []
for sd in ROOT.iterdir():
    if sd.is_dir() and not sd.name.startswith("_"):
        cp = sd / "corners.parquet"
        if cp.exists():
            dfs.append(pd.read_parquet(cp))
df = pd.concat(dfs, ignore_index=True)
print(f"{len(df)} transits across {df.session_id.nunique()} sessions")

# Optionally restrict to fast laps (top decile) for "typical fast lap" values
laps_dfs = [pd.read_csv(sd / "laps.csv") for sd in ROOT.iterdir() if sd.is_dir() and not sd.name.startswith("_") and (sd / "laps.csv").exists()]
laps = pd.concat(laps_dfs, ignore_index=True)
clean = laps[laps.is_clean.astype(bool)]
threshold = clean.lap_time_s.quantile(0.10)
fast_keys = set(zip(clean[clean.lap_time_s <= threshold].session_id, clean[clean.lap_time_s <= threshold].lap.astype(int)))
df_fast = df[df.apply(lambda r: (r.session_id, int(r.lap)) in fast_keys, axis=1)]
print(f"  Top-decile pool: {len(df_fast)} transits across {df_fast.session_id.nunique()} sessions (threshold {threshold:.2f}s)")

# Compute medians per corner (use top-decile pool for "typical fast lap" feel)
stats = {}
for cid in df_fast.corner_id.unique():
    sub = df_fast[df_fast.corner_id == cid]
    stats[cid] = {
        "apex_speed_mph": float(sub.apex_speed_mph.median()),
        "max_lat_g": float(sub.max_lat_g.median()),
        "latg_peak_offset_m": float(sub.latg_peak_offset_m.median()),
        "apex_dist_offset_m": float(sub.apex_dist_offset_m.median()),
        "min_speed_mph": float(sub.min_speed_mph.median()),
        "max_speed_mph": float(sub.max_speed_mph.median()),
        "entry_speed_mph": float(sub.entry_speed_mph.median()),
        "exit_speed_mph": float(sub.exit_speed_mph.median()),
    }
    if "secondary_apex_speed_mph" in sub.columns:
        s = sub.secondary_apex_speed_mph.dropna()
        if len(s) > 0:
            stats[cid]["secondary_apex_speed_mph"] = float(s.median())

# Refresh notes per corner. New format: "<driving context>. Visual apex {apex_m}m,
# top-decile median speed {S} mph, max lat-G {G}. <cross-ref to lat-G peak / speed-min if useful>."
qualitative = {
    "T1": "Fast L sweeper. No brake / minimal lift before turn-in; brake comes for T2.",
    "T2": "Slow L; brake-on starts around 30-40m before this apex for the T1+T2 sequence.",
    "T3": "Quick R coming out of T2.",
    "T4": "T4+T5 driven as one long left — hold the wheel, car takes a set.",
    "T5": "See T4. Brief R lat-G shift after exit as the car straightens before T6 setup.",
    "T6": "The Carousel — long L sweeper. Heavily line-dependent: double-apex, constant-radius, or decreasing-radius. Two pinned apexes: apex_m is the heavy-load / slowest-speed point (trail-in apex); secondary_apex_m is the racing-line / second-of-double-apex.",
    "T7": "Slight R going down the hill at near-WOT. Visual apex falls deep into the corner.",
    "T8": "Long R with multiple lines. Two pinned apexes: apex_m is the heavy-load / slowest-speed (trail-in); secondary_apex_m is the racing-line apex (T8b).",
    "T9": "WOT R going up the hill.",
    "T10": "WOT continuing up the hill; speed climbs through toward T11 brake zone.",
    "T11": "Hard L hairpin.",
    "T12": "R after T11; entry to back straight.",
    "T13": "Slowest corner of the lap. Start of the back-section chicane (T13-T14-T15).",
    "T14": "Tight R; second corner of the back chicane.",
    "T15": "L; final corner of the back chicane. Car continues to rotate past the visual apex.",
    "T16": "Last corner; opens onto the front straight back to start/finish. Light corner, brief lat-G load.",
}


def build_note(cid: str, c: dict, s: dict) -> str:
    parts = [qualitative.get(cid, "")]
    apex_m = c["apex_m"]
    speed = s["apex_speed_mph"]
    g = s["max_lat_g"]
    parts.append(f"Visual apex at {apex_m:.0f}m; top-decile median speed at apex {speed:.0f} mph, max lat-G {g:.2f}.")
    # Cross-references where they add signal
    latg_off = s["latg_peak_offset_m"]
    sm_off = s["apex_dist_offset_m"]
    if abs(latg_off) >= 5 or abs(sm_off) >= 5:
        parts.append(
            f"Speed-min offset from visual apex: {sm_off:+.0f}m; lat-G peak offset: {latg_off:+.0f}m."
        )
    if "secondary_apex_speed_mph" in s and c.get("secondary_apex_m") is not None:
        parts.append(f"Secondary apex at {c['secondary_apex_m']:.0f}m, median speed {s['secondary_apex_speed_mph']:.0f} mph.")
    return " ".join(p for p in parts if p)


# Rewrite corners
print("\n=== New corner notes (top-decile medians) ===")
new_corners = []
for c in track["corners"]:
    cid = c["id"]
    new_c = dict(c)
    if cid in stats:
        new_c["notes"] = build_note(cid, c, stats[cid])
    print(f"\n  {cid}:")
    print(f"    {new_c['notes']}")
    new_corners.append(new_c)

# Strip historical fields
new_track = {
    "track_id": track["track_id"],
    "name": track["name"],
    "configuration": track["configuration"],
    "official_length_m": track["official_length_m"],
    "direction": track["direction"],
    "start_finish": track["start_finish"],
    "lap_length_internal_m": track["lap_length_internal_m"],
    "reference_lap": track["reference_lap"],
    "calibration_anchors": track["calibration_anchors"],
    "corners": new_corners,
}

out_path = Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.proposed.json")
out_path.write_text(json.dumps(new_track, indent=2), encoding="utf-8")
print(f"\nWrote refreshed track to {out_path}")
print("Historical fields removed: _seed_lap_for_centerline_build, _previous_reference_lap, _translation_notes, _calibration_anchors_notes")
