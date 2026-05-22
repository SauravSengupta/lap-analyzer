"""Validate T7, T8 (secondary apex), T9 as GPS calibration anchors.

Reference coords from 20260501-101355 L3, provided by the user from RaceRender
video stills at the visual apex.

Three views:
  RAW: line variation + drift, against raw lat/long (the original validation rule;
       8m threshold is for this view).
  DRIFT-CORRECTED: applies the per-lap GPS drift estimate from the existing T2/T13/T14
       anchors before measuring spread. Shows residual line variation alone.
  Time-in-corner distribution: to test whether slow-line traffic laps are dragging
       the median.

The previous version of this script searched within the corner via `dist_lap_m`,
which has non-uniform OBD integration error and on T7-T9 (~1700-2200m into the lap)
puts the search window on a different chunk of physical track on every lap. This
version searches by `track_dist_m`, the GPS-aligned canonical coord.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))
VALIDATION_THRESHOLD_M = 8.0

CANDIDATES = {
    "T7":  {"lat": 47.25677391, "long": -123.18883709, "start": 1644.0, "end": 1745.0, "apex_mph": 90, "apex_m": 1694.0},
    "T8b": {"lat": 47.25624614, "long": -123.18643640, "start": 1900.0, "end": 1990.0, "apex_mph": 60, "apex_m": 1941.0},
    "T9":  {"lat": 47.25555535, "long": -123.18834718, "start": 2061.0, "end": 2161.0, "apex_mph": 89, "apex_m": 2111.0},
}

ROOT = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
NOTES_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge.json")
notes = json.loads(NOTES_PATH.read_text())
gps_unreliable = {sid for sid, m in notes.items() if m.get("flag") == "gps_unreliable"}
excluded = {sid for sid, m in notes.items() if "exclude" in m}

laps = pd.concat([pd.read_csv(p) for p in ROOT.rglob("laps.csv")], ignore_index=True)
clean = laps[laps.is_clean.astype(bool)]
clean = clean[~clean.session_id.isin(excluded | gps_unreliable)]
threshold = clean.lap_time_s.quantile(0.10)
fast = clean[clean.lap_time_s <= threshold][["session_id", "lap", "lap_time_s"]].copy()
extra = pd.DataFrame([{"session_id": "20260501-101355", "lap": 3, "lap_time_s": 124.4}])
all_laps = pd.concat([fast, extra]).drop_duplicates(["session_id", "lap"]).reset_index(drop=True)
print(f"Top-decile pool: {len(all_laps)} laps  (threshold {threshold:.2f}s)")


def closest_in_track_range(samples: pd.DataFrame, lap: int, ref: dict, use_corrected: bool):
    """Find sample inside [ref.start, ref.end] of track_dist_m closest to ref lat/long.
    If use_corrected, lat/long for the closeness comparison is drift-corrected.
    """
    seg = samples[
        (samples["lap"] == lap)
        & (samples["track_dist_m"] >= ref["start"])
        & (samples["track_dist_m"] <= ref["end"])
    ]
    if len(seg) == 0:
        return None
    if use_corrected:
        lat_c = seg["lat"] - seg["gps_drift_lat_m"] / M_PER_DEG_LAT
        lon_c = seg["long"] - seg["gps_drift_lon_m"] / M_PER_DEG_LON
    else:
        lat_c = seg["lat"]
        lon_c = seg["long"]
    dlat_m = (lat_c - ref["lat"]) * M_PER_DEG_LAT
    dlon_m = (lon_c - ref["long"]) * M_PER_DEG_LON
    d2 = dlat_m ** 2 + dlon_m ** 2
    return float(np.sqrt(d2.min())) if len(d2) else None


# Per-transit time-in-corner from corners.parquet — to check the slow-lap hypothesis
ti = []
for sid in all_laps.session_id.unique():
    cpath = ROOT / sid / "corners.parquet"
    if cpath.exists():
        cdf = pd.read_parquet(cpath)
        ti.append(cdf[cdf.corner_id.isin(["T7", "T8", "T9"])]
                  [["session_id", "lap", "corner_id", "time_in_corner_s", "min_speed_mph", "track_dist_offset_max_m"]])
ti_df = pd.concat(ti, ignore_index=True) if ti else pd.DataFrame()


# Collect distances per lap per candidate, raw and drift-corrected
records = []
for _, r in all_laps.iterrows():
    sid, ln = r.session_id, int(r.lap)
    s = pd.read_parquet(ROOT / sid / "samples.parquet")
    rec = {"session_id": sid, "lap": ln, "lap_time_s": r.lap_time_s}
    for cname, ref in CANDIDATES.items():
        rec[f"{cname}_raw_m"] = closest_in_track_range(s, ln, ref, use_corrected=False)
        rec[f"{cname}_cor_m"] = closest_in_track_range(s, ln, ref, use_corrected=True)
    records.append(rec)

df = pd.DataFrame(records)


def report(label: str, suffix: str) -> None:
    print(f"\n=== {label} (n={len(df)}) ===")
    print(f"  {'corner':<5s} {'apex mph':>9s}  {'median':>9s} {'p75':>9s} {'p90':>9s} {'max':>9s}")
    for cname, ref in CANDIDATES.items():
        col = df[f"{cname}_{suffix}"].dropna()
        if len(col) == 0:
            print(f"  {cname:<5s}  no data"); continue
        print(f"  {cname:<5s} {ref['apex_mph']:>7d}    {col.median():>7.2f}m {col.quantile(0.75):>7.2f}m "
              f"{col.quantile(0.9):>7.2f}m {col.max():>7.2f}m")


report("RAW lat/long (the original 8m-threshold rule)", "raw_m")
report("DRIFT-CORRECTED (T2/T13/T14 anchors removed first; shows residual line variation)", "cor_m")

print("\n--- Reference: established anchors (raw view, the apples-to-apples comparison) ---")
print("  T2  60 mph -> 7.90m median   ACCEPT")
print("  T13 27 mph -> 7.50m median   ACCEPT")
print("  T14 37 mph -> 5.27m median   ACCEPT")
print("  T1  89 mph -> 28.06m median  REJECT")
print("  T5  77 mph -> 15.17m median  REJECT")

# Time-in-corner distribution to test the "slow lap dragging median" hypothesis
print("\n=== Time-in-corner distribution (top-decile laps only) ===")
if not ti_df.empty:
    pool_keys = set(zip(all_laps.session_id, all_laps.lap.astype(int)))
    ti_pool = ti_df[ti_df.apply(lambda r: (r.session_id, int(r.lap)) in pool_keys, axis=1)].copy()
    for cid in ["T7", "T8", "T9"]:
        sub = ti_pool[ti_pool.corner_id == cid]
        print(f"  {cid}: n={len(sub)}  time_in_corner_s median {sub.time_in_corner_s.median():.2f}  "
              f"min {sub.time_in_corner_s.min():.2f}  max {sub.time_in_corner_s.max():.2f}  "
              f"std {sub.time_in_corner_s.std():.3f}  "
              f"min_speed_mph median {sub.min_speed_mph.median():.1f}  "
              f"min {sub.min_speed_mph.min():.1f}  max {sub.min_speed_mph.max():.1f}")

# Per-lap detail for T7 to inspect outliers
print("\n=== Per-lap detail for T7 (sorted by raw spread, descending) ===")
print(f"  {'session_id':<22s} {'lap':>3s} {'lap_t':>6s} {'raw_m':>7s} {'cor_m':>7s} {'tic_T7':>7s} {'minS_T7':>8s} {'glitch_T7':>10s}")
detail = df[["session_id", "lap", "lap_time_s", "T7_raw_m", "T7_cor_m"]].copy()
detail = detail.merge(
    ti_df[ti_df.corner_id == "T7"][["session_id", "lap", "time_in_corner_s", "min_speed_mph", "track_dist_offset_max_m"]]
        .rename(columns={"time_in_corner_s": "tic", "min_speed_mph": "minS", "track_dist_offset_max_m": "glitch"}),
    on=["session_id", "lap"], how="left",
)
for _, r in detail.sort_values("T7_raw_m", ascending=False).iterrows():
    print(f"  {r.session_id:<22s} {int(r.lap):>3d} {r.lap_time_s:>6.2f} "
          f"{r.T7_raw_m if pd.notnull(r.T7_raw_m) else float('nan'):>7.2f} "
          f"{r.T7_cor_m if pd.notnull(r.T7_cor_m) else float('nan'):>7.2f} "
          f"{r.tic if pd.notnull(r.tic) else float('nan'):>7.2f} "
          f"{r.minS if pd.notnull(r.minS) else float('nan'):>8.1f} "
          f"{r.glitch if pd.notnull(r.glitch) else float('nan'):>10.1f}")
