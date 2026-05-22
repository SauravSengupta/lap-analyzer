"""Compare anchor-based vs centerline-based drift estimates per lap.

Loads both estimates from samples.parquet (dual-write phase) and produces
per-lap and aggregate diagnostics:

  - Magnitude difference between the two estimates
  - Per-axis correlation
  - Distribution of differences (clean laps vs anchor-disagreement laps)
  - Maps-pin-based validation: which method gives smaller distance to pin
    after correction
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))

ROOT = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
PINS_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge_apex_pins.json")
NOTES_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge.json")

pins = {k: v for k, v in json.loads(PINS_PATH.read_text(encoding="utf-8")).items() if not k.startswith("_")}
notes = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
excluded = {sid for sid, m in notes.items() if "exclude" in m}

# ---------- Pull per-lap drift estimates from all sessions ----------
records = []
for sid in sorted(d.name for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")):
    if sid in excluded:
        continue
    sp = ROOT / sid / "samples.parquet"
    lp = ROOT / sid / "laps.csv"
    if not sp.exists() or not lp.exists():
        continue
    s = pd.read_parquet(sp, columns=[
        "lap", "gps_drift_lat_m", "gps_drift_lon_m", "gps_drift_disagreement_m",
        "gps_drift_lat_m_centerline", "gps_drift_lon_m_centerline",
        "gps_drift_region_disagreement_m", "gps_drift_outliers_filtered",
        "gps_drift_n_samples_centerline",
    ])
    laps = pd.read_csv(lp)
    clean = laps[laps.is_clean.astype(bool)]
    for _, lr in clean.iterrows():
        ln = int(lr.lap)
        sub = s[s["lap"] == ln]
        if len(sub) == 0:
            continue
        first = sub.iloc[0]
        records.append({
            "session_id": sid, "lap": ln, "lap_time_s": float(lr.lap_time_s),
            "anchor_lat": float(first.gps_drift_lat_m),
            "anchor_lon": float(first.gps_drift_lon_m),
            "anchor_disagreement": float(first.gps_drift_disagreement_m),
            "cl_lat": float(first.gps_drift_lat_m_centerline),
            "cl_lon": float(first.gps_drift_lon_m_centerline),
            "cl_region_disagreement": float(first.gps_drift_region_disagreement_m),
            "cl_outliers_filtered": int(first.gps_drift_outliers_filtered),
            "cl_n_samples": int(first.gps_drift_n_samples_centerline),
        })

df = pd.DataFrame(records)
df["anchor_mag"] = np.sqrt(df.anchor_lat ** 2 + df.anchor_lon ** 2)
df["cl_mag"] = np.sqrt(df.cl_lat ** 2 + df.cl_lon ** 2)
df["lat_diff"] = df.anchor_lat - df.cl_lat
df["lon_diff"] = df.anchor_lon - df.cl_lon
df["delta_mag"] = np.sqrt(df.lat_diff ** 2 + df.lon_diff ** 2)

print(f"Total clean laps: {len(df)}\n")

# ---------- Aggregate stats ----------
print("=== Drift estimate magnitudes ===")
print(f"  {'method':<12s}  {'median':>8s}  {'p75':>8s}  {'p90':>8s}  {'max':>8s}")
print(f"  {'anchor':<12s}  {df.anchor_mag.median():>6.2f}m  {df.anchor_mag.quantile(0.75):>6.2f}m  "
      f"{df.anchor_mag.quantile(0.90):>6.2f}m  {df.anchor_mag.max():>6.2f}m")
print(f"  {'centerline':<12s}  {df.cl_mag.median():>6.2f}m  {df.cl_mag.quantile(0.75):>6.2f}m  "
      f"{df.cl_mag.quantile(0.90):>6.2f}m  {df.cl_mag.max():>6.2f}m")

print("\n=== Per-lap difference between methods (anchor minus centerline, in meters) ===")
print(f"  median {df.delta_mag.median():.2f}m  p75 {df.delta_mag.quantile(0.75):.2f}m  "
      f"p90 {df.delta_mag.quantile(0.90):.2f}m  p95 {df.delta_mag.quantile(0.95):.2f}m  max {df.delta_mag.max():.2f}m")

# Correlation
corr_lat = float(np.corrcoef(df.anchor_lat, df.cl_lat)[0, 1])
corr_lon = float(np.corrcoef(df.anchor_lon, df.cl_lon)[0, 1])
print(f"\n  Per-axis Pearson correlation:  lat r={corr_lat:.4f}  lon r={corr_lon:.4f}")

# Sliced by anchor disagreement (the existing diagnostic)
print("\n=== Stratified by anchor_disagreement (the existing 'bad lap' signal) ===")
for label, mask in [
    ("low anchor-disagreement (<6m, well-behaved)", df.anchor_disagreement < 6),
    ("medium (6-12m)",                              (df.anchor_disagreement >= 6) & (df.anchor_disagreement < 12)),
    ("high (>12m, currently flagged unreliable)",   df.anchor_disagreement >= 12),
]:
    sub = df[mask]
    print(f"  {label}: n={len(sub)}")
    if len(sub) == 0: continue
    print(f"    delta median {sub.delta_mag.median():.2f}m  p90 {sub.delta_mag.quantile(0.90):.2f}m  max {sub.delta_mag.max():.2f}m")
    print(f"    region_disagreement on these laps: median {sub.cl_region_disagreement.median():.2f}m  p90 {sub.cl_region_disagreement.quantile(0.90):.2f}m")

# ---------- Maps-pin validation ----------
print("\n=== Maps-pin validation: post-correction distance to pin ===")
print("(Smaller is better. For each method, distance = closest-sample-to-pin AFTER applying that drift)")

pin_results = []
for sid in df.session_id.unique():
    sp = ROOT / sid / "samples.parquet"
    s = pd.read_parquet(sp, columns=[
        "lap", "lat", "long",
        "gps_drift_lat_m", "gps_drift_lon_m",
        "gps_drift_lat_m_centerline", "gps_drift_lon_m_centerline",
    ])
    for ln in df[df.session_id == sid].lap:
        ls = s[s["lap"] == ln]
        if len(ls) == 0: continue
        # Anchor-corrected
        lat_a = ls["lat"].values - ls["gps_drift_lat_m"].iloc[0] / M_PER_DEG_LAT
        lon_a = ls["long"].values - ls["gps_drift_lon_m"].iloc[0] / M_PER_DEG_LON
        # Centerline-corrected
        lat_c = ls["lat"].values - ls["gps_drift_lat_m_centerline"].iloc[0] / M_PER_DEG_LAT
        lon_c = ls["long"].values - ls["gps_drift_lon_m_centerline"].iloc[0] / M_PER_DEG_LON
        for pid, p in pins.items():
            d_a = np.min(np.sqrt(((lat_a - p["lat"]) * M_PER_DEG_LAT) ** 2 + ((lon_a - p["long"]) * M_PER_DEG_LON) ** 2))
            d_c = np.min(np.sqrt(((lat_c - p["lat"]) * M_PER_DEG_LAT) ** 2 + ((lon_c - p["long"]) * M_PER_DEG_LON) ** 2))
            pin_results.append({"session_id": sid, "lap": int(ln), "pin": pid, "anchor_d": float(d_a), "cl_d": float(d_c)})

pin_df = pd.DataFrame(pin_results)
print(f"  {'pin':<5s}  {'anchor median':>14s}  {'cl median':>11s}  {'anchor p90':>11s}  {'cl p90':>9s}  {'cl wins':>9s}")
for pid in pins.keys():
    sub = pin_df[pin_df.pin == pid]
    if len(sub) == 0: continue
    cl_wins = (sub.cl_d < sub.anchor_d).mean()
    print(f"  {pid:<5s}  {sub.anchor_d.median():>12.2f}m  {sub.cl_d.median():>9.2f}m  "
          f"{sub.anchor_d.quantile(0.9):>9.2f}m  {sub.cl_d.quantile(0.9):>7.2f}m  {100*cl_wins:>7.1f}%")

# ---------- Outliers: laps where the methods most disagree ----------
print("\n=== Top 10 laps where methods disagree most (delta_mag descending) ===")
top = df.nlargest(10, "delta_mag")[
    ["session_id", "lap", "lap_time_s", "anchor_mag", "cl_mag", "delta_mag",
     "anchor_disagreement", "cl_region_disagreement", "cl_outliers_filtered", "cl_n_samples"]
]
print(top.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
