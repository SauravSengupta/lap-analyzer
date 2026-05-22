"""Cluster ridge_candidates.csv into corner clusters using density peaks.

Filters to fast 2025/2026 laps for cleaner reference apex speeds.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import find_peaks

CSV = Path(__file__).resolve().parents[2] / "data" / "corpus" / "ridge_candidates.csv"
df = pd.read_csv(CSV)
df["date"] = pd.to_datetime(df["date"])

# Filter: 2025+ AND lap_time in the fastest 25% of those laps (drop traffic / slow laps)
df = df[df["date"] >= "2025-01-01"].copy()
lap_times = df.groupby(["session_id", "lap"])["lap_time_s"].first()
fast_cutoff = lap_times.quantile(0.25)  # 25th percentile = fastest quarter
print(f"After 2025+ filter: {df.groupby(['session_id','lap']).ngroups} laps from {df['session_id'].nunique()} sessions")
print(f"Lap-time distribution: min={lap_times.min():.2f}  p25={lap_times.quantile(0.25):.2f}  median={lap_times.median():.2f}  max={lap_times.max():.2f}")
print(f"Keeping laps under {fast_cutoff:.2f} s")

fast_keys = set(lap_times[lap_times <= fast_cutoff].index)
df = df[df.set_index(["session_id", "lap"]).index.isin(fast_keys)].copy()
print(f"After fast filter: {df.groupby(['session_id','lap']).ngroups} laps  rows={len(df)}")

# Density histogram of peak_dist_m
BIN = 5.0
bins = np.arange(0, 4000 + BIN, BIN)
hist, _ = np.histogram(df["peak_dist_m"], bins=bins)
centers = (bins[:-1] + bins[1:]) / 2
W = 6
hist_smooth = np.convolve(hist, np.ones(W) / W, mode="same")

# Find local maxima with prominence scaled to lap count
N_LAPS = df.groupby(["session_id", "lap"]).ngroups
peaks_idx, _ = find_peaks(hist_smooth, distance=int(60 / BIN), prominence=max(2, N_LAPS * 0.02))

print(f"\nFound {len(peaks_idx)} density peaks at:")
for idx in peaks_idx:
    print(f"  {centers[idx]:6.0f} m   raw_count={hist[idx]:3d}   smoothed={hist_smooth[idx]:5.1f}")

# Voronoi assignment
peak_pos = centers[peaks_idx]
midpoints = (peak_pos[:-1] + peak_pos[1:]) / 2
boundaries = np.concatenate(([-np.inf], midpoints, [np.inf]))
df["cluster"] = np.searchsorted(boundaries, df["peak_dist_m"]) - 1


def summarize(g):
    return pd.Series({
        "n_rows": len(g),
        "n_laps": g.groupby(["session_id", "lap"]).ngroups,
        "n_sessions": g["session_id"].nunique(),
        "peak_dist_med": g["peak_dist_m"].median(),
        "peak_dist_p10": g["peak_dist_m"].quantile(0.1),
        "peak_dist_p90": g["peak_dist_m"].quantile(0.9),
        "entry_med": g["entry_dist_m"].median(),
        "exit_med": g["exit_dist_m"].median(),
        "min_speed_dist_med": g["min_speed_dist_m"].median(),
        "min_speed_mph_med": g["min_speed_mph"].median(),
        "min_speed_mph_p25": g["min_speed_mph"].quantile(0.25),
        "min_speed_mph_p75": g["min_speed_mph"].quantile(0.75),
        "duration_med": g["duration_s"].median(),
        "lat_g_med": g["peak_lat_g"].median(),
        "left_frac": (g["direction"] == "left").mean(),
        "right_frac": (g["direction"] == "right").mean(),
    })


summary = df.groupby("cluster").apply(summarize, include_groups=False).reset_index()
summary["dir"] = np.where(summary["left_frac"] > summary["right_frac"], "L", "R")
summary["dir_purity"] = np.maximum(summary["left_frac"], summary["right_frac"])
summary = summary.sort_values("peak_dist_med").reset_index(drop=True)
summary.insert(0, "ord", range(1, len(summary) + 1))

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)
cols = ["ord", "n_laps", "peak_dist_med", "entry_med", "exit_med",
        "min_speed_dist_med", "min_speed_mph_med", "min_speed_mph_p25", "min_speed_mph_p75",
        "duration_med", "lat_g_med", "dir", "dir_purity"]
print("\n=== FAST LAPS (2025+, top quartile) ===")
print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:7.1f}"))

out = Path(__file__).parent / "ridge_clusters_fast.csv"
summary.to_csv(out, index=False)
print(f"\nwrote {out}")
