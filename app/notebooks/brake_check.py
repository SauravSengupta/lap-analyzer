"""Find T8 brake-on distance from candidate brake_on_offset_m."""
import pandas as pd
import numpy as np
from pathlib import Path

CSV = Path(__file__).resolve().parents[2] / "data" / "corpus" / "ridge_candidates.csv"
df = pd.read_csv(CSV)
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2025-01-01"].copy()

lap_times = df.groupby(["session_id", "lap"])["lap_time_s"].first()
fast_keys = set(lap_times[lap_times <= lap_times.quantile(0.25)].index)
df = df[df.set_index(["session_id", "lap"]).index.isin(fast_keys)].copy()

# brake_on_dist = entry_dist_m + brake_on_offset_m  (absolute)
df["brake_on_dist"] = df["entry_dist_m"] + df["brake_on_offset_m"]

# G cluster region: peak_dist_m between 1700 and 2000
g_region = df[(df["peak_dist_m"] >= 1700) & (df["peak_dist_m"] <= 2000)].copy()
print("=== Inner-loop R complex (peak 1700-2000m) ===")
print(f"rows={len(g_region)}, laps={g_region.groupby(['session_id','lap']).ngroups}")
print(f"brake_on_dist available: {g_region['brake_on_dist'].notna().sum()} / {len(g_region)}")

# Use one row per lap (the earliest detected peak in this region per lap)
g_first = g_region.sort_values("peak_dist_m").drop_duplicates(["session_id", "lap"], keep="first")
print(f"\nbrake_on_dist (first detected peak per lap):")
print(f"  min={g_first['brake_on_dist'].min():.0f}  p10={g_first['brake_on_dist'].quantile(0.1):.0f}  median={g_first['brake_on_dist'].median():.0f}  p90={g_first['brake_on_dist'].quantile(0.9):.0f}  max={g_first['brake_on_dist'].max():.0f}")
print(f"\nentry_dist (first detected peak per lap):")
print(f"  min={g_first['entry_dist_m'].min():.0f}  p10={g_first['entry_dist_m'].quantile(0.1):.0f}  median={g_first['entry_dist_m'].median():.0f}  p90={g_first['entry_dist_m'].quantile(0.9):.0f}  max={g_first['entry_dist_m'].max():.0f}")

# Now T6 region: peak_dist_m between 1150 and 1650
print("\n=== T6 carousel (peak 1150-1650m) ===")
t6 = df[(df["peak_dist_m"] >= 1150) & (df["peak_dist_m"] <= 1650)].copy()
t6_first = t6.sort_values("peak_dist_m").drop_duplicates(["session_id", "lap"], keep="first")
t6_last = t6.sort_values("peak_dist_m").drop_duplicates(["session_id", "lap"], keep="last")
print(f"earliest entry per lap (any peak in window):")
print(f"  min={t6_first['entry_dist_m'].min():.0f}  median={t6_first['entry_dist_m'].median():.0f}  max={t6_first['entry_dist_m'].max():.0f}")
print(f"latest exit per lap:")
print(f"  min={t6_last['exit_dist_m'].min():.0f}  median={t6_last['exit_dist_m'].median():.0f}  max={t6_last['exit_dist_m'].max():.0f}")
print(f"brake_on_dist (earliest braking input in window, any peak):")
brake_first = t6.dropna(subset=["brake_on_dist"]).sort_values("brake_on_dist").drop_duplicates(["session_id", "lap"], keep="first")
print(f"  median={brake_first['brake_on_dist'].median():.0f}  p10={brake_first['brake_on_dist'].quantile(0.1):.0f}  p90={brake_first['brake_on_dist'].quantile(0.9):.0f}")

# Also T1: peak 400-700
print("\n=== T1+T2 cluster (peak 400-700m) ===")
t1 = df[(df["peak_dist_m"] >= 400) & (df["peak_dist_m"] <= 700)].copy()
print(f"all peaks brake_on_dist median: {t1['brake_on_dist'].median():.0f}")
# T1 alone (peak 400-560)
t1_only = df[(df["peak_dist_m"] >= 400) & (df["peak_dist_m"] <= 560)]
print(f"T1 peaks (400-560m): brake_on_dist median: {t1_only['brake_on_dist'].median():.0f}, n={len(t1_only)}")
t2_only = df[(df["peak_dist_m"] >= 560) & (df["peak_dist_m"] <= 700)]
print(f"T2 peaks (560-700m): brake_on_dist median: {t2_only['brake_on_dist'].median():.0f}, n={len(t2_only)}")
