"""Compare T11 transits: best-on-a-fast-lap vs best 2026-05-01 lap.

Raw 'shortest time in corner' is misleading — it favors laps where the driver
arrived already slow (less ground to make up). Real-world comparison should
filter to competitive lap times.
"""
import pandas as pd
from pathlib import Path

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)

root = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
corners = pd.concat([pd.read_parquet(p) for p in root.rglob("corners.parquet")], ignore_index=True)
laps = pd.concat([pd.read_csv(p) for p in root.rglob("laps.csv")], ignore_index=True)

t11 = corners[corners.corner_id == "T11"].copy()
t11 = t11.merge(laps[["session_id", "lap", "lap_time_s"]], on=["session_id", "lap"], how="left")
clean_laps = laps[laps.is_clean.astype(bool)]

# Three candidates for "best T11 transit"
print("=== T11: three different 'best' definitions ===\n")

# A. Raw fastest time in corner (what you literally asked)
a = t11.sort_values("time_in_corner_s").iloc[0]
print(f"A. Shortest time_in_corner_s: {a.time_in_corner_s:.3f}s  on {a.session_id} L{int(a.lap)}  (lap {a.lap_time_s:.2f}s — {a.lap_time_s - clean_laps.lap_time_s.min():+.1f}s vs all-time best)")
print(f"   entry {a.entry_speed_mph:.1f} mph, min {a.min_speed_mph:.1f}, exit {a.exit_speed_mph:.1f} — entry already at min speed; came in already slow.")

# B. Highest min_speed_mph (carried most apex speed)
b = t11.sort_values("min_speed_mph", ascending=False).iloc[0]
print(f"\nB. Highest min_speed_mph: {b.min_speed_mph:.1f} mph  on {b.session_id} L{int(b.lap)}  (lap {b.lap_time_s:.2f}s)")
print(f"   entry {b.entry_speed_mph:.1f}, min {b.min_speed_mph:.1f}, exit {b.exit_speed_mph:.1f}, time {b.time_in_corner_s:.3f}s")

# C. T11 transit on a top-decile overall lap, fastest within that
fast_lap_threshold = clean_laps.lap_time_s.quantile(0.10)  # top 10% of laps
fast_t11 = t11[t11.lap_time_s <= fast_lap_threshold].sort_values("time_in_corner_s")
c = fast_t11.iloc[0]
print(f"\nC. Fastest T11 on a top-10% lap (<= {fast_lap_threshold:.2f}s): {c.time_in_corner_s:.3f}s  on {c.session_id} L{int(c.lap)}  (lap {c.lap_time_s:.2f}s)")
print(f"   entry {c.entry_speed_mph:.1f}, min {c.min_speed_mph:.1f}, exit {c.exit_speed_mph:.1f}")

# 2026-05-01 best lap
may1 = clean_laps[clean_laps.session_id.str.startswith("20260501")].sort_values("lap_time_s")
may1_best_lap = may1.iloc[0]
may1_t11 = t11[(t11.session_id == may1_best_lap.session_id) & (t11.lap == may1_best_lap.lap)].iloc[0]
print(f"\n=== Your fastest 2026-05-01 lap ===")
print(f"{may1_best_lap.session_id} L{int(may1_best_lap.lap)}  lap {may1_best_lap.lap_time_s:.2f}s  ({may1_best_lap.lap_time_s - clean_laps.lap_time_s.min():+.2f}s vs all-time best)")
print(f"T11 transit: time {may1_t11.time_in_corner_s:.3f}s, entry {may1_t11.entry_speed_mph:.1f}, min {may1_t11.min_speed_mph:.1f}, exit {may1_t11.exit_speed_mph:.1f}")

# Real comparison: option C (fastest T11 on a fast lap) vs May-1 best lap T11
print(f"\n=== T11 SIDE-BY-SIDE: best-fast-lap T11 vs May-1 best-lap T11 ===")
metric_cols = [
    "time_in_corner_s", "entry_speed_mph", "min_speed_mph", "min_speed_dist_m",
    "apex_speed_mph", "apex_dist_offset_m", "exit_speed_mph",
    "max_lat_g", "max_decel_g",
    "brake_on_dist_m", "brake_off_dist_m", "throttle_return_dist_m", "wot_dist_m",
    "mean_throttle_norm", "pct_wot", "pct_braking", "mean_lat_g",
]
compare = pd.DataFrame({
    f"BEST {c.session_id} L{int(c.lap)} (lap {c.lap_time_s:.1f}s)": c[metric_cols],
    f"MAY-1 {may1_best_lap.session_id} L{int(may1_best_lap.lap)} (lap {may1_best_lap.lap_time_s:.1f}s)": may1_t11[metric_cols],
})
compare["delta"] = compare.iloc[:, 1] - compare.iloc[:, 0]
print(compare.to_string())

# How does May-1's T11 rank among the population of clean T11 transits?
n = len(t11)
faster_t11 = (t11.time_in_corner_s < may1_t11.time_in_corner_s).sum()
print(f"\nMay-1's T11 transit ({may1_t11.time_in_corner_s:.3f}s) is faster than {n - faster_t11}/{n} all-time T11 transits "
      f"(rank #{faster_t11 + 1}). Median: {t11.time_in_corner_s.median():.3f}s")
