"""GPS calibration via T10 anchoring.

Hypothesis: T10 line is physically constrained (wheels on kerbs both sides) so
GPS variation in T10 = drift, not line. Use that to per-lap translate the GPS
trace, then re-measure T11 spread.

Step 1: verify T10 is tighter than T11 across fast laps (otherwise calibration
        won't help).
Step 2: compute per-lap translation using T10 centroid alignment to a reference.
Step 3: apply to T11 and re-measure.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

T10_START, T10_END = 2279.0, 2443.0
T11_START, T11_END = 2529.0, 2681.0
M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON_AT_RIDGE = 111_132.0 * np.cos(np.radians(47.255))

root = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
laps = pd.concat([pd.read_csv(p) for p in root.rglob("laps.csv")], ignore_index=True)
clean = laps[laps.is_clean.astype(bool)]
threshold = clean.lap_time_s.quantile(0.10)
fast_laps = clean[clean.lap_time_s <= threshold][["session_id", "lap", "lap_time_s"]]
# Add the May-1 best lap even though it's just outside top-10%
focus_extras = pd.DataFrame([{"session_id": "20260501-101355", "lap": 3, "lap_time_s": 124.5}])
all_laps = pd.concat([fast_laps, focus_extras], ignore_index=True).drop_duplicates(["session_id", "lap"])

print(f"Loading {len(all_laps)} traces...")


def resample_segment(seg: pd.DataFrame, start: float, end: float, step: float = 1.0):
    """Interpolate (lat, long) onto a uniform dist_lap_m grid from start to end."""
    in_seg = seg[(seg["dist_lap_m"] >= start) & (seg["dist_lap_m"] <= end)].sort_values("dist_lap_m")
    if len(in_seg) < 5:
        return None
    grid = np.arange(start, end + step, step)
    lat_i = np.interp(grid, in_seg["dist_lap_m"].values, in_seg["lat"].values)
    lon_i = np.interp(grid, in_seg["dist_lap_m"].values, in_seg["long"].values)
    return grid, lat_i, lon_i


# Pull T10 + T11 resampled traces
records = []
for _, lap in all_laps.iterrows():
    sid, ln = lap.session_id, int(lap.lap)
    s = pd.read_parquet(root / sid / "samples.parquet")
    lap_s = s[s["lap"] == ln]
    t10 = resample_segment(lap_s, T10_START, T10_END)
    t11 = resample_segment(lap_s, T11_START, T11_END)
    if t10 is None or t11 is None:
        continue
    records.append({
        "session_id": sid, "lap": ln, "lap_time_s": lap.lap_time_s,
        "t10_lat": t10[1], "t10_lon": t10[2],
        "t11_lat": t11[1], "t11_lon": t11[2],
    })

df = pd.DataFrame(records)
print(f"Got {len(df)} valid traces")

# Stack into 2D arrays: rows=laps, cols=samples
T10_LAT = np.stack(df["t10_lat"].values)  # shape: (n_laps, n_samples)
T10_LON = np.stack(df["t10_lon"].values)
T11_LAT = np.stack(df["t11_lat"].values)
T11_LON = np.stack(df["t11_lon"].values)

# Step 1: per-position std across laps (= cross-lap spread at each dist)
def position_std_meters(lat_arr, lon_arr):
    lat_std = lat_arr.std(axis=0) * M_PER_DEG_LAT
    lon_std = lon_arr.std(axis=0) * M_PER_DEG_LON_AT_RIDGE
    # combined position std (RMS)
    return np.sqrt(lat_std ** 2 + lon_std ** 2)

t10_std = position_std_meters(T10_LAT, T10_LON)
t11_std = position_std_meters(T11_LAT, T11_LON)

print(f"\n=== BEFORE correction ===")
print(f"T10 cross-lap position std: median={np.median(t10_std):.2f}m  max={t10_std.max():.2f}m")
print(f"T11 cross-lap position std: median={np.median(t11_std):.2f}m  max={t11_std.max():.2f}m")
print(f"  Ratio T11/T10: {np.median(t11_std)/np.median(t10_std):.2f}x")

# Step 2: pick reference (best lap), compute per-lap translation that aligns T10 to ref
# Use the all-time fastest lap as reference
fastest_idx = df["lap_time_s"].idxmin()
print(f"\nReference lap: {df.loc[fastest_idx].session_id} L{df.loc[fastest_idx].lap}  ({df.loc[fastest_idx].lap_time_s:.2f}s)")

ref_lat = T10_LAT[fastest_idx]
ref_lon = T10_LON[fastest_idx]

# Per-lap translation = mean(ref - lap) over all T10 sample points
trans_lat = (ref_lat - T10_LAT).mean(axis=1)  # (n_laps,)
trans_lon = (ref_lon - T10_LON).mean(axis=1)
trans_lat_m = trans_lat * M_PER_DEG_LAT
trans_lon_m = trans_lon * M_PER_DEG_LON_AT_RIDGE
trans_total_m = np.sqrt(trans_lat_m ** 2 + trans_lon_m ** 2)
print(f"Per-lap translations needed (m): median={np.median(trans_total_m):.2f}  90th-pct={np.percentile(trans_total_m, 90):.2f}  max={trans_total_m.max():.2f}")

# Step 3: apply translation to T11 traces and re-measure
T11_LAT_corrected = T11_LAT + trans_lat[:, None]
T11_LON_corrected = T11_LON + trans_lon[:, None]
T10_LAT_corrected = T10_LAT + trans_lat[:, None]
T10_LON_corrected = T10_LON + trans_lon[:, None]

t10_std_after = position_std_meters(T10_LAT_corrected, T10_LON_corrected)
t11_std_after = position_std_meters(T11_LAT_corrected, T11_LON_corrected)

print(f"\n=== AFTER T10-anchored correction ===")
print(f"T10 cross-lap std: median={np.median(t10_std_after):.2f}m  max={t10_std_after.max():.2f}m   (was {np.median(t10_std):.2f} / {t10_std.max():.2f})")
print(f"T11 cross-lap std: median={np.median(t11_std_after):.2f}m  max={t11_std_after.max():.2f}m   (was {np.median(t11_std):.2f} / {t11_std.max():.2f})")

# Step 4: focal laps' top-of-curve in T11 — before vs after
def focus_top_lat(idx):
    raw_max = T11_LAT[idx].max()
    cor_max = T11_LAT_corrected[idx].max()
    return raw_max, cor_max

focus = [
    ("BEST FAST 2024-05-18 L2", "20240518-111554", 2),
    ("MAY-1     2026-05-01 L3", "20260501-101355", 3),
]
print(f"\n=== T11 max-latitude (top of curve) for focal laps ===")
results = []
for label, sid, ln in focus:
    rows = df[(df.session_id == sid) & (df.lap == ln)]
    if len(rows) == 0:
        print(f"{label}: not present")
        continue
    idx = rows.index[0]
    raw, cor = focus_top_lat(idx)
    print(f"{label}: raw_max_lat={raw:.7f}  corrected={cor:.7f}  (translation: {trans_lat_m[idx]:+.2f}m N/S, {trans_lon_m[idx]:+.2f}m E/W)")
    results.append((label, idx, raw, cor))

# Compute the BEST FAST vs MAY-1 delta before and after
if len(results) == 2:
    raw_delta_m = (results[0][2] - results[1][2]) * M_PER_DEG_LAT
    cor_delta_m = (results[0][3] - results[1][3]) * M_PER_DEG_LAT
    print(f"\nBEST FAST vs MAY-1 max-lat delta (positive = BEST FAST further north):")
    print(f"  BEFORE correction: {raw_delta_m:+.2f}m")
    print(f"  AFTER  correction: {cor_delta_m:+.2f}m")

# Plot: T11 envelope before and after correction
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for i, (title, lat_arr, lon_arr) in enumerate([("RAW", T11_LAT, T11_LON), ("T10-anchored corrected", T11_LAT_corrected, T11_LON_corrected)]):
    ax = axes[i]
    for j in range(len(df)):
        ax.plot(lon_arr[j], lat_arr[j], color="gray", alpha=0.3, linewidth=1)
    # focus laps
    for label, sid, ln, color in [("BEST FAST", "20240518-111554", 2, "tab:blue"), ("MAY-1", "20260501-101355", 3, "tab:red")]:
        rows = df[(df.session_id == sid) & (df.lap == ln)]
        if len(rows) == 0:
            continue
        idx = rows.index[0]
        ax.plot(lon_arr[idx], lat_arr[idx], color=color, linewidth=3, label=label)
    ax.set_title(f"T11 {title}\nmedian cross-lap pos std: {np.median(position_std_meters(lat_arr, lon_arr)):.2f}m")
    ax.set_aspect("equal")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

plt.tight_layout()
out = Path(__file__).parent / "t11_calibrated.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nwrote {out}")
