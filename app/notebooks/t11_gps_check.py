"""GPS-drift sanity check for T11.

Plot all top-decile-fast-lap traces through T11 overlaid. If the 2024 'BEST FAST'
and 2026-05-01 laps are within the spread of all fast laps, the 8.3m apex
difference is GPS noise. If they cluster on opposite sides of the spread, real.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

T11_START = 2529.0
T11_END = 2681.0
PRE = 100.0
POST = 80.0
M_PER_DEG_LAT = 111_132.0

root = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
laps = pd.concat([pd.read_csv(p) for p in root.rglob("laps.csv")], ignore_index=True)
clean = laps[laps.is_clean.astype(bool)]

# Top-decile lap times
threshold = clean.lap_time_s.quantile(0.10)
fast_laps = clean[clean.lap_time_s <= threshold][["session_id", "lap", "lap_time_s"]].copy()
print(f"Top-10% lap-time threshold: {threshold:.2f}s ({len(fast_laps)} laps)")

# Pull T11 traces for each fast lap
traces = []
gps_acc_stats = []
for _, lap_row in fast_laps.iterrows():
    sid, ln = lap_row["session_id"], int(lap_row["lap"])
    s = pd.read_parquet(root / sid / "samples.parquet")
    seg = s[
        (s["lap"] == ln)
        & (s["dist_lap_m"] >= T11_START - PRE)
        & (s["dist_lap_m"] <= T11_END + POST)
    ].copy()
    if len(seg) < 5:
        continue
    traces.append((sid, ln, lap_row["lap_time_s"], seg))
    if "gps_accuracy_m" in seg.columns:
        gps_acc_stats.append({
            "session_id": sid,
            "lap": ln,
            "median_acc_m": seg["gps_accuracy_m"].median(),
            "max_acc_m": seg["gps_accuracy_m"].max(),
        })

print(f"Loaded {len(traces)} traces")
print(f"\n=== gps_accuracy_m within T11 region (across top-10% laps) ===")
acc = pd.DataFrame(gps_acc_stats)
print(f"median across all samples in all fast laps: {acc.median_acc_m.median():.2f} m")
print(f"  range of per-lap medians: {acc.median_acc_m.min():.2f} - {acc.median_acc_m.max():.2f} m")
print(f"  range of per-lap maxes:   {acc.max_acc_m.min():.2f} - {acc.max_acc_m.max():.2f} m")
print(f"  90th-percentile per-lap max: {acc.max_acc_m.quantile(0.9):.2f} m")

# Top-of-curve (max latitude) for each trace
tops = []
for sid, ln, lt, seg in traces:
    in_corner = seg[(seg["dist_lap_m"] >= T11_START) & (seg["dist_lap_m"] <= T11_END)]
    if len(in_corner) < 3:
        continue
    top = in_corner.loc[in_corner["lat"].idxmax()]
    tops.append({
        "session_id": sid,
        "lap": ln,
        "lap_time_s": lt,
        "top_lat": top["lat"],
        "top_long": top["long"],
        "top_dist": top["dist_lap_m"],
        "top_speed": top["speed_mph"],
    })

tops_df = pd.DataFrame(tops).sort_values("top_lat", ascending=False)
print(f"\n=== top-of-curve (max latitude in T11) across {len(tops_df)} fast laps ===")
print(f"max-lat range: {tops_df.top_lat.min():.7f} ... {tops_df.top_lat.max():.7f}")
spread_lat = (tops_df.top_lat.max() - tops_df.top_lat.min()) * M_PER_DEG_LAT
print(f"spread (max - min) across all fast laps: {spread_lat:.2f} m")
print(f"std (1-sigma): {tops_df.top_lat.std() * M_PER_DEG_LAT:.2f} m")

# Where are our two laps in this distribution?
def label_for(sid, ln):
    return f"{sid} L{ln}"
focus = [
    ("BEST FAST 2024-05-18 L2", "20240518-111554", 2),
    ("MAY-1     2026-05-01 L3", "20260501-101355", 3),
]
print(f"\n=== focal laps' top-of-curve (vs the {len(tops_df)}-lap distribution) ===")
median_lat = tops_df.top_lat.median()
for label, sid, ln in focus:
    row = tops_df[(tops_df.session_id == sid) & (tops_df.lap == ln)]
    if len(row) == 0:
        print(f"{label}: NOT in top-10% laps (lap time {clean[(clean.session_id==sid)&(clean.lap==ln)].lap_time_s.iloc[0]:.2f}s)")
        # Add manually for plotting purposes
        s = pd.read_parquet(root / sid / "samples.parquet")
        in_corner = s[(s["lap"] == ln) & (s["dist_lap_m"] >= T11_START) & (s["dist_lap_m"] <= T11_END)]
        top = in_corner.loc[in_corner["lat"].idxmax()]
        delta_m = (top["lat"] - median_lat) * M_PER_DEG_LAT
        print(f"  top_lat = {top['lat']:.7f}  (delta from fast-lap median: {delta_m:+.2f} m)")
        continue
    r = row.iloc[0]
    delta_m = (r.top_lat - median_lat) * M_PER_DEG_LAT
    rank = (tops_df.top_lat > r.top_lat).sum() + 1
    print(f"{label}: top_lat = {r.top_lat:.7f}  rank #{rank}/{len(tops_df)} (north-most)  delta from median: {delta_m:+.2f} m")

# Plot all traces overlaid
fig, ax = plt.subplots(figsize=(11, 9))
for sid, ln, lt, seg in traces:
    ax.plot(seg["long"], seg["lat"], color="gray", alpha=0.25, linewidth=1)

# Highlight the two focal laps + add the May-1 lap (which isn't in top-10%)
focus_data = []
for label, sid, ln in focus:
    s = pd.read_parquet(root / sid / "samples.parquet")
    seg = s[(s["lap"] == ln) & (s["dist_lap_m"] >= T11_START - PRE) & (s["dist_lap_m"] <= T11_END + POST)]
    focus_data.append((label, seg))

ax.plot(focus_data[0][1]["long"], focus_data[0][1]["lat"], color="tab:blue", linewidth=3, label=focus_data[0][0])
ax.plot(focus_data[1][1]["long"], focus_data[1][1]["lat"], color="tab:red", linewidth=3, label=focus_data[1][0])

# Mark top-of-curve for each fast lap
ax.scatter(tops_df.top_long, tops_df.top_lat, s=15, c="black", alpha=0.4, zorder=3, label="top-of-curve (each fast lap)")

ax.set_title(f"T11 — all {len(traces)} top-10% lap traces (gray) + 2 focal laps\n"
             f"max-lat spread across fast laps: {spread_lat:.1f} m  |  "
             f"GPS accuracy median: {acc.median_acc_m.median():.1f} m", fontsize=11)
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9)

# Annotate the spread on the plot
xlim = ax.get_xlim()
ax.axhline(tops_df.top_lat.max(), color="green", linestyle=":", alpha=0.5, linewidth=1)
ax.axhline(tops_df.top_lat.min(), color="red",   linestyle=":", alpha=0.5, linewidth=1)
ax.text(xlim[1], tops_df.top_lat.max(), f"  deepest fast-lap apex", va="center", fontsize=8, color="green")
ax.text(xlim[1], tops_df.top_lat.min(), f"  shallowest fast-lap apex", va="center", fontsize=8, color="red")

plt.tight_layout()
out = Path(__file__).parent / "t11_envelope.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nwrote {out}")
