"""Re-create the T11 envelope plot with corrections applied:
  1. dist_lap_m per-lap normalization (already in samples.parquet)
  2. per-lap GPS drift correction (using T2/T13/T14 anchors)
  3. exclude flagged gps_unreliable sessions
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

T11_START = 2529.0
T11_END = 2681.0
PRE = 100.0
POST = 80.0
M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))

root = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
notes_path = Path(r"D:\Projects\lap-analyzer\data\notes\ridge.json")

# Sessions to skip for spatial analysis
notes = json.loads(notes_path.read_text())
gps_bad = {sid for sid, info in notes.items() if info.get("flag") == "gps_unreliable"}
print(f"Excluding {len(gps_bad)} gps_unreliable sessions from envelope: {gps_bad}")

DISAGREEMENT_THRESHOLD_M = 12.0
T11_OFFSET_MAX_THRESHOLD_M = 40.0  # exclude transits where T11 samples are too far off the reference path

# Per-lap drift offsets from corner-transit data
corners_df = pd.concat([pd.read_parquet(p) for p in root.rglob("corners.parquet")], ignore_index=True)
drift = corners_df.drop_duplicates(["session_id", "lap"])[
    ["session_id", "lap", "gps_drift_lat_m", "gps_drift_lon_m", "gps_drift_disagreement_m"]
]
unreliable = drift[drift.gps_drift_disagreement_m > DISAGREEMENT_THRESHOLD_M]
print(f"Per-lap calibration unreliable (disagreement > {DISAGREEMENT_THRESHOLD_M}m): {len(unreliable)} laps")

# Per-transit T11-specific reliability filter
t11_only = corners_df[corners_df.corner_id == "T11"][["session_id","lap","track_dist_offset_max_m"]]
t11_unreliable = t11_only[t11_only.track_dist_offset_max_m > T11_OFFSET_MAX_THRESHOLD_M]
print(f"Per-transit T11 mapping unreliable (max offset > {T11_OFFSET_MAX_THRESHOLD_M}m): {len(t11_unreliable)} transits")
t11_unreliable_keys = set(zip(t11_unreliable.session_id, t11_unreliable.lap.astype(int)))

# Top-decile fast laps
laps = pd.concat([pd.read_csv(p) for p in root.rglob("laps.csv")], ignore_index=True)
clean = laps[laps.is_clean.astype(bool)]
threshold = clean.lap_time_s.quantile(0.10)
fast_laps = clean[clean.lap_time_s <= threshold][["session_id", "lap", "lap_time_s"]]
fast_laps = fast_laps[~fast_laps.session_id.isin(gps_bad)]
unreliable_keys = set(zip(unreliable.session_id, unreliable.lap))
before_per_lap = len(fast_laps)
fast_laps = fast_laps[~fast_laps.apply(lambda r: (r.session_id, int(r.lap)) in unreliable_keys, axis=1)]
dropped_disagreement = before_per_lap - len(fast_laps)
before_t11 = len(fast_laps)
fast_laps = fast_laps[~fast_laps.apply(lambda r: (r.session_id, int(r.lap)) in t11_unreliable_keys, axis=1)]
dropped_t11 = before_t11 - len(fast_laps)
print(f"Top-10% threshold: {threshold:.2f}s; {len(fast_laps)} laps after filters (dropped {dropped_disagreement} for anchor disagreement, {dropped_t11} for T11 mapping unreliable)")

# Pull T11 traces for each fast lap, apply correction
traces = []
tops = []
for _, r in fast_laps.iterrows():
    sid, ln = r.session_id, int(r.lap)
    s = pd.read_parquet(root / sid / "samples.parquet")
    seg = s[
        (s["lap"] == ln)
        & (s["dist_lap_m"] >= T11_START - PRE)
        & (s["dist_lap_m"] <= T11_END + POST)
    ].copy()
    if len(seg) < 5:
        continue
    d = drift[(drift.session_id == sid) & (drift.lap == ln)]
    if len(d) == 0:
        continue
    dlat_m = float(d.gps_drift_lat_m.iloc[0])
    dlon_m = float(d.gps_drift_lon_m.iloc[0])
    seg["lat_corr"] = seg["lat"] - dlat_m / M_PER_DEG_LAT
    seg["long_corr"] = seg["long"] - dlon_m / M_PER_DEG_LON
    traces.append((sid, ln, r.lap_time_s, seg))
    inside = seg[(seg["dist_lap_m"] >= T11_START) & (seg["dist_lap_m"] <= T11_END)]
    top = inside.loc[inside["lat_corr"].idxmax()]
    tops.append({"session_id": sid, "lap": ln, "top_lat": top["lat_corr"], "top_long": top["long_corr"]})

tops_df = pd.DataFrame(tops)
spread_lat_m = (tops_df.top_lat.max() - tops_df.top_lat.min()) * M_PER_DEG_LAT
std_lat_m = tops_df.top_lat.std() * M_PER_DEG_LAT

# Pull focal laps the same way (BEST FAST is in fast_laps, MAY-1 isn't)
def load_focal(sid, ln):
    s = pd.read_parquet(root / sid / "samples.parquet")
    seg = s[(s["lap"] == ln) & (s["dist_lap_m"] >= T11_START - PRE) & (s["dist_lap_m"] <= T11_END + POST)].copy()
    d = drift[(drift.session_id == sid) & (drift.lap == ln)]
    dlat_m = float(d.gps_drift_lat_m.iloc[0])
    dlon_m = float(d.gps_drift_lon_m.iloc[0])
    seg["lat_corr"] = seg["lat"] - dlat_m / M_PER_DEG_LAT
    seg["long_corr"] = seg["long"] - dlon_m / M_PER_DEG_LON
    return seg, dlat_m, dlon_m

bf_seg, bf_dlat, bf_dlon = load_focal("20240518-111554", 2)
m1_seg, m1_dlat, m1_dlon = load_focal("20260501-101355", 3)

print(f"\nBEST FAST drift applied: ({bf_dlat:+.2f}, {bf_dlon:+.2f}) m")
print(f"MAY-1     drift applied: ({m1_dlat:+.2f}, {m1_dlon:+.2f}) m")

# Plot
fig, ax = plt.subplots(figsize=(11, 9))
for sid, ln, lt, seg in traces:
    ax.plot(seg["long_corr"], seg["lat_corr"], color="gray", alpha=0.30, linewidth=1)

ax.plot(bf_seg["long_corr"], bf_seg["lat_corr"], color="tab:blue", linewidth=3, label="BEST FAST 2024-05-18 L2")
ax.plot(m1_seg["long_corr"], m1_seg["lat_corr"], color="tab:red",  linewidth=3, label="MAY-1     2026-05-01 L3")

ax.scatter(tops_df.top_long, tops_df.top_lat, s=15, c="black", alpha=0.4, zorder=3, label="top-of-curve (each fast lap)")

ax.axhline(tops_df.top_lat.max(), color="green", linestyle=":", alpha=0.5, linewidth=1)
ax.axhline(tops_df.top_lat.min(), color="red",   linestyle=":", alpha=0.5, linewidth=1)

ax.set_title(f"T11 — {len(traces)} top-10% lap traces, CORRECTED (track_dist_m + GPS drift)\n"
             f"max-lat spread: {spread_lat_m:.1f}m  |  std: {std_lat_m:.1f}m  |  excluded: gps_unreliable + disagreement>{DISAGREEMENT_THRESHOLD_M}m + T11 offset>{T11_OFFSET_MAX_THRESHOLD_M}m", fontsize=10)
ax.set_xlabel("longitude (drift-corrected)")
ax.set_ylabel("latitude (drift-corrected)")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9)

xlim = ax.get_xlim()
ax.text(xlim[1], tops_df.top_lat.max(), f"  deepest fast-lap apex", va="center", fontsize=8, color="green")
ax.text(xlim[1], tops_df.top_lat.min(), f"  shallowest fast-lap apex", va="center", fontsize=8, color="red")

plt.tight_layout()
out = Path(__file__).parent / "t11_envelope_corrected.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nwrote {out}")

# Compare before/after spread (recompute uncorrected for comparison)
print("\n=== T11 max-lat envelope ===")
print(f"  CORRECTED: spread {spread_lat_m:.1f}m  std {std_lat_m:.1f}m  (n={len(traces)} laps)")
# compute uncorrected for same lap set
uncorr_tops = []
for sid, ln, lt, seg in traces:
    inside = seg[(seg["dist_lap_m"] >= T11_START) & (seg["dist_lap_m"] <= T11_END)]
    uncorr_tops.append(float(inside["lat"].max()))
uncorr_arr = np.array(uncorr_tops)
print(f"  RAW (same laps): spread {(uncorr_arr.max()-uncorr_arr.min())*M_PER_DEG_LAT:.1f}m  std {uncorr_arr.std()*M_PER_DEG_LAT:.1f}m")
