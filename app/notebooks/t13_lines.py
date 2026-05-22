"""Track-map overlay of every clean + reliable lap through T13.

Answers: is the GPS data, after the full cleanup pipeline (anchor drift
correction + synthetic centerline + glitch filtering + reliability flags),
tight enough to draw racing lines on a map (visualizer option 3)?

Plots drift-corrected GPS lines for all T13-reliable laps, plus the centerline
and apex pin, and prints the cross-lap positional scatter at the apex.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

from lap_analyzer.analysis import load_corpus
from lap_analyzer.config import corpus_dir, sessions_dir

TRACK = "ridge"
M_LAT = 111_132.0
M_LON = 111_132.0 * np.cos(np.radians(47.255))

td = json.loads((Path("tracks") / f"{TRACK}.json").read_text(encoding="utf-8"))
T13 = next(c for c in td["corners"] if c["id"] == "T13")
anchor = next(a for a in td["calibration_anchors"] if a["corner_id"] == "T13")
LO, HI = T13["start_m"] - 60.0, T13["end_m"] + 70.0

corpus = load_corpus(TRACK)
laps_csv = pd.concat(
    [pd.read_csv(p) for p in sorted(sessions_dir(TRACK).rglob("laps.csv"))],
    ignore_index=True)
clean = set(map(tuple, laps_csv[laps_csv["is_clean"].fillna(False).astype(bool)]
                [["session_id", "lap"]].astype({"lap": int}).itertuples(index=False, name=None)))

t13 = corpus[corpus["corner_id"] == "T13"]
reliable = {(r.session_id, int(r.lap)) for r in t13.itertuples(index=False) if r.transit_reliable}
keys = sorted((reliable & clean))
lap_time = laps_csv.set_index(["session_id", "lap"])["lap_time_s"]
print(f"T13 transits: {len(t13)} total, {int(t13['transit_reliable'].sum())} reliable, "
      f"{len(keys)} reliable + clean -> plotting these\n")


def drift_xy(s, lat0, lon0):
    """Drift-corrected position in local metres."""
    lat = s["lat"] - s.get("gps_drift_lat_m", 0.0) / M_LAT
    lon = s["long"] - s.get("gps_drift_lon_m", 0.0) / M_LON
    return (lon - lon0) * M_LON, (lat - lat0) * M_LAT


def drop_gps_glitches(s):
    s = s.sort_values("t").reset_index(drop=True)
    sm = s["track_dist_m"] < 500
    if sm.any():
        s = s.iloc[sm.idxmax():].reset_index(drop=True)
    s = s[(s["track_dist_m"] - s["dist_lap_m"]).abs() < 50].reset_index(drop=True)
    rmax = s["track_dist_m"].cummax()
    return s[s["track_dist_m"] >= rmax - 1.0].reset_index(drop=True)


lat0, lon0 = anchor["ref_lat"], anchor["ref_long"]
segs = {}
by_session: dict[str, list[int]] = {}
for sid, lap in keys:
    by_session.setdefault(sid, []).append(lap)
for sid, laps in by_session.items():
    try:
        df = pd.read_parquet(sessions_dir(TRACK) / sid / "samples.parquet")
    except Exception:
        continue
    for lap in laps:
        s = drop_gps_glitches(df[df["lap"] == lap])
        s = s[(s["track_dist_m"] >= LO) & (s["track_dist_m"] <= HI)]
        if len(s) > 5:
            x, y = drift_xy(s, lat0, lon0)
            segs[(sid, lap)] = (x.to_numpy(), y.to_numpy(),
                                s["track_dist_m"].to_numpy())

print(f"laps drawn: {len(segs)}")

# cross-lap scatter at a fixed track_dist_m: how far apart do laps sit at the
# same point on track? Small = trustworthy positions. Measured on the approach
# straight (where GPS is easy) and at the apex (where cornering stresses it).
def scatter_at(target_m):
    pts = []
    for x, y, tdm in segs.values():
        j = int(np.argmin(np.abs(tdm - target_m)))
        pts.append((x[j], y[j]))
    pts = np.array(pts)
    cx, cy = np.median(pts, axis=0)
    r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    return pts, r


straight_pts, straight_r = scatter_at(T13["start_m"] - 45.0)
apex_pts, radial = scatter_at(T13["apex_m"])
print(f"approach-straight scatter: mean {straight_r.mean():.2f} m, "
      f"p90 {np.percentile(straight_r, 90):.2f} m")
print(f"apex scatter (drift-corrected): mean {radial.mean():.2f} m, "
      f"p90 {np.percentile(radial, 90):.2f} m, max {radial.max():.2f} m")

# centerline segment
cl = pd.read_parquet(corpus_dir() / f"{TRACK}_centerline.parquet")
cl = cl[(cl["track_dist_m"] >= LO) & (cl["track_dist_m"] <= HI)]
clx = (cl["long"].to_numpy() - lon0) * M_LON
cly = (cl["lat"].to_numpy() - lat0) * M_LAT

# highlight a fast and a slow lap
timed = [(k, lap_time.get(k, np.nan)) for k in segs]
timed = [(k, t) for k, t in timed if np.isfinite(t)]
timed.sort(key=lambda kt: kt[1])
fast_k, slow_k = timed[0][0], timed[-1][0]

fig, ax = plt.subplots(figsize=(13, 11))
for k, (x, y, _) in segs.items():
    ax.plot(x, y, color="0.45", lw=0.8, alpha=0.30, zorder=1)
ax.plot(clx, cly, color="black", lw=1.6, ls="--", zorder=3, label="centerline")
for k, color, lbl in [(fast_k, "#1f77b4", f"fastest {fast_k[1]} {timed[0][1]:.1f}s"),
                      (slow_k, "#d62728", f"slowest {slow_k[1]} {timed[-1][1]:.1f}s")]:
    x, y, _ = segs[k]
    ax.plot(x, y, color=color, lw=2.6, alpha=0.95, zorder=4, label=lbl)
ax.scatter([0], [0], s=320, marker="*", c="gold", edgecolors="black",
           linewidths=1.5, zorder=6, label="T13 apex pin")
ax.scatter(apex_pts[:, 0], apex_pts[:, 1], s=8, c="cyan", alpha=0.5, zorder=5,
           label="per-lap apex position")
ax.set_aspect("equal")
ax.set_xlabel("metres east of T13 apex pin")
ax.set_ylabel("metres north of T13 apex pin")
ax.set_title(f"T13 racing lines — {len(segs)} clean + reliable laps, drift-corrected GPS\n"
             f"apex scatter: mean {radial.mean():.1f} m / p90 {np.percentile(radial,90):.1f} m")
ax.grid(alpha=0.3)
ax.legend(loc="best", fontsize=9)
out = Path(__file__).parent / "t13_lines.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
