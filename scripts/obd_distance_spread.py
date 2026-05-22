"""How stable is OBD-integrated lap distance across all laps?

dist_m (samples.parquet) is the raw trapezoidal integration of OBD speed, never
rescaled. dist_lap_m is dist_m zeroed per lap THEN rescaled to the 3962m
canonical length. So raw OBD lap distance = per-lap span of dist_m.

This asks: if OBD distance were a trustworthy ruler, every full lap would
integrate to ~the same length. How big is the spread?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lap_analyzer.analysis import load_all_laps, load_samples
from lap_analyzer.config import sessions_dir

TRACK = "ridge"
CANON = 3962.0

laps_csv = load_all_laps(TRACK)
clean_keys = set(map(tuple, laps_csv[laps_csv["is_clean"].fillna(False).astype(bool)]
                     [["session_id", "lap"]].astype({"lap": int}).itertuples(index=False, name=None)))

rows = []
sess_root = sessions_dir(TRACK)
for sdir in sorted(p for p in sess_root.iterdir() if p.is_dir()):
    sid = sdir.name
    try:
        df = pd.read_parquet(sdir / "samples.parquet")
    except Exception:
        continue
    if "dist_m" not in df.columns:
        print(f"{sid}: no dist_m column"); continue
    has_gps = "track_dist_m" in df.columns
    for lap, g in df.groupby("lap"):
        lap = int(lap)
        g = g.sort_values("t")
        obd_dist = float(g["dist_m"].iloc[-1] - g["dist_m"].iloc[0])
        # GPS jitter: track_dist_m must rise monotonically (distance can't shrink).
        # Trim to the in-lap portion first so the start/finish wrap isn't counted.
        gps_backwalk = float("nan")
        if has_gps:
            tg = g["track_dist_m"].to_numpy()
            small = np.where(tg < 500)[0]
            if small.size:
                tg = tg[small[0]:]
            if tg.size > 1:
                gps_backwalk = float(-np.minimum(np.diff(tg), 0).sum())
        rows.append({
            "session_id": sid, "lap": lap,
            "obd_lap_dist_m": obd_dist,
            "gps_backwalk_m": gps_backwalk,
            "is_clean": (sid, lap) in clean_keys,
            "n_samples": len(g),
        })

d = pd.DataFrame(rows)
print(f"total laps scanned: {len(d)}\n")

# Full laps only: clean, and OBD distance within 20% of canonical (drops
# out/in-laps and OBD-dropout laps that integrate to near-zero).
full = d[d["is_clean"] & (d["obd_lap_dist_m"].between(0.8 * CANON, 1.2 * CANON))].copy()
print(f"clean full laps (OBD dist within 20% of {CANON:.0f}m): {len(full)}\n")

o = full["obd_lap_dist_m"]
print("=== RAW OBD-INTEGRATED LAP DISTANCE (the actual answer) ===")
print(f"  min    = {o.min():8.1f} m")
print(f"  max    = {o.max():8.1f} m")
print(f"  median = {o.median():8.1f} m")
print(f"  mean   = {o.mean():8.1f} m   std = {o.std():.1f} m")
print(f"  >>> biggest difference (max - min) = {o.max() - o.min():.1f} m "
      f"({100 * (o.max() - o.min()) / o.median():.2f}% of median lap)")
print(f"  p5..p95 spread = {o.quantile(0.95) - o.quantile(0.05):.1f} m "
      f"({100 * (o.quantile(0.95) - o.quantile(0.05)) / o.median():.2f}%)")
print(f"  per-lap deviation from median: "
      f"mean abs = {(o - o.median()).abs().mean():.1f} m, "
      f"std = {o.std():.1f} m ({100 * o.std() / o.median():.2f}%)\n")

print("10 laps with the most extreme OBD distance:")
ext = full.reindex((full["obd_lap_dist_m"] - o.median()).abs()
                   .sort_values(ascending=False).index)
for r in ext.head(10).itertuples(index=False):
    print(f"  {r.session_id} L{r.lap:<3d} obd_dist={r.obd_lap_dist_m:8.1f} m "
          f"({r.obd_lap_dist_m - o.median():+7.1f} vs median)  "
          f"gps_backwalk={r.gps_backwalk_m:7.1f} m")

print("\n=== FOR CONTRAST: GPS track_dist_m backward-walk on the SAME laps ===")
print("(track_dist_m must only rise; any backward step is pure GPS noise)")
g = full["gps_backwalk_m"].dropna()
print(f"  total backward walk per lap:")
print(f"    median = {g.median():.1f} m   mean = {g.mean():.1f} m")
print(f"    p95 = {g.quantile(0.95):.1f} m   max = {g.max():.1f} m")
print(f"  laps with >10m of backward walk: {int((g > 10).sum())} / {len(g)} "
      f"({100 * (g > 10).mean():.1f}%)")
print(f"  laps with >50m of backward walk: {int((g > 50).sum())} / {len(g)} "
      f"({100 * (g > 50).mean():.1f}%)")
