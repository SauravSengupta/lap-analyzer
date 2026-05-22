"""Where does T1 actually begin vs where the ridge.json box says it does?

The box start_m is meant to be corner entry. This finds the physical entry
events (off-throttle, turn-in, min speed, lat-G peak) across clean laps and
compares them to the box.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from lap_analyzer.analysis import load_corpus, load_all_laps
from lap_analyzer.config import sessions_dir, tracks_dir

TRACK = "ridge"
td = json.loads((tracks_dir() / f"{TRACK}.json").read_text(encoding="utf-8"))
T1 = next(c for c in td["corners"] if c["id"] == "T1")
print(f"ridge.json T1 box: start={T1['start_m']}  apex={T1['apex_m']}  end={T1['end_m']}\n")

corpus = load_corpus(TRACK)
laps_csv = load_all_laps(TRACK)
clean = set(map(tuple, laps_csv[laps_csv["is_clean"].fillna(False).astype(bool)]
                [["session_id", "lap"]].astype({"lap": int}).itertuples(index=False, name=None)))
t1 = corpus[(corpus["corner_id"] == "T1") & corpus["transit_reliable"]]
keys = sorted({(r.session_id, int(r.lap)) for r in t1.itertuples(index=False)} & clean)
print(f"{len(keys)} clean + reliable T1 laps\n")

WIN_LO, WIN_HI = 200.0, 560.0
rows = []
by_session: dict[str, list[int]] = {}
for sid, lap in keys:
    by_session.setdefault(sid, []).append(lap)

for sid, laps in by_session.items():
    try:
        df = pd.read_parquet(sessions_dir(TRACK) / sid / "samples.parquet")
    except Exception:
        continue
    for lap in laps:
        s = df[df["lap"] == lap].sort_values("track_dist_m")
        s = s[(s["track_dist_m"] >= WIN_LO) & (s["track_dist_m"] <= WIN_HI)]
        if len(s) < 10:
            continue
        x = s["track_dist_m"].to_numpy()
        thr = s["throttle_norm"].to_numpy()
        latg = np.abs(s["lat_g"].to_numpy())
        spd = s["speed_mph"].to_numpy()
        # off-throttle: first time throttle drops out of WOT on the approach
        lift = x[np.argmax(thr < 0.80)] if (thr < 0.80).any() else np.nan
        # turn-in: first time lateral load builds past 0.3 g
        turnin = x[np.argmax(latg > 0.30)] if (latg > 0.30).any() else np.nan
        rows.append({
            "lift_m": lift, "turnin_m": turnin,
            "speed_at_lift": np.interp(lift, x, spd) if np.isfinite(lift) else np.nan,
            "speed_at_box_start": float(np.interp(T1["start_m"], x, spd)),
            "speed_at_apex": float(np.interp(T1["apex_m"], x, spd)),
            "min_speed": float(spd.min()),
        })

d = pd.DataFrame(rows)
t1c = t1.set_index(["session_id", "lap"])
print("PHYSICAL T1 ENTRY (median across laps):")
print(f"  off-throttle / lift : {d['lift_m'].median():6.0f} m   "
      f"(speed there {d['speed_at_lift'].median():.0f} mph)")
print(f"  turn-in (>0.3 g)    : {d['turnin_m'].median():6.0f} m")
print(f"  min_speed_dist_m    : {corpus.loc[t1.index, 'min_speed_dist_m'].median() + T1['apex_m']:6.0f} m "
      f"(corpus offset {corpus.loc[t1.index,'min_speed_dist_m'].median():+.0f} from apex)")
print(f"  latg_peak_dist_m    : {corpus.loc[t1.index, 'latg_peak_dist_m'].median() + T1['apex_m']:6.0f} m "
      f"(corpus offset {corpus.loc[t1.index,'latg_peak_dist_m'].median():+.0f} from apex)")
print()
print("BOX vs REALITY:")
print(f"  box start_m         : {T1['start_m']:6.0f} m   "
      f"(speed there median {d['speed_at_box_start'].median():.0f} mph)")
print(f"  -> box start is {T1['start_m'] - d['lift_m'].median():.0f} m LATE vs lift, "
      f"{T1['start_m'] - d['turnin_m'].median():.0f} m late vs turn-in")
print(f"  speed at apex       : {d['speed_at_apex'].median():.0f} mph")
print(f"  min speed in window : {d['min_speed'].median():.0f} mph")
print()
print("corpus T1 metrics (median):")
for c in ["entry_speed_mph", "min_speed_mph", "exit_speed_mph", "apex_speed_mph"]:
    if c in corpus.columns:
        print(f"  {c:18s}: {corpus.loc[t1.index, c].median():.1f}")
