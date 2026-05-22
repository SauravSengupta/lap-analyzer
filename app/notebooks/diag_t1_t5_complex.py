"""Is T1-T5 one continuous WOT-to-WOT complex, and where are its real edges?

A 'complex' is a stretch with no sustained full throttle inside it. This finds
the WOT segments through the T1-T5 region on the top-decile laps and reports
where full throttle was last seen before T1 and first regained after T5.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from lap_analyzer.analysis import load_corpus, load_all_laps
from lap_analyzer.config import sessions_dir, tracks_dir

TRACK = "ridge"
WOT = 0.95          # throttle_norm at/above this is full throttle
MIN_WOT_RUN_M = 40  # a WOT segment must cover at least this much track

td = json.loads((tracks_dir() / f"{TRACK}.json").read_text(encoding="utf-8"))
corners = {c["id"]: c for c in td["corners"]}
for cid in ["T1", "T2", "T3", "T4", "T5"]:
    c = corners[cid]
    print(f"  {cid}: box [{c['start_m']:.0f}, {c['end_m']:.0f}]  apex {c['apex_m']:.0f}  {c['type']}")
T1, T5 = corners["T1"], corners["T5"]
print()

corpus = load_corpus(TRACK)
laps_csv = load_all_laps(TRACK)
clean = set(map(tuple, laps_csv[laps_csv["is_clean"].fillna(False).astype(bool)]
                [["session_id", "lap"]].astype({"lap": int}).itertuples(index=False, name=None)))
fast = corpus[(corpus["corner_id"] == "T1") & (corpus["lap_pace_decile"] == 0)
              & corpus["transit_reliable"]]
keys = sorted({(r.session_id, int(r.lap)) for r in fast.itertuples(index=False)} & clean)
print(f"{len(keys)} top-decile clean+reliable laps\n")

WIN_LO, WIN_HI = 150.0, T5["end_m"] + 350.0
by_session: dict[str, list[int]] = {}
for sid, lap in keys:
    by_session.setdefault(sid, []).append(lap)

entry, exit_, interior_wot = [], [], []
for sid, laps in by_session.items():
    try:
        df = pd.read_parquet(sessions_dir(TRACK) / sid / "samples.parquet")
    except Exception:
        continue
    for lap in laps:
        s = df[df["lap"] == lap].sort_values("track_dist_m")
        s = s[(s["track_dist_m"] >= WIN_LO) & (s["track_dist_m"] <= WIN_HI)]
        if len(s) < 20:
            continue
        x = s["track_dist_m"].to_numpy()
        wot = s["throttle_norm"].to_numpy() >= WOT
        # WOT runs
        runs = []
        i = 0
        while i < len(wot):
            if wot[i]:
                j = i
                while j + 1 < len(wot) and wot[j + 1]:
                    j += 1
                if x[j] - x[i] >= MIN_WOT_RUN_M:
                    runs.append((x[i], x[j]))
                i = j + 1
            else:
                i += 1
        # last WOT end before T1 apex; first WOT start after T5 apex
        before = [r for r in runs if r[1] <= T1["apex_m"]]
        after = [r for r in runs if r[0] >= T5["apex_m"]]
        if before:
            entry.append(before[-1][1])
        if after:
            exit_.append(after[0][0])
        # any WOT run that lives between T1 apex and T5 apex?
        for r0, r1 in runs:
            if r0 > T1["apex_m"] and r1 < T5["apex_m"]:
                interior_wot.append((sid, lap, r0, r1))


def med(v):
    return float(np.median(v)) if v else float("nan")


print("COMPLEX EDGES (median across top-decile laps):")
print(f"  full throttle ends (T1 entry) : {med(entry):6.0f} m")
print(f"  full throttle regained (exit) : {med(exit_):6.0f} m   "
      f"(T5 apex is {T5['apex_m']:.0f} m)")
print(f"  -> complex spans ~{med(entry):.0f} -> {med(exit_):.0f} m "
      f"({med(exit_) - med(entry):.0f} m long)\n")
print(f"sustained WOT runs found strictly inside T1-apex..T5-apex: {len(interior_wot)}")
if interior_wot:
    for sid, lap, r0, r1 in interior_wot[:8]:
        print(f"    {sid} L{lap}: {r0:.0f}-{r1:.0f} m")
else:
    print("    -> none. T1-T5 is a single continuous complex, as described.")
