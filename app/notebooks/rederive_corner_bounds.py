"""Re-derive corner bounds as WOT-to-WOT complexes with tiled corners.

A 'complex' is a run of corners sharing no sustained full throttle — found by
checking, for each consecutive corner pair, whether laps get back to WOT between
the two apexes. Inside a complex the corner boxes tile; only the complex edges
touch a real off-throttle point (entry) and a real WOT point (exit).

Interior boundaries:
  - opposite-direction corners -> the lat-G neutral point between the apexes
  - same-direction corners     -> the apex-to-apex midpoint

Complex edges:
  - entry = early percentile of the off-throttle point (so even early brakers
    are inside the window); exit = late percentile of the WOT-resume point.
  - between two adjacent complexes the percentile edges can reach past each
    other across a short straight; that boundary is clamped to the mid-straight
    point so boxes never overlap.

apex_m is left untouched (Maps-pin calibrated). Prints current vs proposed
start_m/end_m for every corner. Writes nothing — this is the review step.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lap_analyzer.analysis import load_all_laps, load_corpus
from lap_analyzer.config import sessions_dir, tracks_dir

TRACK = "ridge"
WOT = 0.95             # throttle_norm >= this is full throttle
MIN_WOT_RUN_M = 30.0   # a WOT segment must cover at least this much track
LATG_STRAIGHT_G = 0.20 # |lat_g| at/below this == the car is essentially straight
SETTLE_M = 30.0        # ... and it must stay straight this far for the corner to be 'over'


def drop_gps_glitches(s: pd.DataFrame) -> pd.DataFrame:
    s = s.sort_values("t").reset_index(drop=True)
    sm = s["track_dist_m"] < 500
    if sm.any():
        s = s.iloc[sm.idxmax():].reset_index(drop=True)
    s = s[(s["track_dist_m"] - s["dist_lap_m"]).abs() < 50].reset_index(drop=True)
    rmax = s["track_dist_m"].cummax()
    return s[s["track_dist_m"] >= rmax - 1.0].reset_index(drop=True)


def wot_runs(x: np.ndarray, thr: np.ndarray) -> list[tuple[float, float]]:
    """Sustained full-throttle runs as (lo_m, hi_m)."""
    wot = thr >= WOT
    runs, i, n = [], 0, len(wot)
    while i < n:
        if wot[i]:
            j = i
            while j + 1 < n and wot[j + 1]:
                j += 1
            if x[j] - x[i] >= MIN_WOT_RUN_M:
                runs.append((float(x[i]), float(x[j])))
            i = j + 1
        else:
            i += 1
    return runs


td = json.loads((tracks_dir() / f"{TRACK}.json").read_text(encoding="utf-8"))
corners = sorted(td["corners"], key=lambda c: c["apex_m"])
ids = [c["id"] for c in corners]
apex = {c["id"]: float(c["apex_m"]) for c in corners}
ctype = {c["id"]: c["type"] for c in corners}
cur = {c["id"]: (float(c["start_m"]), float(c["end_m"])) for c in td["corners"]}

corpus = load_corpus(TRACK)
laps_csv = load_all_laps(TRACK)
clean = set(map(tuple, laps_csv[laps_csv["is_clean"].fillna(False).astype(bool)]
                [["session_id", "lap"]].astype({"lap": int})
                .itertuples(index=False, name=None)))
fast = (corpus[(corpus["lap_pace_decile"] == 0) & corpus["lap_reliable"]]
        [["session_id", "lap"]].drop_duplicates())
keys = sorted({(r.session_id, int(r.lap)) for r in fast.itertuples(index=False)} & clean)

laps_s: list[pd.DataFrame] = []
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
        if len(s) > 100:
            laps_s.append(s.sort_values("track_dist_m").reset_index(drop=True))
lap_runs = [wot_runs(s["track_dist_m"].to_numpy(), s["throttle_norm"].to_numpy())
            for s in laps_s]
print(f"{len(laps_s)} top-decile clean + reliable laps\n")

# --- Step 1: group corners into complexes -----------------------------------
def wot_between(runs, lo, hi) -> bool:
    return any(r0 >= lo and r1 <= hi for r0, r1 in runs)

complexes: list[list[str]] = []
cur_cx = [ids[0]]
for k in range(len(ids) - 1):
    a, b = apex[ids[k]], apex[ids[k + 1]]
    frac = float(np.mean([wot_between(r, a, b) for r in lap_runs]))
    if frac > 0.5:
        complexes.append(cur_cx); cur_cx = [ids[k + 1]]
    else:
        cur_cx.append(ids[k + 1])
complexes.append(cur_cx)

print("COMPLEXES (chained corners share no sustained full throttle):")
for cx in complexes:
    print(f"  {' - '.join(cx)}")
print()

# --- Step 2: complex edges --------------------------------------------------
# Entry: the typical (median) off-throttle point — median is robust to the few
# early-lift laps (traffic, cautious entries) that dragged an extreme percentile.
# Exit: the corner is over when the car is back to WOT AND settled straight
# (lat-G neutral) — whichever is later. On a corner you accelerate through, that
# lands well past WOT-resume.
def median_or_nan(vals):
    return float(np.median(vals)) if len(vals) else float("nan")

def complex_entry(first_apex):
    vals = [max(r1 for r0, r1 in runs if r1 <= first_apex)
            for runs in lap_runs if any(r1 <= first_apex for r0, r1 in runs)]
    return median_or_nan(vals)

def first_sustained_below(x, v, thresh, settle_m):
    """First x at which v stays below thresh for at least settle_m; else x[-1]."""
    below = v < thresh
    i, n = 0, len(v)
    while i < n:
        if below[i]:
            j = i
            while j + 1 < n and below[j + 1]:
                j += 1
            if x[j] - x[i] >= settle_m:
                return float(x[i])
            i = j + 1
        else:
            i += 1
    return float(x[-1]) if n else float("nan")

def complex_exit(last_apex, limit):
    """Median 'corner over' point after the last apex: the car is back to WOT
    AND settled straight. The search is bounded by `limit` (the next complex's
    entry) so it can't run into the following complex's corners."""
    hi = limit if limit is not None else last_apex + 1500.0
    vals = []
    for s, runs in zip(laps_s, lap_runs):
        after = [r0 for r0, r1 in runs if last_apex <= r0 <= hi]
        wot_pt = min(after) if after else last_apex
        seg = s[(s["track_dist_m"] >= last_apex) & (s["track_dist_m"] <= hi)]
        if len(seg) < 5:
            continue
        x = seg["track_dist_m"].to_numpy()
        lg = seg["lat_g"].rolling(7, center=True, min_periods=1).median().abs().to_numpy()
        straight_pt = first_sustained_below(x, lg, LATG_STRAIGHT_G, SETTLE_M)
        vals.append(min(max(wot_pt, straight_pt), hi))
    return median_or_nan(vals)

# Entries first (independent), then exits bounded by the next complex's entry.
entries = [complex_entry(apex[cx[0]]) for cx in complexes]
exits = []
for ci, cx in enumerate(complexes):
    limit = entries[ci + 1] if ci + 1 < len(complexes) else None
    exits.append(complex_exit(apex[cx[-1]], limit))
# exits are pre-bounded by the next entry, so adjacent complexes never overlap.
final_entry = {ci: entries[ci] for ci in range(len(complexes))}
final_exit = {ci: exits[ci] for ci in range(len(complexes))}

# --- Step 3: interior boundaries + tile -------------------------------------
def latg_neutral(lo, hi):
    """Median position of minimum |lat_g| between two apexes — the direction-change point."""
    vals = []
    for s in laps_s:
        seg = s[(s["track_dist_m"] > lo) & (s["track_dist_m"] < hi)]
        if len(seg) < 5:
            continue
        lg = seg["lat_g"].rolling(5, center=True, min_periods=1).median().abs().to_numpy()
        vals.append(float(seg["track_dist_m"].to_numpy()[int(np.argmin(lg))]))
    return float(np.median(vals)) if vals else (lo + hi) / 2

proposed: dict[str, tuple[float, float]] = {}
for ci, cx in enumerate(complexes):
    bounds = [final_entry[ci]]
    notes = []
    for k in range(len(cx) - 1):
        n, m = cx[k], cx[k + 1]
        if ctype[n] != ctype[m]:
            b = latg_neutral(apex[n], apex[m]); notes.append(f"{n}|{m} lat-G")
        else:
            b = (apex[n] + apex[m]) / 2; notes.append(f"{n}|{m} midpoint")
        bounds.append(b)
    bounds.append(final_exit[ci])
    for k, cid in enumerate(cx):
        proposed[cid] = (round(bounds[k], 1), round(bounds[k + 1], 1))
    print(f"complex {' - '.join(cx)}:")
    print(f"  entry {final_entry[ci]:.0f} m  (median off-throttle)")
    print(f"  exit  {final_exit[ci]:.0f} m  (median straight-and-WOT)")
    if notes:
        print(f"  interior: {', '.join(notes)}")
print()

# --- review table -----------------------------------------------------------
print(f"{'corner':7}{'type':7}{'current':>15}{'proposed':>15}"
      f"{'start chg':>11}{'end chg':>10}")
for cid in ids:
    cs, ce = cur[cid]
    ps, pe = proposed[cid]
    print(f"{cid:7}{ctype[cid]:7}{f'[{cs:.0f},{ce:.0f}]':>15}"
          f"{f'[{ps:.0f},{pe:.0f}]':>15}{ps - cs:>+11.0f}{pe - ce:>+10.0f}")

# --- save proposed bounds for the plot + apply steps ------------------------
out_path = Path(__file__).parent / "proposed_corner_bounds.json"
out_path.write_text(json.dumps({
    "track": TRACK,
    "complexes": complexes,
    "bounds": {cid: {"start_m": proposed[cid][0], "end_m": proposed[cid][1]}
               for cid in ids},
}, indent=2), encoding="utf-8")
print(f"\nwrote {out_path}")
