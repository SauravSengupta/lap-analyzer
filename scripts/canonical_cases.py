"""Item-7 canonical-case table (plan sec 7.3) ON THE ESTIMATOR.

Runs the nine canonical laps through estimate_trajectory + section_timing and
checks each against its expected value-range and tier. Then the two-sided
bandwidth acceptance (sec 9.2 / item 7): the T11 real tight-line lap must WIN T11
by a margin >= 0.4s among rankable laps, AND all four T8 fakes must be below
tier A. Local script (real data is gitignored/absent on CI) — not a pytest.

Run: python scripts/canonical_cases.py
"""
from __future__ import annotations

import json
import sys

import numpy as np

from lap_analyzer.analysis import load_centerline, load_samples, section_bounds
from lap_analyzer.config import sessions_dir
from lap_analyzer.gates import TrackFrame
from lap_analyzer.trajectory import (
    estimate_trajectory, load_corridor, section_timing,
)

CORR = load_corridor("ridge")
FRAME = TrackFrame.from_centerline(load_centerline("ridge"))
BOUNDS = section_bounds(json.load(open("tracks/ridge.json")))

# case, session, lap, corner, (lo, hi) expected value, expected tier predicate
CASES = [
    ("T8 L4 fake #1 (7.07)", "20260702-101432", 4, "T8", (8.0, 8.6), "not_A"),
    ("T8 L6 on-ribbon fake (7.37)", "20260702-101432", 6, "T8", (7.4, 8.5), "not_A"),
    ("T8 L3 mild fake (7.76)", "20260702-101432", 3, "T8", (7.7, 8.4), "not_A"),
    ("T8 L2 glitched red (8.22)", "20260702-101432", 2, "T8", (8.0, 9.0), "C_wide"),
    ("T8 L1 clean slow (9.48)", "20260702-101432", 1, "T8", (9.3, 9.5), "A"),
    ("T8 honest fastest (8.31)", "20260702-091257", 5, "T8", (8.25, 8.45), "A"),
    ("T11 real tight line", "20250725-160443", 3, "T11", (10.9, 11.6), "A"),
    ("T3 ghost (39.79)", "20251018-100026", 6, "T3", (4.9, 5.5), "any"),
    ("T11 ghost-session L5 (38.44)", "20251018-100026", 5, "T11", (11.5, 13.5), "any"),
]


def _time_one(sid: str, lap: int, corner: str):
    s = load_samples("ridge", sid)
    g = s[s["lap"] == lap]
    if g.empty:
        return None
    traj = estimate_trajectory(g, CORR, FRAME)
    a, b = BOUNDS[corner]
    return section_timing(traj, a, b, CORR)


def _check_tier(pred: str, st) -> bool:
    if pred == "any":
        return True
    if pred == "A":
        return st.tier == "A"
    if pred == "not_A":
        return st.tier != "A" and not st.rank_eligible
    if pred == "C_wide":
        return st.tier == "C" and st.sigma_s >= 0.25
    return False


def run_canonical() -> bool:
    print("=== Canonical-case table (sec 7.3) on the estimator ===")
    ok = True
    for label, sid, lap, corner, (lo, hi), pred in CASES:
        st = _time_one(sid, lap, corner)
        if st is None:
            print(f"  MISSING: {label}")
            ok = False
            continue
        val_ok = lo <= st.time_s <= hi
        tier_ok = _check_tier(pred, st)
        row_ok = val_ok and tier_ok
        ok &= row_ok
        flag = "OK " if row_ok else "!! "
        print(f"  {flag}{label:32s} {st.time_s:6.2f}s sig {st.sigma_s:5.2f} "
              f"tier {st.tier} rank {int(st.rank_eligible)} driven {st.driven_m:6.1f}  "
              f"[want {lo}-{hi}s, {pred}]"
              + ("" if val_ok else "  VALUE-FAIL")
              + ("" if tier_ok else "  TIER-FAIL"))
    return ok


def t11_leaderboard():
    """Rankable T11 times across the whole ridge corpus: winner + margin to 2nd."""
    a, b = BOUNDS["T11"]
    rows = []
    for sd in sorted(sessions_dir("ridge").iterdir()):
        sp = sd / "samples.parquet"
        if not sp.exists():
            continue
        try:
            s = load_samples("ridge", sd.name)
        except Exception:
            continue
        for lap, g in s.groupby("lap"):
            if len(g) < 100:
                continue
            try:
                traj = estimate_trajectory(g, CORR, FRAME)
                st = section_timing(traj, a, b, CORR)
            except Exception:
                continue
            if st.rank_eligible and np.isfinite(st.time_s):
                rows.append((st.time_s, sd.name, int(lap), st.sigma_s, st.driven_m))
    rows.sort()
    print("\n=== T11 rankable leaderboard (corpus) ===")
    for tm, sid, lap, sg, dv in rows[:6]:
        print(f"  {tm:6.2f}s  sig {sg:5.2f}  {sid} L{lap}  driven {dv:.1f}m")
    if len(rows) < 2:
        print("  <2 rankable T11 laps — cannot measure margin")
        return None
    winner, runner = rows[0], rows[1]
    margin = runner[0] - winner[0]
    is_real = winner[1] == "20250725-160443" and winner[2] == 3
    print(f"  winner: {winner[1]} L{winner[2]}  margin to 2nd = {margin:.2f}s  "
          f"(real tight-line winner: {is_real})")
    return dict(margin=margin, is_real=is_real, winner=winner)


def main() -> int:
    table_ok = run_canonical()
    lb = t11_leaderboard()

    print("\n=== Item-7 acceptance (sec 9.2, BANDWIDTH_M=60) ===")
    # all four T8 fakes below tier A
    fakes_ok = True
    for label, sid, lap, corner, _r, pred in CASES:
        if pred in ("not_A", "C_wide"):
            st = _time_one(sid, lap, corner)
            fake_below_A = st is not None and st.tier != "A" and not st.rank_eligible
            fakes_ok &= fake_below_A
            print(f"  T8 fake {sid} L{lap}: tier {st.tier if st else '?'} "
                  f"below-A {fake_below_A}")
    real_wins = lb is not None and lb["is_real"]
    print(f"\n  all four T8 fakes below tier A : {fakes_ok}")
    print(f"  T11 real tight-line lap WINS    : {real_wins}")

    # The >=0.4s win-margin is a DOCUMENTED, NON-GATING diagnostic (reserved
    # decision 2026-07-12: "Accept & document"). Bandwidth is not the lever —
    # the margin is flat ~0.11s across BW 25..60; it is capped by borderline
    # Mode-4 laps in 20250726-140555 (slated gps_unreliable in PR5), NOT by the
    # tricube bandwidth the sec 9.2 premise assumed. No band/tier/BW change.
    if lb is not None:
        print(f"  [diagnostic, non-gating] T11 win-margin to 2nd = {lb['margin']:.2f}s "
              f"(target >=0.4s unmet-by-corpus; bandwidth is not the lever)")

    passed = table_ok and fakes_ok and real_wins
    print(f"\n{'PASS' if passed else 'FAIL'}: canonical={table_ok} fakes={fakes_ok} "
          f"real_wins_T11={real_wins}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
