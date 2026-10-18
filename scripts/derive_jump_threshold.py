"""Derive JUMP_THRESH_M by its rule (spec 2026-10-10 §W1): clean-lap residual p99.9
+ 1 m, over CALIBRATION-split 8 Hz OBD laps (lap_reliable_traj). Prints the replacement
source line for trajectory.py. Local; needs real data/.

  $env:PYTHONPATH = "."; python scripts/derive_jump_threshold.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_checks as vc  # noqa: E402

from lap_analyzer.trajectory import MPH_TO_MPS, _jump_residuals  # noqa: E402

RULE_PCTL = 99.9
RULE_MARGIN_M = 1.0


def main() -> int:
    sessions = pd.concat([vc.list_sessions(t) for t in ("pir", "ridge")], ignore_index=True)
    split = vc.session_split(sessions)
    res_all, n_laps = [], 0
    for tr in ("pir", "ridge"):
        ctx = vc.TrackCtx.load(tr)
        cal = [s for (t, s), v in split.items() if t == tr and v == "cal"]
        for _sid, _lap, g in vc.clean_laps(ctx, cal):
            v = g["speed_mph"].to_numpy(float) * MPH_TO_MPS
            jr = _jump_residuals(g, ctx.frame, v, has_obd=True)
            if jr is None:
                continue
            res_all.append(jr[1][np.isfinite(jr[1])])
            n_laps += 1
    r = np.concatenate(res_all)
    p = float(np.percentile(r, RULE_PCTL))
    thr = round(p + RULE_MARGIN_M, 1)
    print(f"laps={n_laps} fixes={len(r)} p50={np.median(r):.2f} p99={np.percentile(r, 99):.2f} p99.9={p:.2f}")
    print("Replacement line for lap_analyzer/trajectory.py:")
    print(f"JUMP_THRESH_M = {thr}  # {date.today()}, rule p{RULE_PCTL}+{RULE_MARGIN_M} m of clean residuals "
          f"(calibration split: {n_laps} laps, {len(r)} fixes, p99.9 = {p:.2f} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
