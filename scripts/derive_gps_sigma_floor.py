"""Derive SIGMA_FLOOR_GPS_M by its rule (spec 2026-10-10 §W3): the smallest floor in a
0.5 m grid at which the GPS-only emulation (V3) is ≥ 95% within 2σ of the OBD-backed
answer on CALIBRATION-split OBD laps. Prints the replacement line. Local; needs data/.

  $env:PYTHONPATH = "."; python scripts/derive_gps_sigma_floor.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_checks as vc  # noqa: E402

import lap_analyzer.trajectory as T  # noqa: E402

GRID = np.arange(1.0, 6.01, 0.5)
TARGET = 0.95


def main() -> int:
    sessions = pd.concat([vc.list_sessions(t) for t in ("pir", "ridge")], ignore_index=True)
    split = vc.session_split(sessions)
    ctxs = {tr: vc.TrackCtx.load(tr) for tr in ("pir", "ridge")}
    laps = {tr: list(vc.clean_laps(ctxs[tr], [s for (t, s), v in split.items() if t == tr and v == "cal"]))
            for tr in ctxs}
    old = T.SIGMA_FLOOR_GPS_M
    chosen, rows = None, []
    try:
        for floor in GRID:
            T.SIGMA_FLOOR_GPS_M = float(floor)
            v3 = pd.concat([vc.run_v3(ctxs[tr], laps[tr]) for tr in ctxs], ignore_index=True)
            a = v3[v3["rank_obd"] & np.isfinite(v3["t_gps"])]
            d = (a["t_gps"] - a["t_obd"]).abs()
            cov = float((d <= 2.0 * a["sigma_gps"]).mean())
            tier_a = a[a["rank_gps"]]
            accA = float(((tier_a["t_gps"] - tier_a["t_obd"]).abs() <= 0.10).mean()) if len(tier_a) else float("nan")
            rows.append((floor, len(a), cov, float(a["rank_gps"].mean()), accA))
            print(f"floor={floor:.1f} n={len(a)} within2σ={cov:.3f} tierA_share={rows[-1][3]:.3f} tierA≤0.10={accA:.3f}")
            if chosen is None and cov >= TARGET:
                chosen = float(floor)
                break
    finally:
        T.SIGMA_FLOOR_GPS_M = old
    if chosen is None:
        print("No floor in the grid reaches the target — STOP and discuss (spec W3).")
        return 1
    print("Replacement line for lap_analyzer/trajectory.py:")
    print(f"SIGMA_FLOOR_GPS_M = {chosen}  # {date.today()}, smallest floor with GPS-only emulation ≥95% within 2σ "
          f"(calibration split, {rows[-1][1]} transits, W3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
