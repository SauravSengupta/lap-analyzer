"""Derive ACCEL_SPEED_TOL_MPS by its rule (W3 amendment, user decision 2026-10-22).

Rule (fixed before looking at any outcome): ACCEL_SPEED_TOL_MPS = p99 of
|accel_speed_resid_mps| over every status=='ok' section of the GPS-only emulation
(validation_checks.gps_only_emulation) of clean CALIBRATION-split OBD laps (both
tracks), rounded UP to 0.1 m/s. The residual is read from section_timing(...).checks —
the production code path — so the rule measures exactly what production checks. By
construction ~1% of clean GPS-only sections lose tier A. Prints the replacement line.
Local; needs data/.

  $env:PYTHONPATH = "."; python scripts/derive_accel_speed_tol.py
"""
from __future__ import annotations

import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_checks as vc

from lap_analyzer.config import sessions_dir
from lap_analyzer.trajectory import estimate_trajectory, section_timing

TRACKS = ("pir", "ridge")
CHECK = "accel_speed_resid_mps"
PROBE = ("pir", "20260926-111509", 2, "T1")   # confirmed fake (final W3 review); sanity only


def p99_ceil(values) -> float:
    """p99 of |values| (non-finite ignored), rounded UP to the 0.1 grid. A value already
    on the grid is not bumped (the 1e-9 guard absorbs float noise in p·10)."""
    a = np.abs(np.asarray(values, dtype=float))
    a = a[np.isfinite(a)]
    return math.ceil(float(np.percentile(a, 99)) * 10.0 - 1e-9) / 10.0


def section_resid(traj, a: float, b: float, corridor) -> tuple[str, float]:
    """(section status, accel_speed_resid_mps) as section_timing records it."""
    st = section_timing(traj, a, b, corridor)
    rec = next(c for c in st.checks if c["name"] == CHECK)
    return st.status, float(rec["value"])


def collect(ctx, laps) -> pd.DataFrame:
    rows = []
    for sid, lap, g in laps:
        traj = estimate_trajectory(vc.gps_only_emulation(g), ctx.corridor, ctx.frame)
        for cid, (a, b) in ctx.bounds.items():
            status, r = section_resid(traj, a, b, ctx.corridor)
            if status == "ok":
                rows.append({"track": ctx.track, "session_id": sid, "lap": lap, "corner_id": cid, "resid": r})
    return pd.DataFrame(rows, columns=["track", "session_id", "lap", "corner_id", "resid"])


def probe(ctxs) -> None:
    track, sid, lap, cid = PROBE
    sp = sessions_dir(track) / sid / "samples.parquet"
    if not sp.exists():
        print(f"probe {track} {sid} L{lap} {cid}: session not under data/ — skipped")
        return
    s = vc._read_traj_samples(sp)
    g = vc._lap_frame(s, lap) if s is not None else None
    if g is None or len(g) < 2:
        print(f"probe {track} {sid} L{lap} {cid}: lap missing — skipped")
        return
    ctx = ctxs[track]
    traj = estimate_trajectory(g, ctx.corridor, ctx.frame)   # the real lap, not emulated
    a, b = ctx.bounds[cid]
    st = section_timing(traj, a, b, ctx.corridor)
    rec = next(c for c in st.checks if c["name"] == CHECK)
    print(f"probe {track} {sid} L{lap} {cid} (real lap, status={traj.status}): "
          f"{CHECK}={rec['value']} m/s, tier={st.tier}, time={st.time_s:.3f}s")


def main() -> int:
    sessions = pd.concat([vc.list_sessions(t) for t in TRACKS], ignore_index=True)
    split = vc.session_split(sessions)
    ctxs = {tr: vc.TrackCtx.load(tr) for tr in TRACKS}
    parts = []
    for tr, ctx in ctxs.items():
        cal = [s for (t, s), v in split.items() if t == tr and v == "cal"]
        parts.append(collect(ctx, vc.clean_laps(ctx, cal)))
    df = pd.concat(parts, ignore_index=True)
    r = df["resid"].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    if not len(r):
        print("No evaluable sections — STOP and discuss.")
        return 1
    a = np.abs(r)
    print(f"n sections (status ok, finite resid) = {len(r)}  (non-finite dropped: {len(df) - len(r)})")
    print("|resid| m/s: median={:.3f} p90={:.3f} p99={:.3f} p99.9={:.3f}".format(
        *np.percentile(a, [50, 90, 99, 99.9])))
    print(f"signed resid median = {np.median(r):+.3f} m/s")
    for tr, sub in df.groupby("track"):
        x = sub["resid"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x):
            print(f"  {tr}: n={len(x)} p99|resid|={np.percentile(np.abs(x), 99):.3f}")
    tol = p99_ceil(r)
    probe(ctxs)
    print("Replacement line for lap_analyzer/trajectory.py:")
    print(f"ACCEL_SPEED_TOL_MPS = {tol}  # {datetime.now(UTC).date()}, p99 |GPS-speed Δv − ∫long_g| on GPS-only "
          f"emulation of clean calibration OBD laps ({len(r)} sections, W3 amendment)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
