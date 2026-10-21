"""Re-derive the trust-net constants by their ORIGINAL rules on lag-corrected,
jump-filtered evidence, CALIBRATION split only (spec 2026-10-10 §W2). Prints
replacement source lines; never chooses values to hit a gate.

Rules (each the documented definition in trajectory.py):
  RHO_SECTION          Fisher-z (n−3)-weighted mean, over corners, of the across-lap
                       correlation of δ = track_dist − dist_lap at each section's two gates.
  DRIVEN_BAND_*        per corner robust bound of dev = driven − (b−a) on clean transits
                       (|median| + 3·1.4826·MAD); least-squares line vs corridor heading
                       ∫|κ|ds → (scale, floor).
  CONSISTENCY_STD_M    1.4826·MAD of the consistency residual, clean transits, pooled.
  SIGMA_PRIOR_MID_M    1.4826·MAD of δ̂ at mid-lap knots (≥ PRIOR_TAPER_M from both ends):
                       how far the OBD backbone wanders from GPS mid-lap.
Clean = lap_reliable_traj OBD laps at ≥ 3 Hz fixes (validation_checks.clean_laps). Caveat:
clean laps are tier-A under the CURRENT nets, so the dev/resid samples are truncated near
the current bands (robust stats handle contamination, not truncation); a second pass
after applying the constants shows the drift. Production centres both the consistency and
prior nets on 0, so the printed medians must be near 0.

  $env:PYTHONPATH = "."; python scripts/derive_trust_constants.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

MAD_K = 1.4826
BOUND_SIGMAS = 3.0
GATE_WIN_M = 10.0


def robust_std(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(MAD_K * np.median(np.abs(x - np.median(x))))


def robust_bound(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(abs(np.median(x)) + BOUND_SIGMAS * robust_std(x))


def fisher_z_mean(rs, ns) -> float:
    rs, ns = np.asarray(rs, dtype=float), np.asarray(ns, dtype=float)
    ok = np.isfinite(rs) & (ns > 3) & (np.abs(rs) < 1)
    z = np.arctanh(rs[ok])
    w = ns[ok] - 3
    return float(np.tanh(np.sum(w * z) / np.sum(w)))


def fit_band(heading_rad, bound_m) -> tuple[float, float]:
    scale, floor = np.polyfit(np.asarray(heading_rad, float), np.asarray(bound_m, float), 1)
    return float(scale), float(floor)


def _collect():
    import validation_checks as vc

    from lap_analyzer.trajectory import (
        PRIOR_TAPER_M,
        _section_heading_rad,
        estimate_trajectory,
        section_timing,
    )
    sessions = pd.concat([vc.list_sessions(t) for t in ("pir", "ridge")], ignore_index=True)
    split = vc.session_split(sessions)
    trans, gates, prior = [], [], []
    for tr in ("pir", "ridge"):
        ctx = vc.TrackCtx.load(tr)
        cal = [s for (t, s), v in split.items() if t == tr and v == "cal"]
        for sid, lap, g in vc.clean_laps(ctx, cal):
            traj = estimate_trajectory(g, ctx.corridor, ctx.frame)
            td = g["track_dist_m"].to_numpy(float)
            delta = td - g["dist_lap_m"].to_numpy(float)
            dl = traj.dl
            mid = (traj.knots >= traj.knots.min() + PRIOR_TAPER_M) & (traj.knots <= traj.knots.max() - PRIOR_TAPER_M)
            prior.append(np.interp(traj.knots[mid], dl, traj.delta_hat))
            for cid, (a, b) in ctx.bounds.items():
                st = section_timing(traj, a, b, ctx.corridor)
                if st.status != "ok":
                    continue
                ck = {c["name"]: c["value"] for c in st.checks}
                trans.append(dict(track=tr, corner=cid, dev=ck["driven_band_dev_m"],
                                  resid=ck["consistency_resid_m"],
                                  heading=_section_heading_rad(ctx.corridor, a, b)))
                # gates a,b are ruler (track_dist) positions; dist_lap differs by δ, so window on td
                da = delta[traj.evidence & (np.abs(td - a) < GATE_WIN_M)]
                db = delta[traj.evidence & (np.abs(td - b) < GATE_WIN_M)]
                if len(da) and len(db):
                    gates.append(dict(track=tr, corner=cid, sid=sid, lap=lap,
                                      da=float(np.median(da)), db=float(np.median(db))))
    return pd.DataFrame(trans), pd.DataFrame(gates), np.concatenate(prior)


def main() -> int:
    trans, gates, prior = _collect()
    rs, ns = [], []
    for _, g in gates.groupby(["track", "corner"]):
        if len(g) > 3:
            rs.append(float(np.corrcoef(g["da"], g["db"])[0, 1]))
            ns.append(len(g))
    rho = fisher_z_mean(rs, ns)
    per = trans.groupby(["track", "corner"]).agg(bound=("dev", robust_bound), heading=("heading", "first"))
    scale, floor = fit_band(per["heading"], per["bound"])
    cons = robust_std(trans["resid"])
    pm = robust_std(prior)
    cons_med = float(np.nanmedian(trans["resid"]))
    pm_med = float(np.nanmedian(prior))
    r_a, n_a = np.asarray(rs, float), np.asarray(ns, float)
    n_rho = int(np.sum(np.isfinite(r_a) & (n_a > 3) & (np.abs(r_a) < 1)))
    d = date.today()
    n_t, n_c = len(trans), len(per)
    print(f"transits={n_t} corners={n_c} gate-pairs={len(gates)} prior-knots={len(prior)}")
    print(per.round(2).to_string())
    print(f"consistency resid: median {cons_med:+.2f} m, robust std {cons:.2f} m")
    print(f"mid-lap delta-hat: median {pm_med:+.2f} m, robust std {pm:.2f} m")
    print("\nReplacement lines for lap_analyzer/trajectory.py:")
    print(f"RHO_SECTION = {rho:.2f}  # {d}, Fisher-z mean gate-δ corr, {n_rho} corners, calibration split (W2)")
    print(f"DRIVEN_BAND_HDG_SCALE = {scale:.1f}  # {d}, LSQ of per-corner robust |dev| bound vs ∫|κ|ds, {n_c} corners (W2)")
    print(f"DRIVEN_BAND_FLOOR_M = {floor:.1f}  # {d}, intercept of the same fit (W2)")
    print(f"CONSISTENCY_STD_M = {cons:.1f}  # {d}, 1.4826·MAD consistency resid, {n_t} clean transits (W2)")
    print(f"SIGMA_PRIOR_MID_M = {pm:.1f}  # {d}, 1.4826·MAD mid-lap δ̂, {len(prior)} knots (W2)")
    if floor < 0 or scale < 0:
        print("WARNING: negative band term — STOP and discuss before applying (spec W2).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
