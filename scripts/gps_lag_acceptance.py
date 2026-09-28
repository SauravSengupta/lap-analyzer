"""GPS-lag correction acceptance gates (§9, spec 2026-09-18). Local; needs real data/.

Revised 2026-09-27 (user-approved) for the position correction: GPS position + Lap
counter are now re-timed by τ_pos = τ_speed − 0.12 s (see the amended top of
docs/superpowers/specs/2026-09-18-gps-lag-correction-design.md). Gate set:

  1. Residual speed lag — kept as written, ACCEPTED-UNMET when it fails (low-corr laps
     fall back to session median): re-estimating on corrected samples gives |τ| ≤ 0.05 s
     on ≥ 95% of clean OBD laps, per track. A failing track prints [ACCEPTED-UNMET], not
     [FAIL], and does not affect the exit code.
  2. Heading gate — dropped (built on the retired position-lag premise).
  3. OBD-session corner outputs identical — REMOVED: position now moves by design, so
     corner outputs are expected to differ.
  P1. Residual position lag: per track, over clean laps (laps.csv `is_clean`) of OBD
      sessions (skipping sessions whose `speed_mph` is all-NaN), skipping laps with
      < 1500 rows or > 5% of rows under 20 mph, estimate the position lag with the
      cumulative-path method (`_position_lag`). PASS if |median over laps| ≤ 0.05 s on
      the corrected root. Baseline median is printed alongside (expected ≈ +0.15 s).
  P2. Trust layer: gps_trust_battery hard gates pass on the corrected ridge corpus, and
      per-track rank_eligible rate falls by ≤ 2 pp vs baseline. ADDED: per-corner
      rank_eligible rate drop (baseline → corrected) ≤ 5 pp; worst corner per track is
      printed.
  P3. Brake-point cross-lap precision: per track, per corner_id, std of
      `brake_on_dist_m` over rows with `lap_pace_decile <= 3` and `transit_reliable`
      true (fillna False), skipping corners with < 10 such rows; take the median over
      corners. PASS if corrected ≤ baseline, per track.

A/B's two fully rebuilt data roots on identical code:
  baseline root  — normalize --no-gps-lag → label_corners → flag_quality → build_corpus
  corrected root — the same without --no-gps-lag

Run:  python scripts/gps_lag_acceptance.py --baseline-root data_nolag --root data
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TRACKS = ["ridge", "pir"]


def _residual(track):
    from lap_analyzer.config import sessions_dir
    from lap_analyzer.gps_lag import estimate_gps_lag
    taus, n_clean = [], 0
    for sdir in sorted(p for p in sessions_dir(track).iterdir() if p.is_dir()):
        sp, lp = sdir / "samples.parquet", sdir / "laps.csv"
        if not sp.exists() or not lp.exists() or sdir.name.startswith("_"):
            continue
        s = pd.read_parquet(sp)
        if s["speed_mph"].isna().all():
            continue
        clean = set(pd.read_csv(lp).query("is_clean")["lap"].astype(int))
        for lap, g in s.groupby("lap"):
            if int(lap) not in clean:
                continue
            n_clean += 1
            e = estimate_gps_lag(g["t"], g["speed_mph"], g["speed_mph_gps"], g["long_g"])
            if e.source == "lap":
                taus.append(e.tau_s)
    return {"n_clean": n_clean, "taus": taus}


def _position_lag(t, lat, lon, speed_mph) -> float:
    """P1: estimate the GPS position lag by cross-correlating a smoothed cumulative
    GPS path against OBD-derived cumulative distance. Returns the lag in seconds,
    already net of OBD_HOLD_LEAD_S (the row-clock lag), or NaN if the lap has a
    teleport glitch or too few usable rows/fixes."""
    from lap_analyzer.gps_lag import OBD_HOLD_LEAD_S

    t = np.asarray(t, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    speed_mph = np.asarray(speed_mph, dtype=float)
    n = len(t)
    if n - 100 < 50:
        return float("nan")

    # D(t): trapezoidal cumulative OBD distance.
    v_mps = speed_mph * 0.44704
    D = np.concatenate([[0.0], np.cumsum((v_mps[:-1] + v_mps[1:]) / 2.0 * np.diff(t))])

    # Fresh fixes: lat or lon changed from the previous row (first row counts).
    fresh = np.ones(n, dtype=bool)
    fresh[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
    if fresh.sum() < 10:
        return float("nan")

    # Local projection with ONE fixed reference (not a per-point cos(lat), which
    # adds a spurious term and shortens the path ~9%).
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    lat0, lon0 = float(np.mean(lat_r)), float(np.mean(lon_r))
    c0 = np.cos(lat0)
    R = 6371000.0
    x = R * c0 * (lon_r - lon0)
    y = R * (lat_r - lat0)

    t_fresh = t[fresh]
    x_fresh = pd.Series(x[fresh]).rolling(7, center=True, min_periods=1).mean().to_numpy()
    y_fresh = pd.Series(y[fresh]).rolling(7, center=True, min_periods=1).mean().to_numpy()

    steps = np.hypot(np.diff(x_fresh), np.diff(y_fresh))
    if len(steps) and steps.max() > 30.0:
        return float("nan")  # teleport glitch

    S_fresh = np.concatenate([[0.0], np.cumsum(steps)])
    S = np.interp(t, t_fresh, S_fresh)

    lo, hi = 50, n - 50
    t_trim, S_trim = t[lo:hi], S[lo:hi]
    if len(t_trim) < 50:
        return float("nan")

    t0 = t_trim[0]
    best_tau, best_std = 0.0, np.inf
    for tau in np.arange(-0.3, 1.51, 0.01):
        D_shift = np.interp(t_trim - tau, t, D)
        A = np.column_stack([np.ones_like(t_trim), D_shift, t_trim - t0])
        coef, *_ = np.linalg.lstsq(A, S_trim, rcond=None)
        resid = S_trim - A @ coef
        std = float(resid.std())
        if std < best_std:
            best_std, best_tau = std, float(tau)

    return best_tau - OBD_HOLD_LEAD_S


def _position_lag_track(track):
    from lap_analyzer.config import sessions_dir
    taus = []
    for sdir in sorted(p for p in sessions_dir(track).iterdir() if p.is_dir()):
        sp, lp = sdir / "samples.parquet", sdir / "laps.csv"
        if not sp.exists() or not lp.exists() or sdir.name.startswith("_"):
            continue
        s = pd.read_parquet(sp)
        if s["speed_mph"].isna().all():
            continue
        clean = set(pd.read_csv(lp).query("is_clean")["lap"].astype(int))
        for lap, g in s.groupby("lap"):
            if int(lap) not in clean:
                continue
            if len(g) < 1500:
                continue
            if float((g["speed_mph"] < 20).mean()) > 0.05:
                continue
            tau = _position_lag(g["t"], g["lat"], g["long"], g["speed_mph"])
            if np.isfinite(tau):
                taus.append(tau)
    return taus


def _corpus_stats():
    from lap_analyzer.config import corpus_dir
    out = {}
    for track in TRACKS:
        c = pd.read_parquet(corpus_dir() / f"{track}_corners.parquet")
        out[f"{track}_rank_eligible"] = float(c["rank_eligible"].astype(bool).mean())
        per_corner = c.groupby("corner_id")["rank_eligible"].apply(lambda s: float(s.astype(bool).mean()))
        out[f"{track}_rank_eligible_by_corner"] = per_corner.to_dict()

        sub = c[(c["lap_pace_decile"] <= 3) & c["transit_reliable"].fillna(False).astype(bool)]
        stds = sub.groupby("corner_id")["brake_on_dist_m"].apply(
            lambda s: float(s.std()) if len(s) >= 10 else np.nan)
        stds = stds.dropna()
        out[f"{track}_brake_std_median"] = float(stds.median()) if len(stds) else float("nan")
    return out


def dump(correct: bool) -> dict:
    import gps_trust_battery as battery
    res = {
        "residual": {t: _residual(t) for t in TRACKS},
        "position_lag": {t: _position_lag_track(t) for t in TRACKS},
        **_corpus_stats(),
    }
    if correct:
        res["battery_ok"] = bool(battery.run(battery.build_transits()))
    return res


def _measure(root: str, correct: bool) -> dict:
    env = {**os.environ, "DATA_ROOT": root}
    args = [sys.executable, __file__, "--dump", root] + ([] if correct else ["--baseline"])
    p = subprocess.run(args, env=env, capture_output=True, text=True, cwd=ROOT)
    sys.stderr.write(p.stderr)   # battery table + diagnostics, pass or fail
    if p.returncode != 0:
        raise SystemExit(2)
    return json.loads(p.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--baseline-root", default="data_nolag")
    ap.add_argument("--dump")
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # τ/≤ in the table; Windows redirects default to cp1252
    if a.dump:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):   # battery prints its own table
            res = dump(correct=not a.baseline)
        sys.stderr.write(buf.getvalue())
        print(json.dumps(res))
        return 0

    base, corr = _measure(a.baseline_root, False), _measure(a.root, True)
    ok_all = True

    def gate(name, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    def gate_accepted_unmet(name, ok, detail):
        # Kept as written; a failure is recorded ACCEPTED-UNMET (2026-09-27: low-corr
        # laps fall back to session median) and does not count toward the exit code.
        status = "PASS" if ok else "ACCEPTED-UNMET"
        suffix = "" if ok else " (accepted 2026-09-27: low-corr laps fall back to session median)"
        print(f"[{status}] {name}: {detail}{suffix}")

    for t in TRACKS:
        r = corr["residual"][t]
        taus = np.abs(np.array(r["taus"]))
        frac = float((taus <= 0.05).sum() / max(r["n_clean"], 1))
        gate_accepted_unmet(
            f"1 residual speed lag {t}", frac >= 0.95,
            f"{frac:.1%} of {r['n_clean']} clean laps |τ|≤0.05s "
            f"({len(taus)} re-estimable; median |τ| {np.median(taus) if len(taus) else float('nan'):.3f})")

    for t in TRACKS:
        taus_c = np.array(corr["position_lag"][t], dtype=float)
        taus_b = np.array(base["position_lag"][t], dtype=float)
        med_c = float(np.median(taus_c)) if len(taus_c) else float("nan")
        med_b = float(np.median(taus_b)) if len(taus_b) else float("nan")
        gate(f"P1 residual position lag {t}", np.isfinite(med_c) and abs(med_c) <= 0.05,
             f"corrected median {med_c:+.3f}s over {len(taus_c)} laps "
             f"(baseline median {med_b:+.3f}s over {len(taus_b)} laps)")

    gate("P2 trust battery (§7.3 rows)", corr.get("battery_ok"), "see stderr battery table")

    for t in TRACKS:
        d = (corr[f"{t}_rank_eligible"] - base[f"{t}_rank_eligible"]) * 100
        gate(f"P2 rank_eligible {t}", d >= -2.0,
             f"{base[f'{t}_rank_eligible']:.1%} -> {corr[f'{t}_rank_eligible']:.1%} ({d:+.1f} pp)")

    for t in TRACKS:
        base_pc = base[f"{t}_rank_eligible_by_corner"]
        corr_pc = corr[f"{t}_rank_eligible_by_corner"]
        drops = {cid: (base_pc[cid] - corr_pc[cid]) * 100 for cid in base_pc if cid in corr_pc}
        worst_cid = max(drops, key=drops.get) if drops else None
        worst_drop = drops[worst_cid] if worst_cid is not None else float("nan")
        gate(f"P2 per-corner rank_eligible {t}", not drops or worst_drop <= 5.0,
             f"worst corner {worst_cid}: {base_pc.get(worst_cid, float('nan')):.1%} -> "
             f"{corr_pc.get(worst_cid, float('nan')):.1%} ({worst_drop:+.1f} pp)")

    for t in TRACKS:
        b = base[f"{t}_brake_std_median"]
        c = corr[f"{t}_brake_std_median"]
        gate(f"P3 brake-point precision {t}", np.isfinite(c) and np.isfinite(b) and c <= b,
             f"median std(brake_on_dist_m) baseline {b:.2f}m -> corrected {c:.2f}m")

    print(f"\n{'PASS' if ok_all else 'FAIL'}: GPS-lag acceptance")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
