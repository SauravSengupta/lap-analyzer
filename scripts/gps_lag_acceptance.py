"""GPS-lag correction acceptance gates (§9, spec 2026-09-18). Local; needs real data/.

A/B's two fully rebuilt data roots on identical code:
  baseline root  — normalize --no-gps-lag → label_corners → flag_quality → build_corpus
  corrected root — the same without --no-gps-lag
and prints every gate as pass/fail:

  1. Residual lag: re-estimating on corrected samples gives |τ| ≤ 0.05 s on ≥ 95% of
     clean OBD laps, per track.
  2. Independent check: GPS heading-rate vs accelerometer lat_g lag (raw CSVs, per-lap τ
     applied) has median within ±0.10 s in each of the four spec sessions.
  3. Gate span vs odometer: |driven_m − gate separation| at PIR T6/T12 on 20260918-103444 L8
     and 20260725-111619 L2 ≤ 8 m in all four cells (baseline should reproduce ~13/6/5/20 m).
  4. Trust layer: gps_trust_battery hard gates pass on the corrected ridge corpus, and
     per-track rank_eligible rate falls by ≤ 2 pp vs baseline.
  5. PIR T1 brake_on_dist_m std over fast reliable laps (pace decile ≤ 3, transit_reliable)
     does not increase vs baseline.

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
HEADING_SESSIONS = ["20260918-103444", "20260725-111619", "20250522-101214", "20240706-100029"]
SPAN_CELLS = [("20260918-103444", 8), ("20260725-111619", 2)]
SPAN_CORNERS = ["T6", "T12"]


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


def _heading_lag(sid, correct):
    """Median over inner laps of the heading-rate vs lat_g lag, from the raw CSV."""
    from lap_analyzer.config import data_root
    from lap_analyzer.gps_lag import (LAG_MIN_SPEED_MPH, apply_gps_lag,
                                      estimate_session_gps_lag, xcorr_lag)
    from lap_analyzer.normalize import OBD_CHANNELS, read_csv
    paths = list((data_root() / "raw").glob(f"*/Log-{sid}*.csv"))
    if not paths:
        return None
    raw = read_csv(paths[0])
    for c in OBD_CHANNELS:
        if c not in raw.columns:
            raw[c] = np.nan
    raw["lat_g"] = -raw["lat_g"]
    raw["long_g"] = -raw["long_g"]
    raw[OBD_CHANNELS] = raw[OBD_CHANNELS].ffill().bfill()
    if correct:
        lags = estimate_session_gps_lag(raw)
        raw = apply_gps_lag(raw, {k: e.tau_s for k, e in lags.items()})
    laps = sorted(raw["lap"].unique())[1:-1]
    out = []
    for lap in laps:
        g = raw[raw["lap"] == lap]
        t = g["t"].to_numpy(float)
        hdg = np.unwrap(np.deg2rad(g["heading"].to_numpy(float)))
        yaw = pd.Series(np.gradient(hdg, t)).rolling(9, center=True, min_periods=1).mean()
        latg = g["lat_g"].rolling(9, center=True, min_periods=1).mean()
        v = g["speed_mph"] if g["speed_mph"].notna().any() else g["speed_mph_gps"]
        tau, corr = xcorr_lag(t, latg.to_numpy(), yaw.to_numpy(),
                              mask=v.to_numpy() > LAG_MIN_SPEED_MPH)
        if np.isfinite(corr) and corr >= 0.6:
            out.append(tau)
    return float(np.median(out)) if out else None


def _spans():
    from lap_analyzer.analysis import section_bounds, section_times
    td = json.loads((ROOT / "tracks" / "pir.json").read_text())
    st = section_times("pir", td)
    bounds = section_bounds(td)
    cells = {}
    for sid, lap in SPAN_CELLS:
        for c in SPAN_CORNERS:
            row = st[(st.session_id == sid) & (st.lap == lap) & (st.corner_id == c)]
            a, b = bounds[c]
            cells[f"{sid} L{lap} {c}"] = (float(abs(row["driven_m"].iloc[0] - (b - a)))
                                          if len(row) else None)
    return cells


def _corpus_stats():
    from lap_analyzer.config import corpus_dir
    out = {}
    for track in TRACKS:
        c = pd.read_parquet(corpus_dir() / f"{track}_corners.parquet")
        out[f"{track}_rank_eligible"] = float(c["rank_eligible"].astype(bool).mean())
        if track == "pir":
            f = c[(c.corner_id == "T1") & (c.lap_pace_decile <= 3)
                  & c.transit_reliable.fillna(False).astype(bool)]
            out["pir_t1_brake_std"] = float(f["brake_on_dist_m"].std())
            out["pir_t1_n"] = int(f["brake_on_dist_m"].notna().sum())
    return out


def dump(correct: bool) -> dict:
    import gps_trust_battery as battery
    res = {"residual": {t: _residual(t) for t in TRACKS},
           "heading": {s: _heading_lag(s, correct) for s in HEADING_SESSIONS},
           "spans": _spans(), **_corpus_stats()}
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

    for t in TRACKS:
        r = corr["residual"][t]
        taus = np.abs(np.array(r["taus"]))
        frac = float((taus <= 0.05).sum() / max(r["n_clean"], 1))
        gate(f"1 residual lag {t}", frac >= 0.95,
             f"{frac:.1%} of {r['n_clean']} clean laps |τ|≤0.05s "
             f"({len(taus)} re-estimable; median |τ| {np.median(taus) if len(taus) else float('nan'):.3f})")
    for s in HEADING_SESSIONS:
        h, hb = corr["heading"][s], base["heading"][s]
        gate(f"2 heading-rate lag {s}", h is not None and abs(h) <= 0.10,
             f"corrected {h} s (baseline {hb} s)")
    for k in corr["spans"]:
        v, vb = corr["spans"][k], base["spans"][k]
        gate(f"3 span vs odometer {k}", v is not None and v <= 8.0,
             f"{v if v is None else round(v, 1)} m (baseline {vb if vb is None else round(vb, 1)} m)")
    gate("4 trust battery (§7.3 rows)", corr.get("battery_ok"), "see stderr battery table")
    for t in TRACKS:
        d = (corr[f"{t}_rank_eligible"] - base[f"{t}_rank_eligible"]) * 100
        gate(f"4 rank_eligible {t}", d >= -2.0,
             f"{base[f'{t}_rank_eligible']:.1%} -> {corr[f'{t}_rank_eligible']:.1%} ({d:+.1f} pp)")
    gate("5 PIR T1 brake_on std", corr["pir_t1_brake_std"] <= base["pir_t1_brake_std"],
         f"{base['pir_t1_brake_std']:.2f} -> {corr['pir_t1_brake_std']:.2f} m "
         f"(n {base['pir_t1_n']} -> {corr['pir_t1_n']})")
    print(f"\n{'PASS' if ok_all else 'FAIL'}: GPS-lag acceptance")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
