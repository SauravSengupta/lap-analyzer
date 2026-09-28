"""GPS-lag correction acceptance gates (§9, spec 2026-09-18). Local; needs real data/.

Revised 2026-09-27 (user decision) for the speed-only correction: GPS position was
measured to be ~on time (only `speed_mph_gps` is re-timed now — see the amended top of
docs/superpowers/specs/2026-09-18-gps-lag-correction-design.md). Gates:

  1. Residual lag — kept as written, ACCEPTED-UNMET when it fails (low-corr laps fall back
     to session median): re-estimating on corrected samples gives |τ| ≤ 0.05 s on ≥ 95% of
     clean OBD laps, per track. A failing track prints [ACCEPTED-UNMET], not [FAIL], and
     does not affect the exit code.
  2. Heading gate — dropped (built on the retired position-lag premise).
  3. OBD-session corner outputs identical: for each track, the corrected root's
     {track}_corners.parquet must match the baseline root's row-for-row (inner join on
     session_id/lap/corner_id, restricted to obd_present rows) since only GPS speed moved.
  4. Trust layer: gps_trust_battery hard gates pass on the corrected ridge corpus, and
     per-track rank_eligible rate falls by ≤ 2 pp vs baseline.
  5. PIR T1 brake std gate — dropped (built on the retired position-lag premise).

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


def _corpus_stats():
    from lap_analyzer.config import corpus_dir
    out = {}
    for track in TRACKS:
        c = pd.read_parquet(corpus_dir() / f"{track}_corners.parquet")
        out[f"{track}_rank_eligible"] = float(c["rank_eligible"].astype(bool).mean())
    return out


def dump(correct: bool) -> dict:
    import gps_trust_battery as battery
    res = {"residual": {t: _residual(t) for t in TRACKS}, **_corpus_stats()}
    if correct:
        res["battery_ok"] = bool(battery.run(battery.build_transits()))
    return res


def _corners_identical(baseline_root: str, root: str, track: str) -> dict:
    """Gate 3: OBD-session corner outputs must be identical between roots (only GPS
    speed moved; position/labels are untouched for OBD sessions)."""
    pb = ROOT / baseline_root / "corpus" / f"{track}_corners.parquet"
    pc = ROOT / root / "corpus" / f"{track}_corners.parquet"
    cb, cc = pd.read_parquet(pb), pd.read_parquet(pc)
    key = ["session_id", "lap", "corner_id"]
    obd_b = cb[cb["obd_present"].astype(bool)]
    obd_c = cc[cc["obd_present"].astype(bool)]
    merged = obd_b.merge(obd_c, on=key, suffixes=("_base", "_corr"), how="inner")
    row_count_ok = len(merged) == len(obd_b) == len(obd_c)
    numeric_cols = [c for c in obd_b.columns if c not in key
                    and c in obd_c.columns
                    and pd.api.types.is_numeric_dtype(obd_b[c])
                    and pd.api.types.is_numeric_dtype(obd_c[c])]
    diff_cols = []
    for c in numeric_cols:
        a = merged[f"{c}_base"].to_numpy(dtype=float)
        b = merged[f"{c}_corr"].to_numpy(dtype=float)
        if not np.allclose(a, b, equal_nan=True):
            diff_cols.append(c)
    ok = row_count_ok and not diff_cols
    detail = (f"rows base={len(obd_b)} corr={len(obd_c)} joined={len(merged)}; "
              f"differing columns: {diff_cols if diff_cols else '[]'}")
    return {"ok": ok, "detail": detail}


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
            f"1 residual lag {t}", frac >= 0.95,
            f"{frac:.1%} of {r['n_clean']} clean laps |τ|≤0.05s "
            f"({len(taus)} re-estimable; median |τ| {np.median(taus) if len(taus) else float('nan'):.3f})")
    for t in TRACKS:
        res = _corners_identical(a.baseline_root, a.root, t)
        gate(f"3 OBD corner outputs identical {t}", res["ok"], res["detail"])
    gate("4 trust battery (§7.3 rows)", corr.get("battery_ok"), "see stderr battery table")
    for t in TRACKS:
        d = (corr[f"{t}_rank_eligible"] - base[f"{t}_rank_eligible"]) * 100
        gate(f"4 rank_eligible {t}", d >= -2.0,
             f"{base[f'{t}_rank_eligible']:.1%} -> {corr[f'{t}_rank_eligible']:.1%} ({d:+.1f} pp)")
    print(f"\n{'PASS' if ok_all else 'FAIL'}: GPS-lag acceptance")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
