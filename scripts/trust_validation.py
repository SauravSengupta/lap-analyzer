"""Trust-layer validation report (W0). Local; needs real data/.

Scores a data root against checks the pipeline does not produce (V1–V4, see
validation_checks.py) plus the existing trust battery (V5), on HELD-OUT sessions, and
optionally compares against a baseline report and evaluates the acceptance gates.

  python scripts/trust_validation.py --root data --label w0-baseline
  python scripts/trust_validation.py --root data --label w1 --baseline-report docs/superpowers/artifacts/trust-validation/w0-baseline.json

Pairing: every report records its `population` (held sessions + evaluated laps). A run
with --baseline-report evaluates exactly the baseline's population (not today's
lap_reliable_traj selection), so G3/G7 compare the same laps; they are N/A otherwise.

Spec: docs/superpowers/specs/2026-10-10-sensor-fusion-rethink-design.md §W0 + gates.
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

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "docs" / "superpowers" / "artifacts" / "trust-validation"
TRACKS = ("pir", "ridge")

# Acceptance gates — spec §"Proposed acceptance gates". These are the PROPOSED values;
# Task 6 replaces them with the user's decision. Never edit them to make a run pass.
GATES = {
    "G2_burst_gate_bad_rankable_max": 0.05,
    "G4_v3_within_2sigma_min": 0.95,
    "G4_v3_tierA_within_010_min": 0.98,
    "G5_coverage_drop_pp_max": 2.0,
    "G5_pir_T1_min": 0.78,
}


def _held_corners(tr: str, held: set) -> pd.DataFrame:
    """Held-out, non-reference transits of one track's corpus corners table."""
    from lap_analyzer.config import corpus_dir
    c = pd.read_parquet(corpus_dir() / f"{tr}_corners.parquet")
    c = c[~c.get("is_reference", pd.Series(False, index=c.index)).fillna(False).astype(bool)]
    return c[[(tr, s) in held for s in c["session_id"]]]


def _coverage(held: set) -> dict:
    out = {}
    for tr in TRACKS:
        c = _held_corners(tr, held)
        obd = c[c["obd_present"]]
        out[tr] = dict(
            all=float(c["rank_eligible"].mean()),
            obd=float(obd["rank_eligible"].mean()),
            gps_only=float(c.loc[~c["obd_present"], "rank_eligible"].mean()) if (~c["obd_present"]).any() else None,
            per_corner_obd={k: float(v) for k, v in obd.groupby("corner_id")["rank_eligible"].mean().items()},
            per_corner_all={k: float(v) for k, v in c.groupby("corner_id")["rank_eligible"].mean().items()},
            jump_flagged_transit_share=(float((c["gps_jump_n"] > 0).mean()) if "gps_jump_n" in c else None),
            n=int(len(c)))
    return out


def _brake_std(held: set) -> dict:
    """gps_lag P3 rule: per corner std of brake_on_dist_m over pace decile ≤ 3 and
    transit_reliable rows (≥ 10 rows), median over corners. Reference laps excluded."""
    out = {}
    for tr in TRACKS:
        c = _held_corners(tr, held)
        c = c[(c["lap_pace_decile"] <= 3) & c["transit_reliable"].fillna(False).astype(bool)]
        per = c.groupby("corner_id")["brake_on_dist_m"].agg(["count", "std"])
        per = per[per["count"] >= 10]
        out[tr] = float(per["std"].median()) if len(per) else None
    return out


def _v2_summary(v2: pd.DataFrame) -> dict:
    from validation_checks import V2_BAD_ERR_S
    if "planted" not in v2.columns:
        v2 = v2.assign(planted=True, batch="mixed")
    batches = {b: dict(n_attempted=int(len(g)), n_skipped=int((~g["planted"].astype(bool)).sum()))
               for b, g in v2.groupby("batch")}
    v2 = v2[v2["planted"].astype(bool)].copy()
    v2["abs_err"] = v2["err_s"].abs()
    v2["bad_rankable"] = (v2["abs_err"] > V2_BAD_ERR_S) & v2["rank"].astype(bool)
    out = {"batches": batches}
    for (kind, where), g in v2.groupby(["kind", "where"]):
        out[f"{kind}/{where}"] = dict(
            n=int(len(g)), caught=float(g["caught"].mean()) if g["caught"].notna().any() else None,
            err_p90=float(g["abs_err"].quantile(0.9)), bad_rankable=float(g["bad_rankable"].mean()))
    burst = v2[(v2["kind"] == "burst") & (v2["where"] == "gate") & (v2["mag_m"] >= 10)]
    short = burst[burst["n_fix"] <= 8]                     # G2 is defined on 0.4–1 s bursts (g2 + mixed)
    long_ = burst[burst["n_fix"] > 8]                      # 1–2 s: reported (long-run capture), not gated
    out["burst_gate_ge10_bad_rankable"] = float(short["bad_rankable"].mean()) if len(short) else None
    out["burst_gate_ge10_n"] = int(len(short))
    out["burst_gate_ge10_n_skipped"] = batches.get("g2", {}).get("n_skipped", 0)
    out["long_burst_gate_ge10_bad_rankable"] = float(long_["bad_rankable"].mean()) if len(long_) else None
    drift = v2[v2["kind"] == "slow_drift"]
    out["slow_drift_bad_rankable"] = float(drift["bad_rankable"].mean()) if len(drift) else None
    out["slow_drift_n"] = int(len(drift))
    return out


def _v3_summary(v3: pd.DataFrame) -> dict:
    empty = dict(n=0, median_abs=None, p90_abs=None, within_010=None, within_2sigma=None,
                 gps_tierA_share=None, gps_tierA_within_010=None)
    need = {"rank_obd", "t_gps", "t_obd", "sigma_gps", "rank_gps"}
    if len(v3) == 0 or not need.issubset(v3.columns):
        return empty
    a = v3[v3["rank_obd"] & np.isfinite(v3["t_gps"])]
    if len(a) == 0:
        return empty
    d = (a["t_gps"] - a["t_obd"]).abs()
    tier_a = a[a["rank_gps"]]
    dA = (tier_a["t_gps"] - tier_a["t_obd"]).abs()
    return dict(n=int(len(a)), median_abs=float(d.median()), p90_abs=float(d.quantile(0.9)),
                within_010=float((d <= 0.10).mean()),
                within_2sigma=float((d <= 2.0 * a["sigma_gps"]).mean()),
                gps_tierA_share=float(a["rank_gps"].mean()),
                gps_tierA_within_010=float((dA <= 0.10).mean()) if len(dA) else None)


def _v4_summary(v4: pd.DataFrame) -> dict:
    if len(v4) == 0 or not {"rank", "t_odo", "t_prod"}.issubset(v4.columns):
        return dict(n_rankable=0, gt_020=0, gt_030=0, gt_030_rate=None)
    r = v4[v4["rank"] & np.isfinite(v4["t_odo"])]
    d = (r["t_prod"] - r["t_odo"]).abs()
    n, gt_030 = int(len(r)), int((d > 0.3).sum())
    return dict(n_rankable=n, gt_020=int((d > 0.2).sum()), gt_030=gt_030,
                gt_030_rate=gt_030 / n if n else None)


def _v5(root: str) -> dict:
    env = dict(os.environ, DATA_ROOT=root, PYTHONPATH=str(REPO))

    def run(cmd):
        try:
            r = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return r.returncode == 0, r.stdout[-1500:] + "\n--stderr--\n" + r.stderr[-800:]

    bat_ok, bat_tail = run([sys.executable, str(REPO / "scripts" / "gps_trust_battery.py")])
    cal_ok, cal_tail = run([sys.executable, str(REPO / "scripts" / "gps_trust_calibration.py"), "500"])
    return dict(battery_pass=bat_ok, calibration_pass=cal_ok,
                battery_tail=bat_tail, calibration_tail=cal_tail)


_MISSING = object()


def _dig(d, *path):
    """Nested lookup; returns _MISSING if any key is absent."""
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return _MISSING
        d = d[k]
    return d


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and bool(np.isfinite(x))


def _population_key(rep: dict):
    pop = rep.get("population")
    if not isinstance(pop, dict) or "held" not in pop or "laps" not in pop:
        return None
    return (sorted(tuple(x) for x in pop["held"]), sorted(tuple(x) for x in pop["laps"]))


def same_population(cur: dict, base: dict) -> bool:
    """True only when both reports record a population and they are identical."""
    k = _population_key(cur)
    return k is not None and k == _population_key(base)


POP_DIFF = "populations differ — re-run with --baseline-report"


def evaluate_gates(cur: dict, base: dict) -> list[tuple[str, str, str]]:
    """Return (name, status, detail) with status in PASS / FAIL / N/A.

    Missing evidence is N/A, never PASS. FAIL is only a real threshold miss."""
    g = []

    def lacks(*path):
        return f"baseline lacks {'.'.join(path)}"

    # G1
    scope = ("scope: gps_trust_battery = full Ridge corpus (not held-out, no PIR); "
             "gps_trust_calibration = synthetic laps")
    v5 = cur.get("v5")
    if v5 is None:
        g.append(("G1 trust battery + calibration", "N/A", f"V5 not run (--skip-v5); {scope}"))
    else:
        ok = bool(v5.get("battery_pass") and v5.get("calibration_pass"))
        g.append(("G1 trust battery + calibration", "PASS" if ok else "FAIL", scope))

    # G2
    bg = _dig(cur, "v2", "burst_gate_ge10_bad_rankable")
    if bg is _MISSING or bg is None:
        g.append(("G2 burst@gate bad-but-rankable", "N/A", "no burst@gate evidence (n=0)"))
    else:
        thr = GATES["G2_burst_gate_bad_rankable_max"]
        n2 = (cur.get("v2") or {}).get("burst_gate_ge10_n", "?")
        sk2 = (cur.get("v2") or {}).get("burst_gate_ge10_n_skipped", "?")
        g.append(("G2 burst@gate bad-but-rankable", "PASS" if bg <= thr else "FAIL",
                  f"{bg} <= {thr}; n_skipped={sk2}; n={n2}"))

    # G3
    cd = _dig(cur, "v2", "slow_drift_bad_rankable")
    bd = _dig(base, "v2", "slow_drift_bad_rankable")
    n3d = (cur.get("v2") or {}).get("slow_drift_n", "?")
    paired = same_population(cur, base)
    if not paired:
        g.append(("G3 slow-drift bad-but-rankable not worse", "N/A", POP_DIFF))
    elif bd is _MISSING:
        g.append(("G3 slow-drift bad-but-rankable not worse", "N/A", lacks("v2", "slow_drift_bad_rankable")))
    elif cd is _MISSING or cd is None or bd is None:
        g.append(("G3 slow-drift bad-but-rankable not worse", "N/A", f"no slow-drift evidence: {cd} vs {bd}"))
    else:
        g.append(("G3 slow-drift bad-but-rankable not worse", "PASS" if cd <= bd else "FAIL",
                  f"{cd} vs {bd}; n={n3d}"))

    # G4
    v3 = cur.get("v3") or {}
    w2, tA, n4 = v3.get("within_2sigma"), v3.get("gps_tierA_within_010"), v3.get("n")
    shA = v3.get("gps_tierA_share")
    nA = round(shA * n4) if _finite(shA) and _finite(n4) else "?"
    if w2 is None:
        g.append(("G4 GPS-only emulation honest", "N/A", f"no V3 evidence, n={n4}"))
    else:
        w2_ok = w2 >= GATES["G4_v3_within_2sigma_min"]
        det = f"within_2sigma {w2:.3f}, n={n4}; "
        if tA is None:
            det += "tierA: no GPS-only tier-A transits"
            status = "FAIL" if not w2_ok else "N/A"
        else:
            tA_ok = tA >= GATES["G4_v3_tierA_within_010_min"]
            det += f"tierA<=0.10 {tA:.3f}, n_tierA={nA}"
            status = "PASS" if (w2_ok and tA_ok) else "FAIL"
        g.append(("G4 GPS-only emulation honest", status, det))

    # G5
    det, fail, na = [], False, []
    for tr in TRACKS:
        b = _dig(base, "coverage", tr, "obd")
        c = _dig(cur, "coverage", tr, "obd")
        if b is _MISSING or c is _MISSING:
            na.append(lacks("coverage", tr, "obd"))
            continue
        if not (_finite(b) and _finite(c)):
            na.append(f"{tr} obd coverage not finite ({c} vs {b})")
            continue
        drop = 100.0 * (b - c)
        fail |= drop > GATES["G5_coverage_drop_pp_max"]
        det.append(f"{tr} obd drop {drop:+.1f}pp")
    t1 = _dig(cur, "coverage", "pir", "per_corner_obd")
    t1all = _dig(cur, "coverage", "pir", "per_corner_all")
    t1all_v = t1all.get("T1") if isinstance(t1all, dict) else None
    t1all_txt = f"{t1all_v:.3f}" if _finite(t1all_v) else "n/a"
    t1_missing = not isinstance(t1, dict) or not _finite(t1.get("T1"))
    if t1_missing:
        det.append(f"PIR T1 missing (obd=n/a all={t1all_txt})")
    else:
        fail |= t1["T1"] < GATES["G5_pir_T1_min"]
        det.append(f"PIR T1 obd={t1['T1']:.3f} all={t1all_txt}")
    status = "FAIL" if fail else ("N/A" if (na or t1_missing) else "PASS")
    g.append(("G5 coverage + PIR T1", status, ", ".join(det + na)))

    # G6
    brake_fail, brake_na = False, []
    for tr in TRACKS:
        c = _dig(cur, "brake_std", tr)
        b = _dig(base, "brake_std", tr)
        if b is _MISSING:
            brake_na.append(lacks("brake_std", tr))
        elif c is _MISSING or c is None or b is None:
            brake_na.append(f"no brake evidence for {tr}")
        elif c > b:
            brake_fail = True
    v1c = _dig(cur, "v1", "shift_std_median_m")
    v1b = _dig(base, "v1", "shift_std_median_m")
    v1_na = v1c is _MISSING or v1b is _MISSING or v1c is None or v1b is None
    v1_fail = (not v1_na) and v1c > v1b
    v1_txt = "N/A (no V1 shift_std)" if v1_na else f"{v1c} vs {v1b}"
    det = f"v1 {v1_txt}; brake {_dig(cur, 'brake_std')} vs {_dig(base, 'brake_std')}"
    if brake_na:
        det += "; " + ", ".join(brake_na)
    if brake_fail or v1_fail:
        status = "FAIL"
    elif v1_na or brake_na:
        status = "N/A"
    else:
        status = "PASS"
    g.append(("G6 along-track + brake scatter not worse", status, det))

    # G7 — rate of rankable V4 transits off by > 0.3 s, not more than the baseline's
    c4, b4 = _dig(cur, "v4", "gt_030_rate"), _dig(base, "v4", "gt_030_rate")
    counts = (f"gt_030 {_dig(cur, 'v4', 'gt_030')} vs {_dig(base, 'v4', 'gt_030')}; "
              f"n_rankable {_dig(cur, 'v4', 'n_rankable')} vs {_dig(base, 'v4', 'n_rankable')}")
    if not paired:
        g.append(("G7 V4 >0.3 s rate not more", "N/A", POP_DIFF))
    elif b4 is _MISSING:
        g.append(("G7 V4 >0.3 s rate not more", "N/A", lacks("v4", "gt_030_rate")))
    elif not (_finite(c4) and _finite(b4)):
        g.append(("G7 V4 >0.3 s rate not more", "N/A", f"no rankable V4 evidence: {c4} vs {b4}; {counts}"))
    else:
        g.append(("G7 V4 >0.3 s rate not more", "PASS" if c4 <= b4 else "FAIL",
                  f"rate {c4:.4f} vs {b4:.4f}; {counts}"))
    return g


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--label", required=True)
    p.add_argument("--baseline-report")
    p.add_argument("--max-laps", type=int, default=None,
                   help="per-track cap (laps taken in hash order); default: every clean held-out lap")
    p.add_argument("--skip-v5", action="store_true")
    p.add_argument("--seed", type=int, default=20261010)
    args = p.parse_args()
    os.environ["DATA_ROOT"] = args.root
    sys.path.insert(0, str(REPO / "scripts"))
    import validation_checks as vc
    from lap_analyzer.labeler import build_reference_index, load_track

    base = json.loads(Path(args.baseline_report).read_text()) if args.baseline_report else None
    base_pop = (base or {}).get("population")
    if base_pop:
        # Paired run: evaluate exactly the baseline's held set and laps.
        held = {(t, s) for t, s in base_pop["held"]}
        if args.max_laps is not None:
            print("note: --max-laps ignored — population fixed by the baseline report")
    else:
        sessions = pd.concat([vc.list_sessions(tr) for tr in TRACKS], ignore_index=True)
        split = vc.session_split(sessions)
        held = {k for k, v in split.items() if v == "held"}
    v2s, v3s, v4s, pop_laps, n_missing = [], [], [], [], 0
    for tr in TRACKS:
        ctx = vc.TrackCtx.load(tr)
        if base_pop:
            laps, miss = vc.load_laps(ctx, [(s, lap) for t, s, lap in base_pop["laps"] if t == tr])
            n_missing += miss
        else:
            sids = [s for (t, s) in held if t == tr]
            laps = list(vc.clean_laps(ctx, sids, max_laps=args.max_laps))
        pop_laps += [[tr, sid, lap] for sid, lap, _ in laps]
        v2s.append(vc.run_v2(ctx, laps, build_reference_index(load_track(tr)), args.seed))
        v3s.append(vc.run_v3(ctx, laps))
        v4s.append(vc.run_v4(ctx, laps))
    v1 = vc.run_v1("pir", [s for (t, s) in held if t == "pir"])
    v1ok = v1[v1["valid"]] if len(v1) else v1
    report = dict(
        label=args.label, root=args.root, n_sessions_held=len(held),
        population=dict(held=sorted([t, s] for t, s in held), laps=sorted(pop_laps)),
        population_source="baseline" if base_pop else "selected",
        population_missing_laps=n_missing,
        coverage=_coverage(held), brake_std=_brake_std(held),
        v1=dict(sessions=int(len(v1)), valid=int(len(v1ok)),
                shift_std_median_m=float(v1ok["shift_std_median_m"].median()) if len(v1ok) else None),
        v2=_v2_summary(pd.concat(v2s, ignore_index=True)),
        v3=_v3_summary(pd.concat(v3s, ignore_index=True)),
        v4=_v4_summary(pd.concat(v4s, ignore_index=True)),
        v5=None if args.skip_v5 else _v5(args.root))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.label}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k not in ("v5", "population")}, indent=2, default=str))
    print(f"population: {len(report['population']['held'])} held sessions, "
          f"{len(pop_laps)} laps ({report['population_source']}; {n_missing} missing)")
    print(f"report: {out}")
    if base is not None:
        rc, counts = 0, {"PASS": 0, "FAIL": 0, "N/A": 0}
        for name, status, detail in evaluate_gates(report, base):
            print(f"[{status}] {name}: {detail}")
            counts[status] += 1
            rc |= 1 if status == "FAIL" else 0
        print(f"gates: {counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['N/A']} N/A")
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
