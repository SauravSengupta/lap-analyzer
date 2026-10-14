"""Trust-layer validation report (W0). Local; needs real data/.

Scores a data root against checks the pipeline does not produce (V1–V4, see
validation_checks.py) plus the existing trust battery (V5), on HELD-OUT sessions, and
optionally compares against a baseline report and evaluates the acceptance gates.

  python scripts/trust_validation.py --root data --label w0-baseline
  python scripts/trust_validation.py --root data --label w1 --baseline-report docs/superpowers/artifacts/trust-validation/w0-baseline.json

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


def _coverage(held: set) -> dict:
    from lap_analyzer.config import corpus_dir
    out = {}
    for tr in TRACKS:
        c = pd.read_parquet(corpus_dir() / f"{tr}_corners.parquet")
        c = c[~c.get("is_reference", pd.Series(False, index=c.index)).fillna(False).astype(bool)]
        c = c[[(tr, s) in held for s in c["session_id"]]]
        obd = c[c["obd_present"]]
        out[tr] = dict(
            all=float(c["rank_eligible"].mean()),
            obd=float(obd["rank_eligible"].mean()),
            gps_only=float(c.loc[~c["obd_present"], "rank_eligible"].mean()) if (~c["obd_present"]).any() else None,
            per_corner_obd={k: float(v) for k, v in obd.groupby("corner_id")["rank_eligible"].mean().items()},
            jump_flagged_transit_share=(float((c["gps_jump_n"] > 0).mean()) if "gps_jump_n" in c else None),
            n=int(len(c)))
    return out


def _brake_std(held: set) -> dict:
    """gps_lag P3 rule: per corner std of brake_on_dist_m over pace decile ≤ 3 and
    transit_reliable rows (≥ 10 rows), median over corners."""
    from lap_analyzer.config import corpus_dir
    out = {}
    for tr in TRACKS:
        c = pd.read_parquet(corpus_dir() / f"{tr}_corners.parquet")
        c = c[[(tr, s) in held for s in c["session_id"]]]
        c = c[(c["lap_pace_decile"] <= 3) & c["transit_reliable"].fillna(False).astype(bool)]
        per = c.groupby("corner_id")["brake_on_dist_m"].agg(["count", "std"])
        per = per[per["count"] >= 10]
        out[tr] = float(per["std"].median()) if len(per) else None
    return out


def _v2_summary(v2: pd.DataFrame) -> dict:
    from validation_checks import V2_BAD_ERR_S
    v2 = v2.copy()
    v2["abs_err"] = v2["err_s"].abs()
    v2["bad_rankable"] = (v2["abs_err"] > V2_BAD_ERR_S) & v2["rank"]
    out = {}
    for (kind, where), g in v2.groupby(["kind", "where"]):
        out[f"{kind}/{where}"] = dict(
            n=int(len(g)), caught=float(g["caught"].mean()) if g["caught"].notna().any() else None,
            err_p90=float(g["abs_err"].quantile(0.9)), bad_rankable=float(g["bad_rankable"].mean()))
    burst = v2[(v2["kind"] == "burst") & (v2["where"] == "gate") & (v2["mag_m"] >= 10)]
    short = burst[burst["n_fix"] <= 8]                     # G2 is defined on 0.4–1 s bursts
    long_ = burst[burst["n_fix"] > 8]                      # 1–2 s: reported (long-run capture), not gated
    out["burst_gate_ge10_bad_rankable"] = float(short["bad_rankable"].mean()) if len(short) else None
    out["long_burst_gate_ge10_bad_rankable"] = float(long_["bad_rankable"].mean()) if len(long_) else None
    drift = v2[v2["kind"] == "slow_drift"]
    out["slow_drift_bad_rankable"] = float(drift["bad_rankable"].mean()) if len(drift) else None
    return out


def _v3_summary(v3: pd.DataFrame) -> dict:
    a = v3[v3["rank_obd"] & np.isfinite(v3["t_gps"])]
    d = (a["t_gps"] - a["t_obd"]).abs()
    tier_a = a[a["rank_gps"]]
    dA = (tier_a["t_gps"] - tier_a["t_obd"]).abs()
    return dict(n=int(len(a)), median_abs=float(d.median()), p90_abs=float(d.quantile(0.9)),
                within_010=float((d <= 0.10).mean()),
                within_2sigma=float((d <= 2.0 * a["sigma_gps"]).mean()),
                gps_tierA_share=float(a["rank_gps"].mean()),
                gps_tierA_within_010=float((dA <= 0.10).mean()) if len(dA) else None)


def _v4_summary(v4: pd.DataFrame) -> dict:
    r = v4[v4["rank"] & np.isfinite(v4["t_odo"])]
    d = (r["t_prod"] - r["t_odo"]).abs()
    return dict(n_rankable=int(len(r)), gt_020=int((d > 0.2).sum()), gt_030=int((d > 0.3).sum()))


def _v5(root: str) -> dict:
    env = dict(os.environ, DATA_ROOT=root, PYTHONPATH=str(REPO))
    bat = subprocess.run([sys.executable, str(REPO / "scripts" / "gps_trust_battery.py")], env=env,
                         cwd=REPO, capture_output=True, text=True)
    cal = subprocess.run([sys.executable, str(REPO / "scripts" / "gps_trust_calibration.py"), "500"],
                         env=env, cwd=REPO, capture_output=True, text=True)
    return dict(battery_pass=bat.returncode == 0, calibration_pass=cal.returncode == 0,
                battery_tail=bat.stdout[-1500:], calibration_tail=cal.stdout[-1500:])


def evaluate_gates(cur: dict, base: dict) -> list[tuple[str, bool, str]]:
    g = []
    v5 = cur.get("v5") or {}
    g.append(("G1 trust battery + calibration", bool(v5.get("battery_pass") and v5.get("calibration_pass")), ""))
    bg = cur["v2"].get("burst_gate_ge10_bad_rankable")
    g.append(("G2 burst@gate bad-but-rankable", bg is not None and bg <= GATES["G2_burst_gate_bad_rankable_max"],
              f"{bg} <= {GATES['G2_burst_gate_bad_rankable_max']}"))
    cd, bd = cur["v2"].get("slow_drift_bad_rankable"), base["v2"].get("slow_drift_bad_rankable")
    g.append(("G3 slow-drift bad-but-rankable not worse", cd is not None and bd is not None and cd <= bd, f"{cd} vs {bd}"))
    v3 = cur["v3"]
    g.append(("G4 GPS-only emulation honest", v3["within_2sigma"] >= GATES["G4_v3_within_2sigma_min"]
              and (v3["gps_tierA_within_010"] is None or v3["gps_tierA_within_010"] >= GATES["G4_v3_tierA_within_010_min"]),
              f"within2σ {v3['within_2sigma']:.3f}, tierA≤0.10 {v3['gps_tierA_within_010']}"))
    ok5, det = True, []
    for tr in TRACKS:
        drop = 100.0 * (base["coverage"][tr]["obd"] - cur["coverage"][tr]["obd"])
        ok5 &= drop <= GATES["G5_coverage_drop_pp_max"]
        det.append(f"{tr} obd drop {drop:+.1f}pp")
    t1 = cur["coverage"]["pir"]["per_corner_obd"].get("T1", 0.0)
    ok5 &= t1 >= GATES["G5_pir_T1_min"]
    g.append(("G5 coverage + PIR T1", ok5, ", ".join(det) + f", PIR T1 {t1:.3f}"))
    v1c = cur["v1"].get("shift_std_median_m")
    v1b = base["v1"].get("shift_std_median_m")
    ok6 = all(cur["brake_std"][tr] is not None and base["brake_std"][tr] is not None
              and cur["brake_std"][tr] <= base["brake_std"][tr] for tr in TRACKS)
    ok6 &= (v1c is None or v1b is None or v1c <= v1b)
    g.append(("G6 along-track + brake scatter not worse", ok6, f"v1 {v1c} vs {v1b}; brake {cur['brake_std']} vs {base['brake_std']}"))
    g.append(("G7 V4 >0.3 s not more", cur["v4"]["gt_030"] <= base["v4"]["gt_030"],
              f"{cur['v4']['gt_030']} vs {base['v4']['gt_030']}"))
    return g


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--label", required=True)
    p.add_argument("--baseline-report")
    p.add_argument("--max-laps", type=int, default=40)
    p.add_argument("--skip-v5", action="store_true")
    p.add_argument("--seed", type=int, default=20261010)
    args = p.parse_args()
    os.environ["DATA_ROOT"] = args.root
    sys.path.insert(0, str(REPO / "scripts"))
    import validation_checks as vc
    from lap_analyzer.labeler import build_reference_index, load_track

    sessions = pd.concat([vc.list_sessions(tr) for tr in TRACKS], ignore_index=True)
    split = vc.session_split(sessions)
    held = {k for k, v in split.items() if v == "held"}
    rng = np.random.default_rng(args.seed)
    v2s, v3s, v4s = [], [], []
    for tr in TRACKS:
        ctx = vc.TrackCtx.load(tr)
        sids = [s for (t, s) in held if t == tr]
        laps = list(vc.clean_laps(ctx, sids, max_laps=args.max_laps))
        v2s.append(vc.run_v2(ctx, laps, build_reference_index(load_track(tr)), rng))
        v3s.append(vc.run_v3(ctx, laps))
        v4s.append(vc.run_v4(ctx, laps))
    v1 = vc.run_v1("pir", [s for (t, s) in held if t == "pir"])
    v1ok = v1[v1["valid"]] if len(v1) else v1
    report = dict(
        label=args.label, root=args.root, n_sessions_held=len(held),
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
    print(json.dumps({k: v for k, v in report.items() if k != "v5"}, indent=2, default=str))
    print(f"report: {out}")
    if args.baseline_report:
        base = json.loads(Path(args.baseline_report).read_text())
        rc = 0
        for name, ok, detail in evaluate_gates(report, base):
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
            rc |= 0 if ok else 1
        return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
