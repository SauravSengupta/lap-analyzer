"""evaluate_gates reports PASS / FAIL / N/A honestly (missing evidence is never PASS)."""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path


def _tv():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("trust_validation")


def _report():
    cov = {"obd": 0.90, "per_corner_obd": {"T1": 0.80}}
    return {
        "v1": {"shift_std_median_m": 1.0},
        "v2": {"burst_gate_ge10_bad_rankable": 0.01, "slow_drift_bad_rankable": 0.02},
        "v3": {"n": 100, "within_2sigma": 0.97, "gps_tierA_within_010": 0.99},
        "v4": {"n_rankable": 50, "gt_020": 2, "gt_030": 1},
        "v5": {"battery_pass": True, "calibration_pass": True},
        "coverage": {"pir": dict(cov), "ridge": dict(cov)},
        "brake_std": {"pir": 3.0, "ridge": 3.0},
    }


def _status(gates):
    return {n.split()[0]: s for n, s, _ in gates}


def test_all_evidence_within_thresholds_all_pass():
    tv = _tv()
    st = _status(tv.evaluate_gates(_report(), _report()))
    assert set(st) == {"G1", "G2", "G3", "G4", "G5", "G6", "G7"}
    assert all(v == "PASS" for v in st.values()), st


def test_g4_tier_a_missing_is_na_not_pass():
    tv = _tv()
    cur = _report()
    cur["v3"]["gps_tierA_within_010"] = None
    assert _status(tv.evaluate_gates(cur, _report()))["G4"] == "N/A"
    cur["v3"]["within_2sigma"] = 0.5
    assert _status(tv.evaluate_gates(cur, _report()))["G4"] == "FAIL"


def test_g6_v1_missing_is_na_and_brake_worse_fails():
    tv = _tv()
    cur, base = _report(), _report()
    cur["v1"]["shift_std_median_m"] = None
    base["v1"]["shift_std_median_m"] = None
    assert _status(tv.evaluate_gates(cur, base))["G6"] == "N/A"
    cur["brake_std"] = {"pir": 9.0, "ridge": 3.0}
    assert _status(tv.evaluate_gates(cur, base))["G6"] == "FAIL"


def test_g1_na_when_v5_skipped():
    tv = _tv()
    cur = _report()
    cur["v5"] = None
    gates = tv.evaluate_gates(cur, _report())
    g1 = next(x for x in gates if x[0].startswith("G1"))
    assert g1[1] == "N/A"
    assert "not held-out" in g1[2]


def test_g2_burst_bad_rankable_fails():
    tv = _tv()
    cur = _report()
    cur["v2"]["burst_gate_ge10_bad_rankable"] = 0.5
    assert _status(tv.evaluate_gates(cur, _report()))["G2"] == "FAIL"


def test_baseline_missing_key_is_na_not_keyerror():
    tv = _tv()
    base = copy.deepcopy(_report())
    del base["v4"]
    gates = tv.evaluate_gates(_report(), base)
    g7 = next(x for x in gates if x[0].startswith("G7"))
    assert g7[1] == "N/A" and "baseline lacks" in g7[2]


def test_summaries_tolerate_empty_frames():
    import pandas as pd
    tv = _tv()
    assert tv._v3_summary(pd.DataFrame())["n"] == 0
    assert tv._v4_summary(pd.DataFrame())["n_rankable"] == 0


def test_g2_detail_prints_burst_gate_denominator():
    tv = _tv()
    cur = _report()
    cur["v2"]["burst_gate_ge10_n"] = 37
    g2 = next(x for x in tv.evaluate_gates(cur, _report()) if x[0].startswith("G2"))
    assert g2[1] == "PASS"
    assert g2[2].endswith("n=37"), g2[2]


def test_g5_missing_pir_t1_is_na_not_fail():
    tv = _tv()
    cur, base = _report(), _report()
    cur["coverage"]["pir"]["per_corner_obd"] = {"T2": 0.95}
    g5 = next(x for x in tv.evaluate_gates(cur, base) if x[0].startswith("G5"))
    assert g5[1] == "N/A", g5
    assert "PIR T1 missing" in g5[2]
