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
    cov = {"obd": 0.90, "per_corner_obd": {"T1": 0.80}, "per_corner_all": {"T1": 0.75}}
    return {
        "population": {"held": [["pir", "s1"], ["ridge", "s2"]],
                       "laps": [["pir", "s1", 3], ["ridge", "s2", 5]]},
        "v1": {"shift_std_median_m": 1.0},
        "v2": {"burst_gate_ge10_bad_rankable": 0.01, "slow_drift_bad_rankable": 0.02},
        "v3": {"n": 100, "within_2sigma": 0.97, "gps_tierA_within_010": 0.99, "gps_tierA_share": 0.6},
        "v4": {"n_rankable": 50, "gt_020": 2, "gt_030": 1, "gt_030_rate": 0.02},
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


def _gate(gates, key):
    return next(x for x in gates if x[0].startswith(key))


def test_g5_coverage_drop_beyond_2pp_fails_within_passes():
    # SPEC gates G5: OBD coverage may not drop by more than 2 pp.
    tv = _tv()
    base = _report()
    cur = _report()
    cur["coverage"]["ridge"]["obd"] = 0.87          # −3 pp
    assert _status(tv.evaluate_gates(cur, base))["G5"] == "FAIL"
    cur["coverage"]["ridge"]["obd"] = 0.89          # −1 pp
    assert _status(tv.evaluate_gates(cur, base))["G5"] == "PASS"


def test_g5_non_finite_coverage_is_na_not_pass():
    tv = _tv()
    cur = _report()
    cur["coverage"]["ridge"]["obd"] = float("nan")
    assert _status(tv.evaluate_gates(cur, _report()))["G5"] == "N/A"
    cur["coverage"]["ridge"]["obd"] = None
    assert _status(tv.evaluate_gates(cur, _report()))["G5"] == "N/A"


def test_g5_detail_prints_pir_t1_both_ways():
    # G5 stays on per_corner_obd (definition is the user's call) but shows both views.
    tv = _tv()
    g5 = _gate(tv.evaluate_gates(_report(), _report()), "G5")
    assert "PIR T1 obd=0.800 all=0.750" in g5[2], g5


def test_g7_compares_rate_not_count():
    # SPEC gates G7: the >0.3 s share may not rise; counts alone can differ with n.
    tv = _tv()
    base = _report()
    cur = _report()
    cur["v4"].update({"n_rankable": 100, "gt_030": 2, "gt_030_rate": 0.02})   # more count, same rate
    assert _status(tv.evaluate_gates(cur, base))["G7"] == "PASS"
    cur["v4"].update({"gt_030": 3, "gt_030_rate": 0.03})                      # rate up
    g7 = _gate(tv.evaluate_gates(cur, base), "G7")
    assert g7[1] == "FAIL"
    assert "n_rankable 100 vs 50" in g7[2]


def test_g3_g7_na_when_populations_differ_or_missing():
    # Paired comparison only: G3/G7 are meaningless across different lap populations.
    tv = _tv()
    cur = _report()
    cur["population"]["laps"].append(["pir", "s1", 4])
    for c in (cur, {k: v for k, v in _report().items() if k != "population"}):
        gates = tv.evaluate_gates(c, _report())
        for key in ("G3", "G7"):
            st, det = _gate(gates, key)[1:]
            assert st == "N/A" and "populations differ" in det


def test_population_equality_ignores_order():
    tv = _tv()
    cur = _report()
    cur["population"]["laps"] = list(reversed(cur["population"]["laps"]))
    assert tv.same_population(cur, _report())


def test_g4_detail_prints_tier_a_n():
    tv = _tv()
    g4 = _gate(tv.evaluate_gates(_report(), _report()), "G4")
    assert "n_tierA=60" in g4[2], g4


def test_v2_summary_counts_skips_per_batch_and_g2_includes_mixed():
    import pandas as pd
    tv = _tv()
    rows = [
        {"batch": "g2", "kind": "burst", "where": "gate", "mag_m": 10.0, "n_fix": 4, "planted": True,
         "err_s": 0.01, "rank": True, "caught": 1.0},
        {"batch": "g2", "kind": None, "where": None, "mag_m": float("nan"), "n_fix": 0, "planted": False,
         "err_s": float("nan"), "rank": False, "caught": float("nan")},
        {"batch": "mixed", "kind": "burst", "where": "gate", "mag_m": 20.0, "n_fix": 6, "planted": True,
         "err_s": 0.2, "rank": True, "caught": 0.0},
        {"batch": "mixed", "kind": "burst", "where": "gate", "mag_m": 5.0, "n_fix": 6, "planted": True,
         "err_s": 0.2, "rank": True, "caught": 0.0},
    ]
    s = tv._v2_summary(pd.DataFrame(rows))
    assert s["batches"] == {"g2": {"n_attempted": 2, "n_skipped": 1}, "mixed": {"n_attempted": 2, "n_skipped": 0}}
    assert s["burst_gate_ge10_n"] == 2 and s["burst_gate_ge10_n_skipped"] == 1
    assert s["burst_gate_ge10_bad_rankable"] == 0.5
