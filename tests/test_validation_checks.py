"""Tests for the W0 trust-validation harness machinery (scripts/validation_checks.py).

Source of truth: docs/superpowers/specs/2026-10-10-sensor-fusion-rethink-design.md §W0.
The harness is test infrastructure for the trust layer; these tests check that its own
machinery does what it claims (deterministic split, exact fault planting, shift
recovery), never values observed on real data.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _vc():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("validation_checks")


def _sessions(n_obd=10, n_gps=5):
    rows = [{"track": "ridge", "session_id": f"2026010{i:02d}-120000", "has_obd": True} for i in range(n_obd)]
    rows += [{"track": "ridge", "session_id": f"2026020{i:02d}-120000", "has_obd": False} for i in range(n_gps)]
    rows += [{"track": "pir", "session_id": f"2026030{i:02d}-120000", "has_obd": True} for i in range(n_obd)]
    return pd.DataFrame(rows)


def test_session_split_is_deterministic_and_total():
    # SPEC W0: deterministic hash split; every session assigned exactly once.
    vc = _vc()
    s = _sessions()
    a, b = vc.session_split(s), vc.session_split(s.sample(frac=1.0, random_state=3))
    assert a == b
    assert set(a) == {(r.track, r.session_id) for r in s.itertuples()}
    assert set(a.values()) <= {"cal", "held"}


def test_session_split_is_stratified_60_40():
    # SPEC W0: ~60% calibration within each (track, has_obd) stratum.
    vc = _vc()
    s = _sessions(n_obd=10, n_gps=5)
    split = vc.session_split(s)
    for (track, obd), grp in s.groupby(["track", "has_obd"]):
        n_cal = sum(split[(track, sid)] == "cal" for sid in grp.session_id)
        assert n_cal == round(0.6 * len(grp))


def test_fix_rows_marks_position_changes_only():
    vc = _vc()
    lat = np.array([45.0, 45.0, 45.1, 45.1, 45.1, 45.2])
    lon = np.array([-122.0, -122.0, -122.0, -122.0, -121.9, -121.9])
    assert vc.fix_rows(lat, lon).tolist() == [0, 2, 4, 5]


def test_odometer_section_time_on_constant_speed():
    # 40 m/s on a linear odometer: section 400 m → 10 s exactly.
    vc = _vc()
    t = np.arange(0, 60, 0.1)
    g = pd.DataFrame({"t": t, "dist_lap_m": 40.0 * t})
    assert vc.odometer_section_time(g, 200.0, 600.0) == pytest.approx(10.0, abs=1e-9)


def test_gps_only_emulation_drops_obd_and_keeps_lap_length():
    vc = _vc()
    t = np.arange(0, 50, 0.1)
    v_mph = 60.0 + 10.0 * np.sin(t / 5.0)
    g = pd.DataFrame({"t": t, "speed_mph": v_mph, "speed_mph_gps": v_mph * 0.99,
                      "dist_lap_m": np.linspace(0.0, 1500.0, len(t))})
    h = vc.gps_only_emulation(g)
    assert h["speed_mph"].isna().all()
    assert h["dist_lap_m"].iloc[0] == pytest.approx(0.0)
    assert h["dist_lap_m"].max() == pytest.approx(1500.0)
    assert np.all(np.diff(h["dist_lap_m"].to_numpy()) >= 0)
    assert g["speed_mph"].notna().all()   # input untouched
