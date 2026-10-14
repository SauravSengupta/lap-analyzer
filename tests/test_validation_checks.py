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


class _IdentityRef:
    """Stub ReferenceIndex: track_dist = metres east of lon0 (frame x)."""
    def __init__(self, frame):
        self.frame = frame

    def lookup(self, lat, lon):
        x, _ = self.frame.to_xy(lat, lon)
        return np.asarray(x, float), np.zeros(len(x))


def _frame():
    from lap_analyzer.gates import TrackFrame
    return TrackFrame.from_centerline(pd.DataFrame({"lat": [45.0, 45.0], "long": [-122.0, -121.99]}))


def _east_lap(n=80, v=40.0, hz=8.0):
    fr = _frame()
    t = np.arange(n) / hz
    x = v * t
    lat = np.full(n, fr.lat0)
    lon = fr.lon0 + x / fr.m_per_deg_lon
    return pd.DataFrame({"t": t, "lat": lat, "long": lon, "track_dist_m": x,
                         "gps_drift_lat_m": 0.0, "gps_drift_lon_m": 0.0}), fr


def test_plant_offset_moves_exactly_the_target_rows():
    # SPEC W0/V2: planting moves only the requested rows, by exactly the requested vector,
    # and re-projects their track_dist_m; every other row is untouched.
    vc = _vc()
    g, fr = _east_lap()
    rows = np.zeros(len(g), bool)
    rows[10:14] = True
    dx = np.full(len(g), 12.0)
    dy = np.full(len(g), -5.0)
    h = vc.plant_offset(g, rows, dx, dy, fr, _IdentityRef(fr))
    x0, y0 = fr.to_xy(g["lat"], g["long"])
    x1, y1 = fr.to_xy(h["lat"], h["long"])
    assert np.allclose(x1[rows] - x0[rows], 12.0, atol=1e-6)
    assert np.allclose(y1[rows] - y0[rows], -5.0, atol=1e-6)
    assert np.allclose(h["track_dist_m"].to_numpy()[rows], g["track_dist_m"].to_numpy()[rows] + 12.0, atol=1e-6)
    assert h.loc[~rows].equals(g.loc[~rows])


def test_burst_rows_covers_n_fixes_and_their_held_rows():
    vc = _vc()
    g, _ = _east_lap(n=20)
    g.loc[5, ["lat", "long"]] = g.loc[4, ["lat", "long"]].to_numpy()  # row 5 holds fix 4
    m = vc.burst_rows(g, start_fix=3, n_fix=2)   # fixes at rows 3 and 4 (4 is held through row 5)
    assert m.tolist()[:7] == [False, False, False, True, True, True, False]


def test_slow_drift_is_smooth_bounded_and_local():
    # SPEC W0/V2: slow drift is a sin² bump of the requested amplitude, perpendicular to
    # travel, confined to ±half_len_m of the centre.
    vc = _vc()
    from lap_analyzer.trajectory import CALIB_VERSION, Corridor
    g, _ = _east_lap(n=120)
    nb = 60
    corr = Corridor(s_bin=np.arange(nb) * 10.0, e_lo=np.full(nb, -15.0), e_hi=np.full(nb, 15.0),
                    tx=np.ones(nb), ty=np.zeros(nb), gx=np.zeros(nb), gy=np.zeros(nb),
                    kappa_signed=np.zeros(nb), meta={"calib_version": CALIB_VERSION})
    rows, dx, dy = vc.slow_drift_offsets(g, center_m=300.0, amp_m=20.0, corridor=corr, half_len_m=75.0)
    s = g["track_dist_m"].to_numpy()
    assert rows.any() and not rows[np.abs(s - 300.0) > 75.0].any()
    assert np.allclose(dx[rows], 0.0, atol=1e-9)               # eastbound → offset is pure north/south
    assert np.max(np.abs(dy[rows])) == pytest.approx(20.0, abs=0.5)


def test_register_profiles_recovers_a_planted_shift():
    # SPEC W0/V1 self-test: a known along-track shift of one lap's road profile is
    # recovered by registration (the ruler reports itself INVALID otherwise).
    vc = _vc()
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 1200)).rolling(5, center=True, min_periods=1).mean().to_numpy()
    laps = np.array([base + rng.normal(0, 0.05, base.size) for _ in range(6)])
    shifted = laps.copy()
    shifted[2] = np.interp(np.arange(base.size) - 7, np.arange(base.size), laps[2])  # lap 2 is +7 m late
    R = vc.register_profiles(shifted, win=120, max_shift=25, step=40, min_corr=0.6)
    lap2 = R[(R["lap"] == 2) & (R["corr"] > 0.6)]
    assert len(lap2) > 10
    assert lap2["shift"].median() == pytest.approx(7.0, abs=1.0)
    others = R[(R["lap"] != 2) & (R["corr"] > 0.6)]
    assert others["shift"].abs().median() <= 1.0


def test_register_profiles_plant_m_self_test_recovers_shift():
    # SPEC W0/V1 self-test: plant_m shifts only the lap under test, so the recovered shift
    # equals plant_m (planting on every lap would cancel against the reference).
    vc = _vc()
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 1, 1200)).rolling(5, center=True, min_periods=1).mean().to_numpy()
    laps = np.array([base + rng.normal(0, 0.05, base.size) for _ in range(6)])
    R = vc.register_profiles(laps, win=120, max_shift=25, step=40, min_corr=0.6, plant_m=7.0)
    hi = R[R["corr"] > 0.6]
    assert len(hi) > 10
    assert hi["shift"].median() == pytest.approx(7.0, abs=1.0)


def test_runners_emit_documented_columns(sample_data_root):
    # SPEC W0: V2/V3/V4 runners return one row per evaluated transit with the
    # documented columns; V2 errors are finite for planted trials.
    vc = _vc()
    from lap_analyzer.labeler import build_reference_index, load_track
    ctx = vc.TrackCtx.load("ridge")
    sids = list(vc.list_sessions("ridge")["session_id"])
    laps = list(vc.clean_laps(ctx, sids, max_laps=2))
    assert laps, "sample bundle should contain at least one clean 8 Hz OBD lap"
    v2 = vc.run_v2(ctx, laps, build_reference_index(load_track("ridge")),
                   np.random.default_rng(1), trials_per_lap=3)
    assert {"kind", "where", "mag_m", "err_s", "rank", "caught"} <= set(v2.columns)
    assert set(v2["kind"]) <= set(vc.V2_KINDS)
    assert len(v2) > 0, "at least one fault must be planted on a clean lap"
    assert np.isfinite(v2["err_s"]).all()
    v3 = vc.run_v3(ctx, laps)
    assert {"t_obd", "rank_obd", "t_gps", "sigma_gps", "rank_gps"} <= set(v3.columns)
    v4 = vc.run_v4(ctx, laps)
    assert {"t_prod", "rank", "t_odo"} <= set(v4.columns)
    assert len(v3) == len(v4) == len(laps) * len(ctx.bounds)
