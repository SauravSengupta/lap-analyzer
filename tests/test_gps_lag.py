"""Unit tests for lap_analyzer.gps_lag — synthetic data only, no I/O.

Contracts from docs/superpowers/specs/2026-09-18-gps-lag-correction-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.gps_lag import (
    LAG_DEFAULT_S,
    LAG_MIN_CORR,
    METHOD_VERSION,
    OBD_HOLD_LEAD_S,
    LagEstimate,
    apply_gps_lag,
    estimate_gps_lag,
    estimate_session_gps_lag,
    resolve_session_lags,
    summarize_lags,
    xcorr_lag,
)

RATE_HZ = 27.0  # TrackAddict log rate


def _speed(t):
    """A lap-like OBD speed trace (mph): straights and braking zones, never < 25."""
    return 70 + 30 * np.sin(2 * np.pi * t / 23.0) + 12 * np.sin(2 * np.pi * t / 6.7 + 1.0)


def _lap(delay_s, duration=90.0):
    """(t, v_obd, v_gps, long_g) where GPS speed is OBD speed delayed by delay_s."""
    t = np.arange(0, duration, 1 / RATE_HZ)
    v_obd = _speed(t)
    v_gps = _speed(t - delay_s)
    # long_g (canonical: + = accel) is the true acceleration in g, aligned with the row clock.
    long_g = np.gradient(v_obd * 0.44704, t) / 9.80665
    return t, v_obd, v_gps, long_g


# Spec §Testing: a known lag of 0.30 s and 0.62 s is recovered within 0.02 s (`lap`).
@pytest.mark.parametrize("true_lag", [0.30, 0.62])
def test_obd_path_recovers_known_lag(true_lag):
    # GPS vs OBD delay = net lag + the OBD hold lead the estimator subtracts.
    t, v_obd, v_gps, long_g = _lap(true_lag + OBD_HOLD_LEAD_S)
    est = estimate_gps_lag(t, v_obd, v_gps, long_g)
    assert est.source == "lap"
    assert est.tau_s == pytest.approx(true_lag, abs=0.02)
    assert est.corr >= LAG_MIN_CORR


def test_xcorr_finds_negative_lag_within_search_range():
    t = np.arange(0, 60, 1 / RATE_HZ)
    tau, corr = xcorr_lag(t, _speed(t), _speed(t + 0.2))  # sig LEADS ref by 0.2 s
    assert tau == pytest.approx(-0.2, abs=0.02)
    assert corr > 0.99


def test_short_lap_is_not_accepted():
    # < LAG_MIN_SPAN_S of qualifying data -> not a per-lap estimate.
    t, v_obd, v_gps, long_g = _lap(0.5, duration=20.0)
    assert estimate_gps_lag(t, v_obd, v_gps, long_g).source == "none"


def test_low_speed_rows_are_excluded():
    # A lap that is mostly pit-lane crawl (< 20 mph) has too little qualifying span.
    t, v_obd, v_gps, long_g = _lap(0.5)
    slow = t < 70
    v_obd = np.where(slow, 10.0, v_obd)
    v_gps = np.where(slow, 10.0, v_gps)
    assert estimate_gps_lag(t, v_obd, v_gps, long_g).source == "none"


def test_uncorrelated_gps_is_not_accepted():
    t, v_obd, _, long_g = _lap(0.5)
    rng = np.random.default_rng(0)
    v_gps = 60 + rng.normal(0, 15, len(t))
    est = estimate_gps_lag(t, v_obd, v_gps, long_g)
    assert est.source == "none"
    assert np.isnan(est.tau_s)


def test_no_obd_uses_accelerometer_path():
    # GPS-only session: OBD speed all-NaN -> accel path; nothing subtracted.
    t, _, v_gps, long_g = _lap(0.5)
    v_obd = np.full(len(t), np.nan)
    est = estimate_gps_lag(t, v_obd, v_gps, long_g)
    assert est.source == "accel"
    assert est.tau_s == pytest.approx(0.5, abs=0.05)


def test_no_obd_and_flat_long_g_gives_none():
    t, _, v_gps, _ = _lap(0.5)
    est = estimate_gps_lag(t, np.full(len(t), np.nan), v_gps, np.zeros(len(t)))
    assert est.source == "none"


# ---------------------------------------------------------------------------
# resolve_session_lags / estimate_session_gps_lag — the fallback ladder
# ---------------------------------------------------------------------------

def _ok(tau, src="lap"):
    return LagEstimate(tau, 0.99, src)


NONE = LagEstimate(float("nan"), float("nan"), "none")


def test_failed_lap_falls_back_to_session_median():
    per_lap = {1: _ok(0.9), 2: _ok(0.4), 3: NONE, 4: _ok(0.5), 5: _ok(0.6), 6: _ok(0.9)}
    out = resolve_session_lags(per_lap, edge_laps={1, 6})
    assert out[3].source == "session"
    assert out[3].tau_s == pytest.approx(0.5)  # median of laps 2,4,5 (edges excluded)
    assert out[2] == per_lap[2]                # accepted laps keep their own τ


def test_edge_laps_use_session_median_even_when_they_estimated():
    per_lap = {1: _ok(1.2), 2: _ok(0.4), 3: _ok(0.6), 4: _ok(0.1)}
    out = resolve_session_lags(per_lap, edge_laps={1, 4})
    assert out[1].source == out[4].source == "session"
    assert out[1].tau_s == pytest.approx(0.5)


def test_per_lap_movement_is_not_squashed():
    # 20260725-111619 moved 0.55 -> 0.80 s lap to lap at corr 0.99+; keep it.
    per_lap = {1: _ok(0.6), 2: _ok(0.55), 3: _ok(0.80), 4: _ok(0.6)}
    out = resolve_session_lags(per_lap, edge_laps={1, 4})
    assert out[2].tau_s == 0.55 and out[3].tau_s == 0.80


def test_accel_estimates_resolve_like_lap_estimates():
    per_lap = {1: NONE, 2: _ok(0.5, "accel"), 3: NONE, 4: NONE}
    out = resolve_session_lags(per_lap, edge_laps={1, 4})
    assert out[2].source == "accel"
    assert out[3].source == "session" and out[3].tau_s == pytest.approx(0.5)


def test_nothing_accepted_gives_default():
    out = resolve_session_lags({1: NONE, 2: NONE, 3: NONE}, edge_laps={1, 3})
    assert all(e.source == "default" and e.tau_s == LAG_DEFAULT_S for e in out.values())


def _session(delays, lap_s=90.0, obd=True, flat_long_g=False):
    """Canonical frame of consecutive laps; lap k's GPS is delayed by delays[k-1]."""
    parts = []
    for k, d in enumerate(delays, start=1):
        t, v_obd, v_gps, long_g = _lap(d, duration=lap_s)
        parts.append(pd.DataFrame({
            "t": t + (k - 1) * lap_s, "lap": k,
            "speed_mph": v_obd if obd else np.nan, "speed_mph_gps": v_gps,
            "long_g": 0.0 if flat_long_g else long_g,
        }))
    return pd.concat(parts, ignore_index=True)


def test_session_estimates_each_lap_and_resolves_edges():
    d = [0.5, 0.40 + OBD_HOLD_LEAD_S, 0.60 + OBD_HOLD_LEAD_S, 0.5]
    lags = estimate_session_gps_lag(_session(d))
    assert set(lags) == {1, 2, 3, 4}
    assert lags[2].source == lags[3].source == "lap"
    assert lags[2].tau_s == pytest.approx(0.40, abs=0.02)
    assert lags[1].source == lags[4].source == "session"


def test_session_without_obd_uses_accel():
    lags = estimate_session_gps_lag(_session([0.5] * 4, obd=False))
    assert {e.source for e in lags.values()} <= {"accel", "session"}
    assert any(e.source == "accel" for e in lags.values())


def test_session_without_obd_and_flat_long_g_is_default():
    lags = estimate_session_gps_lag(_session([0.5] * 4, obd=False, flat_long_g=True))
    assert all(e.source == "default" for e in lags.values())


def test_disabled_is_zero_everywhere():
    lags = estimate_session_gps_lag(_session([0.5] * 3), enabled=False)
    assert all(e.source == "disabled" and e.tau_s == 0.0 for e in lags.values())


# ---------------------------------------------------------------------------
# apply_gps_lag — sample-and-hold re-timing
# ---------------------------------------------------------------------------

def _held_frame():
    """10 Hz rows, GPS fixes held for 3 rows (like ~8 Hz fixes on a 27 Hz log), 2 laps."""
    n = 60
    t = np.arange(n) * 0.1
    fix = np.arange(n) // 3
    return pd.DataFrame({
        "t": t,
        "lap": np.where(t < 3.0, 1, 2),
        "lat": 45.0 + fix * 1e-5, "long": -122.0 - fix * 1e-5,
        "altitude_m": 100.0 + fix, "gps_accuracy_m": 3.0,
        "speed_mph_gps": 50.0 + fix, "long_g": 0.0,
    })


def test_hold_semantics_positions_are_subset():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.45, 2: 0.30})
    before = set(zip(df["lat"], df["long"]))
    after = set(zip(out["lat"], out["long"]))
    assert after <= before


def test_rows_take_the_fix_from_tau_later():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.5, 2: 0.5})
    # row 0 (t=0.0) holds the value logged at t=0.5 (row 5)
    assert out.loc[0, "speed_mph_gps"] == df.loc[5, "speed_mph_gps"]


def test_lap_counter_moves_exactly_rows_within_tau_before_boundary():
    df = _held_frame()                     # boundary at t = 3.0 (row 30)
    out = apply_gps_lag(df, {1: 0.45, 2: 0.45})
    assert len(out) == len(df)
    moved = (out["lap"] != df["lap"])
    # rows with 2.55 <= t < 3.0 (t + 0.45 >= 3.0) now belong to lap 2
    expected = (df["t"] >= 3.0 - 0.45 - 1e-9) & (df["t"] < 3.0)
    assert (moved == expected).all()
    assert out["lap"].is_monotonic_increasing


def test_tail_holds_last_fix_no_nan():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.45, 2: 1.2})
    assert not out[["lat", "long", "speed_mph_gps", "lap"]].isna().any().any()
    assert out["lat"].iloc[-1] == df["lat"].iloc[-1]


def test_gps_lag_s_is_blended_and_equals_tau_at_each_lap_midpoint():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.45, 2: 0.30})
    assert out["gps_lag_s"].dtype == np.float32

    t = df["t"].to_numpy()
    lag = out["gps_lag_s"].to_numpy()

    # Lap 1: t in [0, 2.9], midpoint 1.45. Lap 2: t in [3.0, 5.9], midpoint 4.45.
    mid1_idx = int(np.argmin(np.abs(t - 1.45)))
    mid2_idx = int(np.argmin(np.abs(t - 4.45)))
    assert lag[mid1_idx] == pytest.approx(0.45, abs=0.01)
    assert lag[mid2_idx] == pytest.approx(0.30, abs=0.01)

    # Between the two midpoints, gps_lag_s is monotone (decreasing, since 0.45 -> 0.30).
    between = lag[mid1_idx:mid2_idx + 1]
    assert np.all(np.diff(between) <= 1e-9)


def test_source_index_never_goes_backward_across_seam():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 1.2, 2: 0.3})
    # altitude_m increases monotonically with fix index in the fixture; if the source
    # index j ever goes backward, altitude_m (and lap) would dip at the seam.
    alt = out["altitude_m"].to_numpy()
    assert np.all(np.diff(alt) >= 0)
    assert np.all(np.diff(out["lap"].to_numpy()) >= 0)


def test_non_gps_columns_untouched():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.45, 2: 0.45})
    pd.testing.assert_series_equal(out["t"], df["t"])
    pd.testing.assert_series_equal(out["long_g"], df["long_g"])


def test_zero_tau_is_identity():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.0, 2: 0.0})
    pd.testing.assert_frame_equal(out.drop(columns="gps_lag_s"), df)


def test_summarize_lags_counts_sources():
    lags = {1: LagEstimate(0.5, np.nan, "session"), 2: LagEstimate(0.4, 0.99, "lap"),
            3: LagEstimate(0.6, 0.99, "lap"), 4: LagEstimate(0.5, np.nan, "session")}
    s = summarize_lags(lags)
    assert s["method_version"] == METHOD_VERSION
    assert (s["n_lap"], s["n_session"], s["n_accel"], s["n_default"]) == (2, 2, 0, 0)
    assert s["session_median_s"] == pytest.approx(0.5)
