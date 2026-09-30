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
    POS_MIN_LAPS,
    POS_SPEED_OFFSET_S,
    LagEstimate,
    apply_gps_lag,
    estimate_gps_lag,
    estimate_position_lag,
    estimate_session_gps_lag,
    estimate_session_position_lag,
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
# estimate_position_lag / estimate_session_position_lag — path-length method
# (2026-09-27 decision: measure position lag directly instead of τ_speed − offset)
# ---------------------------------------------------------------------------

R_EARTH_M = 6371000.0
OVAL_RADIUS_M = 200.0


def _oval_lap(true_delay_s, duration=90.0, rate=27.0, fix_hz=9.0):
    """A synthetic lap on a circle of radius OVAL_RADIUS_M, arc-length-parametrised
    so cumulative path length equals cumulative OBD distance exactly (noiseless).
    GPS positions are the TRUE track positions evaluated true_delay_s late, held
    at ~fix_hz fresh fixes (sample-and-hold), like a real GPS receiver."""
    t = np.arange(0, duration, 1 / rate)
    v_mph = 40.0 + 15.0 * np.sin(2 * np.pi * t / 17.3)  # varies ~25-55 mph, never < 20
    v_mps = v_mph * 0.44704
    D = np.concatenate([[0.0], np.cumsum((v_mps[:-1] + v_mps[1:]) / 2.0 * np.diff(t))])

    fix_times = np.arange(0, duration, 1 / fix_hz)
    fix_D = np.interp(fix_times - true_delay_s, t, D, left=D[0], right=D[-1])
    angle = fix_D / OVAL_RADIUS_M
    fix_x = OVAL_RADIUS_M * np.sin(angle)
    fix_y = OVAL_RADIUS_M * (1 - np.cos(angle))

    lat0, lon0 = 45.0, -122.0
    c0 = np.cos(np.radians(lat0))
    fix_lat = lat0 + np.degrees(fix_y / R_EARTH_M)
    fix_lon = lon0 + np.degrees(fix_x / (R_EARTH_M * c0))

    idx = np.clip(np.searchsorted(fix_times, t, side="right") - 1, 0, len(fix_times) - 1)
    lat, lon = fix_lat[idx], fix_lon[idx]
    return t, lat, lon, v_mph


def test_estimate_position_lag_recovers_known_delay():
    d = 0.20 + OBD_HOLD_LEAD_S
    t, lat, lon, v_mph = _oval_lap(d)
    tau = estimate_position_lag(t, lat, lon, v_mph)
    assert tau == pytest.approx(0.20, abs=0.03)


def test_estimate_position_lag_nan_on_teleport():
    t, lat, lon, v_mph = _oval_lap(0.20 + OBD_HOLD_LEAD_S)
    lat = lat.copy()
    lat[len(lat) // 2] += 1.0  # ~100km jump, way past the 30m teleport threshold
    assert np.isnan(estimate_position_lag(t, lat, lon, v_mph))


def test_estimate_position_lag_nan_on_short_lap():
    t, lat, lon, v_mph = _oval_lap(0.20 + OBD_HOLD_LEAD_S, duration=3.0)
    assert np.isnan(estimate_position_lag(t, lat, lon, v_mph))


def test_estimate_position_lag_drops_nonfinite_positions_then_fits():
    # Contract: non-finite lat/lon rows are dropped before fresh-fix detection; if
    # >= 10 fresh fixes remain the fit proceeds on the remaining rows.
    t, lat, lon, v_mph = _oval_lap(0.20 + OBD_HOLD_LEAD_S)
    lat = lat.copy()
    lat[1000:1010] = np.nan
    tau = estimate_position_lag(t, lat, lon, v_mph)
    assert tau == pytest.approx(0.20, abs=0.05)


def test_estimate_position_lag_nan_when_too_few_finite_fixes_remain():
    t, lat, lon, v_mph = _oval_lap(0.20 + OBD_HOLD_LEAD_S)
    lat = np.full_like(lat, np.nan)
    assert np.isnan(estimate_position_lag(t, lat, lon, v_mph))


def test_estimate_position_lag_nan_on_constant_obd_speed():
    # Constant speed makes D(t) linear, so the residual std is flat across the tau grid.
    t, lat, lon, _ = _oval_lap(0.20 + OBD_HOLD_LEAD_S)
    v_const = np.full_like(t, 40.0)
    assert np.isnan(estimate_position_lag(t, lat, lon, v_const))


@pytest.mark.parametrize("delay", [-0.6, 1.9])
def test_estimate_position_lag_nan_when_edge_pinned(delay):
    # True delay outside the search range -> argmin lands on a grid endpoint.
    t, lat, lon, v_mph = _oval_lap(delay)
    assert np.isnan(estimate_position_lag(t, lat, lon, v_mph))


def _pos_session(delays, lap_s=90.0):
    """OBD session frame: lap k's GPS position is delayed by delays[k-1]."""
    parts = []
    for k, d in enumerate(delays, start=1):
        t, lat, lon, v_mph = _oval_lap(d + OBD_HOLD_LEAD_S, duration=lap_s)
        parts.append(pd.DataFrame({
            "t": t + (k - 1) * lap_s, "lap": k,
            "lat": lat, "long": lon, "speed_mph": v_mph,
        }))
    return pd.concat(parts, ignore_index=True)


def test_estimate_session_position_lag_obd_session_uses_path_method():
    # Asymmetric edge delays: if edge laps 1 and 4 were not excluded, the median
    # would move well away from the inner-lap median.
    df = _pos_session([0.9, 0.20, 0.25, 0.9])
    speed_lags = {k: _ok(0.5) for k in (1, 2, 3, 4)}
    tau_pos, source, n_laps = estimate_session_position_lag(df, speed_lags)
    assert source == "path"
    assert n_laps == 2
    assert tau_pos == pytest.approx(0.225, abs=0.03)  # median of inner laps 0.20, 0.25


def test_estimate_session_position_lag_too_few_eligible_laps_uses_offset():
    assert POS_MIN_LAPS == 2
    df = _pos_session([0.20, 0.20, 0.9])  # 3 laps -> only lap 2 is non-edge
    speed_lags = {k: _ok(0.5) for k in (1, 2, 3)}
    tau_pos, source, n_laps = estimate_session_position_lag(df, speed_lags)
    assert source == "offset"
    assert n_laps == 3
    assert tau_pos == pytest.approx(0.5 - POS_SPEED_OFFSET_S)


def test_estimate_session_position_lag_no_finite_speed_tau_is_offset_default_not_nan():
    df = _pos_session([0.2, 0.2, 0.2, 0.2])
    df["speed_mph"] = np.nan
    speed_lags = {k: LagEstimate(float("nan"), float("nan"), "none") for k in (1, 2, 3, 4)}
    tau_pos, source, n_laps = estimate_session_position_lag(df, speed_lags)
    assert source == "offset-default"
    assert np.isfinite(tau_pos)
    assert tau_pos == pytest.approx(LAG_DEFAULT_S - POS_SPEED_OFFSET_S)
    assert n_laps == 0


def test_estimate_session_position_lag_all_default_speed_laps_is_offset_default():
    df = _pos_session([0.2, 0.2, 0.2, 0.2])
    df["speed_mph"] = np.nan
    speed_lags = {k: _ok(LAG_DEFAULT_S, "default") for k in (1, 2, 3, 4)}
    tau_pos, source, _ = estimate_session_position_lag(df, speed_lags)
    assert source == "offset-default"
    assert tau_pos == pytest.approx(LAG_DEFAULT_S - POS_SPEED_OFFSET_S)


def test_estimate_session_position_lag_gps_only_falls_back_to_offset():
    df = _pos_session([0.2, 0.2, 0.2, 0.2])
    df["speed_mph"] = np.nan  # GPS-only session: path method has no OBD to check against
    speed_lags = {1: _ok(0.5, "accel"), 2: _ok(0.6, "accel"),
                  3: _ok(0.4, "accel"), 4: _ok(0.5, "accel")}
    tau_pos, source, _n_laps = estimate_session_position_lag(df, speed_lags)
    expect = float(np.median([e.tau_s - POS_SPEED_OFFSET_S for e in speed_lags.values()]))
    assert source == "offset"
    assert tau_pos == pytest.approx(expect)


def test_estimate_session_position_lag_disabled():
    df = _pos_session([0.2, 0.2, 0.2])
    tau_pos, source, n_laps = estimate_session_position_lag(df, {}, enabled=False)
    assert (tau_pos, source, n_laps) == (0.0, "disabled", 0)


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


def test_hold_semantics_speed_is_subset():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.45, 2: 0.30})
    before = set(df["speed_mph_gps"])
    after = set(out["speed_mph_gps"])
    assert after <= before


def test_hold_semantics_position_is_subset():
    # Position is re-timed too (2026-09-27): still sample-and-hold, so the set of
    # distinct (lat, long) values after correction is a subset of the input's.
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.45, 2: 0.30}, tau_pos=0.38)
    assert not out["lat"].equals(df["lat"])  # the position really was moved
    before = set(zip(df["lat"], df["long"]))
    after = set(zip(out["lat"], out["long"]))
    assert after <= before


def test_rows_take_the_fix_from_tau_later():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.5, 2: 0.5})
    # row 0 (t=0.0) holds the value logged at t=0.5 (row 5)
    assert out.loc[0, "speed_mph_gps"] == df.loc[5, "speed_mph_gps"]


def test_position_is_retimed_by_tau_pos():
    # tau_pos is now an explicit scalar, independent of tau_speed.
    # Row 0 (t=0.0) holds the fix logged at t=tau_pos, i.e. the last row with t <= tau_pos.
    df = _held_frame()
    tau_pos = 0.38
    out = apply_gps_lag(df, {1: 0.5, 2: 0.5}, tau_pos=tau_pos)
    t = df["t"].to_numpy()
    j = int(np.searchsorted(t, 0.0 + tau_pos, side="right") - 1)
    assert out.loc[0, "lat"] == df.loc[j, "lat"]
    assert out.loc[0, "long"] == df.loc[j, "long"]
    assert out.loc[0, "speed_mph_gps"] == df.loc[5, "speed_mph_gps"]


def test_lap_counter_moves_exactly_rows_within_tau_pos_of_boundary():
    df = _held_frame()
    tau_pos = 0.38
    out = apply_gps_lag(df, {1: 0.5, 2: 0.5}, tau_pos=tau_pos)
    t = df["t"].to_numpy()
    boundary = df.loc[df["lap"] == 2, "t"].iloc[0]
    expect_moved = (df["lap"] == 1) & (t + tau_pos >= boundary)
    moved = (out["lap"] == 2) & (df["lap"] == 1)
    assert moved.equals(expect_moved)
    assert len(out) == len(df)
    assert out["lap"].is_monotonic_increasing


def test_tail_holds_last_fix_no_nan():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.45, 2: 1.2})
    assert not out["speed_mph_gps"].isna().any()
    assert out["speed_mph_gps"].iloc[-1] == df["speed_mph_gps"].iloc[-1]
    assert not out["lat"].isna().any()
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


@pytest.mark.parametrize("tau_pos", [0.38, -0.2])
def test_source_index_never_goes_backward_across_seam(tau_pos):
    df = _held_frame()
    out = apply_gps_lag(df, {1: 1.2, 2: 0.3}, tau_pos=tau_pos)
    # speed_mph_gps and altitude_m increase monotonically with fix index in the
    # fixture; if the source index j ever goes backward, they would dip at the seam.
    speed = out["speed_mph_gps"].to_numpy()
    altitude = out["altitude_m"].to_numpy()
    assert np.all(np.diff(speed) >= 0)
    assert np.all(np.diff(altitude) >= 0)
    assert not np.array_equal(altitude, df["altitude_m"].to_numpy())  # position was moved


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_apply_gps_lag_rejects_non_finite_tau_pos(bad):
    with pytest.raises(ValueError):
        apply_gps_lag(_held_frame(), {1: 0.4, 2: 0.4}, tau_pos=bad)


def test_non_gps_columns_untouched():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.45, 2: 0.45})
    pd.testing.assert_series_equal(out["t"], df["t"])
    pd.testing.assert_series_equal(out["long_g"], df["long_g"])


def test_zero_tau_is_identity_with_duplicate_timestamps():
    # TrackAddict logs occasional duplicate timestamps; with τ = 0 (--no-gps-lag) every
    # row must pass through untouched, not take the last duplicate's value.
    df = _held_frame()
    df.loc[10, "t"] = df.loc[11, "t"]
    df.loc[10, "speed_mph_gps"] = -1.0
    out = apply_gps_lag(df, {1: 0.0, 2: 0.0}, tau_pos=0.0)
    pd.testing.assert_frame_equal(out.drop(columns=["gps_lag_s", "gps_pos_lag_s"]), df)
    assert (out["gps_pos_lag_s"] == 0).all()


def test_zero_tau_is_identity():
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.0, 2: 0.0}, tau_pos=0.0)
    pd.testing.assert_frame_equal(out.drop(columns=["gps_lag_s", "gps_pos_lag_s"]), df)
    assert (out["gps_pos_lag_s"] == 0).all()


def test_negative_tau_pos_is_allowed_no_clamp():
    # An explicit negative session τ_pos is allowed, no clamping.
    df = _held_frame()
    out = apply_gps_lag(df, {1: 0.05, 2: 0.05}, tau_pos=-0.07)
    assert not out["gps_pos_lag_s"].isna().any()
    np.testing.assert_allclose(out["gps_pos_lag_s"].to_numpy(), -0.07, atol=1e-4)


def test_gps_pos_lag_s_is_float32_and_equals_tau_pos_everywhere():
    df = _held_frame()
    tau_pos = 0.33
    out = apply_gps_lag(df, {1: 0.45, 2: 0.30}, tau_pos=tau_pos)
    assert out["gps_pos_lag_s"].dtype == np.float32
    np.testing.assert_allclose(out["gps_pos_lag_s"].to_numpy(), tau_pos, atol=1e-5)


def test_summarize_lags_counts_sources():
    lags = {1: LagEstimate(0.5, np.nan, "session"), 2: LagEstimate(0.4, 0.99, "lap"),
            3: LagEstimate(0.6, 0.99, "lap"), 4: LagEstimate(0.5, np.nan, "session")}
    s = summarize_lags(lags)
    assert s["method_version"] == METHOD_VERSION
    assert (s["n_lap"], s["n_session"], s["n_accel"], s["n_default"]) == (2, 2, 0, 0)
    assert s["session_median_s"] == pytest.approx(0.5)
    assert s["pos_speed_offset_s"] == pytest.approx(POS_SPEED_OFFSET_S)
    assert "pos_lag_s" not in s


def test_summarize_lags_adds_pos_lag_block_when_given():
    lags = {1: LagEstimate(0.5, 0.99, "lap"), 2: LagEstimate(0.5, 0.99, "lap")}
    s = summarize_lags(lags, pos_lag=(0.22, "path", 2))
    assert s["method_version"] == "gps-lag-v3"
    assert s["pos_lag_s"] == pytest.approx(0.22)
    assert s["pos_lag_source"] == "path"
    assert s["pos_lag_n_laps"] == 2
