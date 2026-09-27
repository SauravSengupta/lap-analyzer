"""Unit tests for lap_analyzer.gps_lag — synthetic data only, no I/O.

Contracts from docs/superpowers/specs/2026-09-18-gps-lag-correction-design.md.
"""
from __future__ import annotations

import numpy as np
import pytest

from lap_analyzer.gps_lag import (
    LAG_MIN_CORR,
    OBD_HOLD_LEAD_S,
    estimate_gps_lag,
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
