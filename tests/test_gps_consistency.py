"""Tests for lap_analyzer.gps_consistency (spec 2026-10-10 sensor-fusion rethink §W1).

The path shape comes from GPS-independent sensors (odometer speed + lat_g yaw); each
fix's residual is its distance from that shape rigidly fitted to its neighbours.
Synthetic laps here are generated ANALYTICALLY (circle geometry), not with the module's
own integrator, so the lat_g sign convention (+ = right turn) is tested, not assumed.
"""
from __future__ import annotations

import numpy as np

from lap_analyzer.gps_consistency import (
    JUMP_MIN_NB,
    fix_residuals,
    fresh_fix_rows,
)

G = 9.80665


def _right_arc(v=40.0, radius=150.0, hz=8.0, seconds=12.0, straight_s=3.0):
    """Straight east for straight_s, then a right-hand circle of `radius`."""
    t = np.arange(0.0, seconds, 1.0 / hz)
    x = np.empty_like(t)
    y = np.empty_like(t)
    lat_g = np.zeros_like(t)
    w = v / radius
    for i, ti in enumerate(t):
        if ti <= straight_s:
            x[i], y[i] = v * ti, 0.0
        else:
            u = ti - straight_s
            x[i] = v * straight_s + radius * np.sin(w * u)
            y[i] = -radius + radius * np.cos(w * u)        # centre to the right (south)
            lat_g[i] = v * v / radius / G                   # + = right turn
    return t, np.full_like(t, v), lat_g, x, y


def test_clean_right_arc_has_small_residuals():
    # SPEC W1: on consistent sensors the residual is ~0 everywhere it is evaluable.
    t, v, lg, x, y = _right_arc()
    fr = np.arange(len(t))
    res = fix_residuals(t, v, lg, x, y, fr)
    assert np.isfinite(res).sum() > 0.8 * len(res)
    assert np.nanmax(res) < 1.0


def test_wrong_lat_g_sign_is_inconsistent():
    # The convention is load-bearing: flipping lat_g (a left-turn reading on a right-hand
    # arc) makes the arc fixes disagree with the sensor path.
    t, v, lg, x, y = _right_arc()
    res = fix_residuals(t, v, -lg, x, y, np.arange(len(t)))
    assert np.nanmax(res) > 3.0


def test_planted_spike_is_isolated():
    # SPEC W1: a single 15 m spike gets a large residual; its neighbours stay small
    # (the robust fit does not let one bad fix drag the path).
    t, v, lg, x, y = _right_arc()
    k = 60
    y2 = y.copy()
    y2[k] += 15.0
    res = fix_residuals(t, v, lg, x, y2, np.arange(len(t)))
    assert res[k] > 10.0
    others = np.delete(res, k)
    assert np.nanpercentile(others, 95) < 3.0


def test_planted_burst_is_flagged():
    # SPEC W1: a 0.75 s burst (6 fixes at 8 Hz) displaced 15 m is detected on every fix.
    t, v, lg, x, y = _right_arc()
    x2 = x.copy()
    x2[50:56] += 15.0
    res = fix_residuals(t, v, lg, x2, y, np.arange(len(t)))
    assert np.all(res[50:56] > 6.0)


def test_partial_nan_inputs_are_not_judged():
    # Contract: NaN (never a judgement) when it cannot judge. A missing lat_g or speed
    # sample inside a fix's ±JUMP_WIN_S window must not be zero-filled into the sensor
    # path and scored; fixes whose window is clean stay finite and small.
    t, v, lg, x, y = _right_arc()
    fr = np.arange(len(t))

    lg2 = lg.copy()
    lg2[40:43] = np.nan
    res = fix_residuals(t, v, lg2, x, y, fr)
    near = np.min(np.abs(t[:, None] - t[40:43][None, :]), axis=1) <= 1.5
    far = np.min(np.abs(t[:, None] - t[40:43][None, :]), axis=1) >= 3.0
    assert near.any() and far.any()
    assert np.isnan(res[near]).all()
    assert np.isfinite(res[far]).all()
    assert np.nanmax(res[far]) < 1.0

    v2 = v.copy()
    v2[70] = np.nan
    res = fix_residuals(t, v2, lg, x, y, fr)
    near = np.abs(t - t[70]) <= 1.5
    far = np.abs(t - t[70]) >= 3.0
    assert np.isnan(res[near]).all()
    assert np.isfinite(res[far]).all()
    assert np.nanmax(res[far]) < 1.0


def test_disabled_conditions_return_nan():
    # SPEC W1: never flags when it cannot judge — flat lat_g, slow car, too few neighbours.
    t, v, lg, x, y = _right_arc()
    fr = np.arange(len(t))
    assert np.isnan(fix_residuals(t, v, np.zeros_like(lg), x, y, fr)).all()
    assert np.isnan(fix_residuals(t, np.full_like(v, 5.0), lg, x, y, fr)).all()
    sparse = fr[::8]                                    # 1 Hz fixes → < JUMP_MIN_NB neighbours
    assert JUMP_MIN_NB > 2
    assert np.isnan(fix_residuals(t, v, lg, x, y, sparse)).all()


def test_fresh_fix_rows():
    lat = np.array([1.0, 1.0, 2.0, np.nan, 3.0])
    lon = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    assert fresh_fix_rows(lat, lon).tolist() == [0, 2, 4]
