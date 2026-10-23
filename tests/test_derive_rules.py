"""Tests for the W2 constant-derivation rules (spec 2026-10-10 §W2). The rules are
re-statements of how each constant was originally defined; these tests pin the rule's
math on synthetic inputs with known answers."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def _dr():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("derive_trust_constants")


def test_robust_std_matches_gaussian_sigma_and_ignores_contamination():
    dr = _dr()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 6.0, 20000)
    assert dr.robust_std(x) == pytest.approx(6.0, rel=0.03)
    contaminated = np.concatenate([x, np.full(400, 60.0)])   # 2% gross outliers
    assert dr.robust_std(contaminated) == pytest.approx(6.0, rel=0.06)


def test_robust_bound_is_offset_plus_three_sigma():
    dr = _dr()
    rng = np.random.default_rng(1)
    x = rng.normal(-4.0, 2.0, 20000)
    assert dr.robust_bound(x) == pytest.approx(4.0 + 3 * 2.0, rel=0.03)


def test_fisher_z_mean_weights_by_n_minus_3():
    dr = _dr()
    r = dr.fisher_z_mean([0.5, 0.5, 0.9], [50, 50, 4])          # tiny-n 0.9 barely counts
    assert r == pytest.approx(0.5, abs=0.02)
    assert dr.fisher_z_mean([0.3, float("nan")], [30, 30]) == pytest.approx(0.3, abs=1e-9)


def test_fit_band_recovers_line():
    dr = _dr()
    h = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    scale, floor = dr.fit_band(h, 6.0 * h + 7.0)
    assert scale == pytest.approx(6.0, abs=1e-9) and floor == pytest.approx(7.0, abs=1e-9)


def _da():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("derive_accel_speed_tol")


def test_p99_ceil_is_p99_of_abs_rounded_up_to_tenth():
    # W3 amendment rule: ACCEL_SPEED_TOL_MPS = p99 of |resid|, rounded UP to 0.1 m/s.
    da = _da()
    x = np.concatenate([np.linspace(-1.0, 1.0, 9900), np.full(100, 50.0)])
    v = da.p99_ceil(x)
    p = float(np.percentile(np.abs(x), 99))
    assert v >= p and v - p < 0.1 + 1e-9
    assert round(v * 10) == pytest.approx(v * 10, abs=1e-9)        # on the 0.1 grid
    assert da.p99_ceil(np.full(100, 1.2)) == pytest.approx(1.2)      # exact grid value not bumped
    assert da.p99_ceil([0.5, float("nan"), -0.5] * 50) == pytest.approx(0.5)   # NaN ignored, |.| taken
