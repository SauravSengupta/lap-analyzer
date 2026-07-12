"""Clean-room tests for lap_analyzer.trajectory (Stage 1: the corridor).

Source of truth: tests/SPEC.md (## trajectory.Corridor / build_corridor) +
docs/superpowers/specs/2026-07-11-unified-gps-trust-trajectory-design.md. No
implementation was read to derive expected BEHAVIOUR; expectations come from the
documented contract (envelope EM-trim, lateral-offset sign, NaN-safe κ, save/load).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.trajectory import (
    CALIB_VERSION, ENV_CAP_M, Corridor, _fit_envelope, build_corridor,
    load_corridor,
)


def _straight_corridor(n=5) -> Corridor:
    """A minimal corridor: eastbound tangent, centerline at the origin per bin."""
    return Corridor(
        s_bin=np.arange(n) * 10.0,
        e_lo=np.full(n, -15.0), e_hi=np.full(n, 15.0),
        tx=np.ones(n), ty=np.zeros(n),          # travel due +x (east)
        gx=np.zeros(n), gy=np.zeros(n),
        kappa_signed=np.zeros(n),
        meta={"calib_version": CALIB_VERSION},
    )


# --- lateral_offset sign convention -----------------------------------------

def test_lateral_offset_positive_is_left_of_travel():
    # SPEC: Corridor.lateral_offset — + = left of travel. Travelling east (+x),
    # a point to the north (+y) is on the left → positive offset; south → negative.
    corr = _straight_corridor()
    off = corr.lateral_offset(np.array([0.0, 0.0]), np.array([5.0, -5.0]),
                              np.array([0.0, 0.0]))
    assert off[0] == pytest.approx(5.0)    # north of an eastbound car = left
    assert off[1] == pytest.approx(-5.0)   # south = right


def test_bin_index_clips_to_range():
    # SPEC: Corridor.bin_index — track_dist_m → clipped bin indices.
    corr = _straight_corridor(n=5)          # bins at 0,10,20,30,40
    idx = corr.bin_index(np.array([-100.0, 0.0, 24.0, 26.0, 1e6]))
    assert idx.tolist() == [0, 0, 2, 3, 4]


def test_bin_index_maps_non_finite_to_valid_bin():
    # SPEC: non-finite positions (NaN/inf) map to a defined bin, not an undefined
    # int cast. Result must stay in [0, n-1] and never crash.
    corr = _straight_corridor(n=5)
    idx = corr.bin_index(np.array([np.nan, np.inf, -np.inf, 30.0]))
    assert idx.dtype.kind == "i"
    assert np.all((idx >= 0) & (idx <= 4))
    assert idx[0] == 0 and idx[3] == 3   # NaN → bin 0; finite value unaffected


# --- EM-trim convergence (judge-mandated) -----------------------------------

def test_em_trim_purges_mode4_contamination():
    # SPEC: build_corridor envelope is built in two EM-trim passes; on clean votes
    # plus a Mode-4 contamination cluster the 2-pass envelope recovers the clean
    # ribbon in the contaminated bins, where 1-pass is dragged toward the cap.
    n_bins = 40
    clean_bin = 5
    dirty_bin = 25
    # 60 clean laps: per-bin offset spans a deterministic [-10, +10] across laps.
    clean = np.linspace(-10.0, 10.0, 60)
    votes = []
    for v in clean:
        votes.append((np.arange(n_bins), np.full(n_bins, v)))
    # 5 Mode-4 laps: drifted to -30m ONLY in bins 20..30, clean elsewhere.
    for _ in range(5):
        off = np.zeros(n_bins)
        off[20:31] = -30.0
        votes.append((np.arange(n_bins), off))

    lo1, hi1 = _fit_envelope(votes, n_bins, reject=None)          # pass 1
    lo2, hi2 = _fit_envelope(votes, n_bins, reject=(lo1, hi1))    # pass 2 (EM-trim)

    # Clean bin: both passes agree and sit near the clean p2 (≈ -10, +pad).
    assert lo2[clean_bin] == pytest.approx(lo1[clean_bin], abs=1.0)
    assert lo1[clean_bin] < -8.0
    # Contaminated bin: pass 1 is dragged far below the clean ribbon; pass 2
    # purges the -30 cluster and recovers to ≈ the clean bin's floor.
    assert lo1[dirty_bin] < lo2[dirty_bin] - 5.0            # 2-pass materially tighter
    assert lo2[dirty_bin] == pytest.approx(lo2[clean_bin], abs=2.0)  # ribbon recovered


def test_fit_envelope_respects_cap_and_ordering():
    # SPEC invariants: e_lo <= e_hi and |e_lo|,|e_hi| <= ENV_CAP_M.
    n_bins = 10
    votes = [(np.arange(n_bins), np.full(n_bins, v)) for v in np.linspace(-50, 50, 60)]
    lo, hi = _fit_envelope(votes, n_bins, reject=None)
    assert np.all(lo <= hi)
    assert np.all(lo >= -ENV_CAP_M - 1e-9)
    assert np.all(hi <= ENV_CAP_M + 1e-9)


# --- save / load round-trip + calib gate ------------------------------------

def test_corridor_save_load_roundtrip(tmp_path, monkeypatch):
    # SPEC: load_corridor round-trips arrays + meta.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    from lap_analyzer.trajectory import _save_corridor
    corr = _straight_corridor(n=7)
    object.__setattr__(corr, "meta", {"calib_version": CALIB_VERSION, "n_laps": 42})
    _save_corridor("ridge", corr)
    back = load_corridor("ridge")
    assert np.array_equal(back.s_bin, corr.s_bin)
    assert np.array_equal(back.e_lo, corr.e_lo)
    assert np.array_equal(back.kappa_signed, corr.kappa_signed)
    assert back.meta["n_laps"] == 42


def test_load_corridor_rejects_stale_calib_version(tmp_path, monkeypatch):
    # SPEC: load raises when the persisted calib_version differs from the code's.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    from lap_analyzer.trajectory import _save_corridor
    corr = _straight_corridor(n=4)
    object.__setattr__(corr, "meta", {"calib_version": "corridor-vOLD"})
    _save_corridor("ridge", corr)
    with pytest.raises(ValueError, match="calib_version"):
        load_corridor("ridge")


def test_load_corridor_rejects_changed_centerline(tmp_path, monkeypatch):
    # SPEC: load raises when the stored input_hash no longer matches the current
    # centerline (a stale corridor must be rebuilt, not silently used).
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    import lap_analyzer.trajectory as traj
    monkeypatch.setattr(traj, "load_centerline", lambda track: pd.DataFrame(
        {"track_dist_m": [0.0, 1.0], "lat": [0.0, 0.0], "long": [0.0, 0.0]}))
    corr = _straight_corridor(n=4)
    object.__setattr__(corr, "meta", {"calib_version": CALIB_VERSION, "input_hash": "deadbeef"})
    traj._save_corridor("ridge", corr)
    with pytest.raises(ValueError, match="different centerline|rebuild"):
        load_corridor("ridge")


# --- integration on the committed sample bundle -----------------------------

def test_build_corridor_structural(sample_data_root):
    # SPEC invariants on a real (small) corpus: aligned arrays, monotone 10m bins,
    # ordered/​capped envelope, all-finite NaN-safe κ.
    corr = build_corridor("ridge", save=False)
    n = len(corr.s_bin)
    for a in (corr.e_lo, corr.e_hi, corr.tx, corr.ty, corr.gx, corr.gy, corr.kappa_signed):
        assert len(a) == n
    assert np.allclose(np.diff(corr.s_bin), 10.0)
    assert np.all(corr.e_lo <= corr.e_hi)
    assert np.all(np.abs(corr.e_lo) <= ENV_CAP_M + 1e-9)
    assert np.all(np.abs(corr.e_hi) <= ENV_CAP_M + 1e-9)
    assert np.all(np.isfinite(corr.kappa_signed))      # NaN-safe even with OBD dropout
    assert np.allclose(np.hypot(corr.tx, corr.ty), 1.0, atol=1e-6)  # unit tangents
