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

def test_estimate_trajectory_monotone_and_aligned(sample_data_root):
    # SPEC: estimate_trajectory — s_hat is monotone non-decreasing and all
    # per-sample arrays are aligned; status is one of the three documented values.
    from lap_analyzer.analysis import load_centerline, load_samples
    from lap_analyzer.gates import TrackFrame
    from lap_analyzer.trajectory import build_corridor, estimate_trajectory
    corr = build_corridor("ridge", save=False)
    frame = TrackFrame.from_centerline(load_centerline("ridge"))
    s = load_samples("ridge", "20260517-100304")
    lap = int(s["lap"].iloc[len(s) // 2])
    traj = estimate_trajectory(s[s["lap"] == lap], corr, frame)
    assert np.all(np.diff(traj.s_hat) >= -1e-9)
    assert (len(traj.s_hat) == len(traj.t) == len(traj.sigma_m)
            == len(traj.delta_hat) == len(traj.dl))
    assert traj.status in ("ok", "rescale_invalid", "gps_backbone")


def test_estimate_trajectory_monotone_under_any_input_order(make_lap_samples):
    # SPEC (§5 PR1 property): s_hat is monotone non-decreasing regardless of input
    # row order — the estimator time-sorts internally, so a shuffled frame yields a
    # monotone s_hat aligned to that lap's own time-sorted order.
    from lap_analyzer.gates import TrackFrame
    from lap_analyzer.trajectory import estimate_trajectory
    corr = _straight_corridor(n=300)
    frame = TrackFrame.from_centerline(pd.DataFrame(
        {"lat": [45.0, 45.0], "long": [-122.0, -121.99]}))
    n = 200
    dl = np.linspace(0.0, 2500.0, n)
    lap = make_lap_samples(n=n, t=np.arange(n) * 0.1, dist_lap_m=dl,
                           track_dist_m=dl + 3.0, lat=np.full(n, 45.0),
                           long=np.full(n, -122.0), lat_g=np.zeros(n))
    shuffled = lap.sample(frac=1.0, random_state=1).reset_index(drop=True)
    traj = estimate_trajectory(shuffled, corr, frame)
    assert np.all(np.diff(traj.s_hat) >= -1e-9)
    assert np.all(np.diff(traj.t) >= 0.0)          # internally time-sorted


def test_time_at_returns_none_outside_range(sample_data_root):
    # SPEC: Trajectory.time_at — None when s is outside [s_hat.min, s_hat.max]
    # (np.interp would otherwise clamp and fabricate a crossing time).
    from lap_analyzer.analysis import load_centerline, load_samples
    from lap_analyzer.gates import TrackFrame
    from lap_analyzer.trajectory import build_corridor, estimate_trajectory
    corr = build_corridor("ridge", save=False)
    frame = TrackFrame.from_centerline(load_centerline("ridge"))
    s = load_samples("ridge", "20260517-100304")
    lap = int(s["lap"].iloc[len(s) // 2])
    traj = estimate_trajectory(s[s["lap"] == lap], corr, frame)
    assert traj.time_at(traj.s_hat.max() + 1000.0) is None
    assert traj.time_at(traj.s_hat.min() - 1000.0) is None
    assert traj.time_at(float(np.median(traj.s_hat))) is not None


def test_estimate_trajectory_gps_only_is_gps_backbone(make_lap_samples):
    # SPEC: a GPS-only lap (no OBD speed) → status 'gps_backbone' with the wider
    # GPS σ floor, even with GPS evidence present. dist_lap_m stays valid
    # (normalize integrates it from speed_mph_gps).
    from lap_analyzer.gates import TrackFrame
    from lap_analyzer.trajectory import SIGMA_FLOOR_GPS_M, estimate_trajectory
    corr = _straight_corridor(n=250)
    frame = TrackFrame.from_centerline(pd.DataFrame(
        {"lat": [45.0, 45.0], "long": [-122.0, -121.99]}))
    n = 200
    lap = make_lap_samples(
        n=n, speed_mph=np.full(n, np.nan), speed_mph_gps=np.full(n, 100.0),
        dist_lap_m=np.linspace(0, 2000, n), track_dist_m=np.linspace(0, 2000, n),
        lat=np.full(n, 45.0), long=np.full(n, -122.0), lat_g=np.zeros(n))
    traj = estimate_trajectory(lap, corr, frame)
    assert traj.status == "gps_backbone"
    assert np.all(traj.knot_sigma >= SIGMA_FLOOR_GPS_M - 1e-9)
    assert np.all(np.diff(traj.s_hat) >= -1e-9)


def test_estimate_trajectory_rescale_invalid_is_prior_only(make_lap_samples):
    # SPEC: a lap with an invalid dist_lap_m rescale (|median δ| > 150m) →
    # status 'rescale_invalid', δ̂ ≡ 0 (prior only), still monotone, nothing dropped.
    from lap_analyzer.gates import TrackFrame
    from lap_analyzer.trajectory import estimate_trajectory
    corr = _straight_corridor(n=250)
    frame = TrackFrame.from_centerline(pd.DataFrame(
        {"lat": [45.0, 45.0], "long": [-122.0, -121.99]}))
    lap = make_lap_samples(n=200, dist_lap_m=np.linspace(0, 2000, 200),
                           track_dist_m=np.linspace(0, 2000, 200) + 700.0)  # +700m offset
    traj = estimate_trajectory(lap, corr, frame)
    assert traj.status == "rescale_invalid"
    assert np.allclose(traj.delta_hat, 0.0)
    assert np.all(np.diff(traj.s_hat) >= -1e-9)


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


# ===========================================================================
# section_timing (Stage 2, item 5) — SPEC: tests/SPEC.md ## trajectory.section_timing
# ===========================================================================

def _straight_trajectory(sigma0=1.0, v=40.0, length=2500.0, n=250):
    """A controlled straight-line Trajectory: s_hat == dl (monotone), constant
    speed, constant per-knot σ, on-centerline (e_lat ≡ 0, κ ≡ 0). Lets a test
    isolate the correlation-aware section-σ formula from the nets."""
    from lap_analyzer.trajectory import Trajectory
    dl = np.linspace(0.0, length, n)
    t = dl / v
    knots = np.arange(dl.min(), dl.max(), 10.0)
    return Trajectory(
        t=t, s_hat=dl.copy(), sigma_m=np.full(n, sigma0), delta_hat=np.zeros(n),
        dl=dl, v=np.full(n, v), evidence=np.ones(n, bool), status="ok",
        knots=knots, knot_sigma=np.full(len(knots), sigma0),
        e_lat=np.zeros(n), checks=[])


def test_section_timing_always_emits_no_coverage():
    # SPEC: gate outside the lap's s_hat range → status 'no_coverage', NaN value,
    # tier C, not rank-eligible — a row is ALWAYS emitted (R10), never None.
    from lap_analyzer.trajectory import section_timing
    traj = _straight_trajectory(length=2000.0)
    st = section_timing(traj, 100.0, 5000.0, _straight_corridor(n=600))
    assert st.status == "no_coverage"
    assert np.isnan(st.time_s) and np.isnan(st.sigma_s)
    assert st.tier == "C" and st.rank_eligible is False


def test_section_timing_clean_straight_is_rankable_and_true():
    # SPEC: a clean section (tight σ, driven == b−a, on-ribbon) → status 'ok',
    # value = true traversal time, tier A, rank-eligible.
    from lap_analyzer.trajectory import section_timing
    v = 40.0
    traj = _straight_trajectory(sigma0=1.0, v=v, length=2500.0)
    a, b = 500.0, 1000.0
    st = section_timing(traj, a, b, _straight_corridor(n=300))
    assert st.status == "ok"
    assert st.time_s == pytest.approx((b - a) / v, rel=1e-6)
    assert st.driven_m == pytest.approx(b - a, abs=1e-6)
    assert st.tier == "A" and st.rank_eligible is True


def test_section_timing_sigma_is_correlation_aware():
    # SPEC: Var(T) = [σ_A² + σ_B² − 2ρσ_Aσ_B]/(v_A v_B). With equal per-gate σ and
    # ρ = RHO_SECTION > 0 the section σ is BELOW the independence value √2·σ/v.
    from lap_analyzer.trajectory import RHO_SECTION, section_timing
    sigma0, v = 2.0, 40.0
    traj = _straight_trajectory(sigma0=sigma0, v=v, length=2500.0)
    st = section_timing(traj, 500.0, 1000.0, _straight_corridor(n=300))
    expected = np.sqrt((sigma0**2 + sigma0**2 - 2 * RHO_SECTION * sigma0 * sigma0)
                       / (v * v))
    assert st.sigma_s == pytest.approx(expected, rel=1e-6)
    assert 0.0 < RHO_SECTION < 1.0
    assert st.sigma_s < np.sqrt(2) * sigma0 / v          # sharper than independent


def test_section_timing_r6_zero_gps_equals_obd_anchored(make_lap_samples):
    # SPEC R6: a lap with no accepted GPS evidence (δ̂≡0) → emitted time equals the
    # OBD-anchored fallback limit to 1e-6. With δ̂≡0, s_hat ≡ dist_lap_m, so the
    # crossing time reduces to interpolating t at the OBD-distance bounds (this is
    # exactly what the retired analysis._obd_anchored_time computed).
    from lap_analyzer.gates import TrackFrame
    from lap_analyzer.trajectory import estimate_trajectory, section_timing
    corr = _straight_corridor(n=300)
    frame = TrackFrame.from_centerline(pd.DataFrame(
        {"lat": [45.0, 45.0], "long": [-122.0, -121.99]}))
    n = 200
    dl = np.linspace(0.0, 2500.0, n)
    t = np.arange(n) * 0.1
    # constant track_dist_m → no fresh fixes → zero GPS evidence (δ̂≡0 prior only)
    lap = make_lap_samples(n=n, t=t, dist_lap_m=dl,
                           track_dist_m=np.full(n, 1250.0), lat_g=np.zeros(n))
    traj = estimate_trajectory(lap, corr, frame)
    assert not traj.evidence.any()
    a, b = 500.0, 1500.0
    st = section_timing(traj, a, b, corr)
    # OBD-anchored oracle: elapsed t between the OBD-distance crossings of a and b.
    expected = float(np.interp(b, dl, t) - np.interp(a, dl, t))
    assert st.time_s == pytest.approx(expected, abs=1e-6)


def test_section_timing_driven_band_inflates_and_demotes():
    # SPEC: the driven-band net inflates σ (never rejects) when the odometer distance
    # between the posterior crossings is implausibly short for the section geometry.
    # A lap that drives far under (b−a)−band over a straight (band = 7m, κ=0) is
    # demoted below tier A but STILL emits a value.
    from lap_analyzer.trajectory import Trajectory, section_timing
    v = 40.0
    n = 250
    # s_hat spans 0..2500 (so both gates are covered) but the odometer dl is
    # compressed: the car "covers" the section on the ruler while its odometer
    # advances far less than the ruler distance → short driven distance.
    s_hat = np.linspace(0.0, 2500.0, n)
    dl = np.linspace(0.0, 2500.0, n).copy()
    dl[s_hat >= 500.0] -= 0.0  # baseline
    # compress odometer between 500 and 1000 on the ruler: only 400m driven for a
    # 500m section (dev −100m ≫ band 7m)
    seg = (s_hat >= 500.0) & (s_hat <= 1000.0)
    comp = np.interp(s_hat, [500.0, 1000.0], [0.0, 100.0])
    dl = dl - np.where(s_hat > 1000.0, 100.0, np.where(seg, comp, 0.0))
    t = np.linspace(0.0, s_hat[-1] / v, n)
    knots = np.arange(dl.min(), dl.max(), 10.0)
    traj = Trajectory(
        t=t, s_hat=s_hat, sigma_m=np.full(n, 1.0), delta_hat=s_hat - dl, dl=dl,
        v=np.full(n, v), evidence=np.ones(n, bool), status="ok", knots=knots,
        knot_sigma=np.full(len(knots), 1.0), e_lat=np.zeros(n), checks=[])
    st = section_timing(traj, 500.0, 1000.0, _straight_corridor(n=300))
    assert st.status == "ok"
    assert not np.isnan(st.time_s)              # value STILL emitted (never rejected)
    assert st.driven_m == pytest.approx(400.0, abs=5.0)
    assert st.rank_eligible is False            # demoted by the σ inflation
    band_checks = [c for c in st.checks if "driven_band" in c["name"]]
    assert band_checks and band_checks[0]["pass"] is False


def test_section_timing_rescale_invalid_emits_value_not_rankable(make_lap_samples):
    # SPEC: a rescale_invalid trajectory still yields a (prior-only) value with
    # status 'rescale_invalid' and rank_eligible False — nothing silently dropped.
    from lap_analyzer.gates import TrackFrame
    from lap_analyzer.trajectory import estimate_trajectory, section_timing
    corr = _straight_corridor(n=300)
    frame = TrackFrame.from_centerline(pd.DataFrame(
        {"lat": [45.0, 45.0], "long": [-122.0, -121.99]}))
    lap = make_lap_samples(n=200, dist_lap_m=np.linspace(0, 2500, 200),
                           track_dist_m=np.linspace(0, 2500, 200) + 700.0)
    traj = estimate_trajectory(lap, corr, frame)
    st = section_timing(traj, 500.0, 1500.0, corr)
    assert st.status == "rescale_invalid"
    assert not np.isnan(st.time_s)
    assert st.rank_eligible is False


def test_section_timing_gps_backbone_caps_at_B():
    # SPEC (reserved §9.3): a gps_backbone trajectory never reaches tier A even when
    # its σ would otherwise qualify.
    from lap_analyzer.trajectory import section_timing
    traj = _straight_trajectory(sigma0=0.5, v=40.0, length=2500.0)
    object.__setattr__(traj, "status", "gps_backbone")
    st = section_timing(traj, 500.0, 1000.0, _straight_corridor(n=300))
    assert st.tier in ("B", "C")
    assert st.rank_eligible is False


def test_section_timing_emits_all_three_net_checks():
    # SPEC R12: every row carries structured check records for the driven-band,
    # consistency, and speed-consistency nets.
    from lap_analyzer.trajectory import section_timing
    traj = _straight_trajectory()
    st = section_timing(traj, 500.0, 1000.0, _straight_corridor(n=300))
    names = {c["name"] for c in st.checks}
    assert any("driven_band" in x for x in names)
    assert any("consistency" in x for x in names)
    assert any("speed_consistency" in x for x in names)


# ===========================================================================
# Per-mode σ-calibration (Stage 2, item 6 / design R7) — the CI-enforced guard.
# The authoritative ≥500-draws/mode hard gate runs on the REAL corridor via
# scripts/gps_trust_calibration.py (output archived in the Stage-2 commit); here
# a self-contained SYNTHETIC corridor keeps CI honest without the corpus.
# ===========================================================================

def _load_calibration_module():
    import importlib
    import sys
    from pathlib import Path
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("gps_trust_calibration")


def test_per_mode_calibration_within_2sigma():
    # SPEC / design R7: on randomised synthetic laps PER failure mode, the emitted
    # 1σ honestly covers the section-time error — |err| < 2σ̂ in ≥95% of draws,
    # asserted SEPARATELY for each mode (a global scale can hide Mode-4
    # under-coverage behind Mode-3 over-coverage). Synthetic corridor → CI-safe.
    cal = _load_calibration_module()
    ctx = cal._Ctx.synthetic()
    res = cal.run_calibration(n_per_mode=40, seed=7, ctx=ctx)
    for mode in cal.MODES:
        r = res[mode]
        assert r["n"] >= 30, f"{mode}: too few valid draws ({r['n']})"
        assert r["coverage"] >= 0.95, (
            f"{mode}: only {r['coverage']:.2%} within 2σ (σ under-covers this mode)")


def test_calibration_line_offset_is_rankable_and_true():
    # design R4 discrimination (executable): a genuine in-corridor, odometer-consistent
    # tight line is recovered ACCURATELY (the estimator follows the real δ), not
    # shrunk toward the OBD backbone — the property that keeps T11's real line fast.
    cal = _load_calibration_module()
    ctx = cal._Ctx.synthetic()
    import numpy as np
    rng = np.random.default_rng(3)
    from lap_analyzer.trajectory import estimate_trajectory, section_timing
    errs = []
    for _ in range(30):
        lap, a, b, truth = cal.make_synth_lap("line_offset", rng, ctx)
        traj = estimate_trajectory(lap, ctx.corr, ctx.frame)
        st = section_timing(traj, a, b, ctx.corr)
        errs.append(abs(st.time_s - truth))
    assert np.median(errs) < 0.10   # follows the real line, not shrunk to OBD
