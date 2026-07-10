"""Clean-room tests for lap_analyzer.fused_axis.

Source of truth: tests/SPEC.md (## fused_axis.compute_fused_dist). NO
implementation under lap_analyzer/ was read while writing these.

compute_fused_dist is a PURE function. The headline invariant is:
  OUTPUT IS MONOTONIC NON-DECREASING (cumulative max) even when track_dist_m
  jitters or steps backward.

Coverage:
  - < 2 rows -> a copy of dist_lap_m
  - output monotonic non-decreasing under backward-stepping track_dist_m
  - output monotonic non-decreasing under noisy/jittery track_dist_m
  - output aligned to input row order (rows unsorted by t)
  - a teleport (|track_dist_m - dist_lap_m| >= 50 m) does not drag the lowpass
  - on clean data (track_dist_m ~= dist_lap_m), fused ~= dist_lap_m
  - returns an ndarray of the right length
"""
from __future__ import annotations

import numpy as np
import pytest

from lap_analyzer.fused_axis import compute_fused_dist


def _is_monotonic_nondecreasing(arr) -> bool:
    a = np.asarray(arr, dtype=float)
    return bool(np.all(np.diff(a) >= -1e-9))


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

def test_single_row_returns_copy_of_dist_lap(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — < 2 rows -> a copy of dist_lap_m.
    lap = make_lap_samples(n=1, dist_lap_m=np.array([42.0]),
                           track_dist_m=np.array([999.0]))
    out = compute_fused_dist(lap)
    assert np.asarray(out).shape[0] == 1
    assert out[0] == pytest.approx(42.0, abs=0.01)


def test_zero_rows_handled(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — < 2 rows -> a copy of dist_lap_m
    #       (an empty frame is < 2 rows; output mirrors dist_lap_m -> empty).
    lap = make_lap_samples(n=2).iloc[0:0]
    out = compute_fused_dist(lap)
    assert np.asarray(out).shape[0] == 0


# ---------------------------------------------------------------------------
# THE headline invariant: monotonic non-decreasing output
# ---------------------------------------------------------------------------

def test_output_monotonic_under_backward_stepping_track_dist(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — output is monotonic non-decreasing by
    #       construction (cumulative max), even when track_dist_m steps BACKWARD.
    n = 200
    dist_lap = np.linspace(0.0, 2000.0, n)  # clean monotone OBD base
    # track_dist_m advances but with periodic backward steps (GPS folding).
    track = dist_lap.copy()
    track[50:60] -= 80.0   # a backward step
    track[120:130] -= 120.0
    track[170] -= 200.0
    lap = make_lap_samples(n=n, dist_lap_m=dist_lap, track_dist_m=track)
    out = compute_fused_dist(lap)
    assert _is_monotonic_nondecreasing(out)


def test_output_monotonic_under_jittery_track_dist(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — monotone even when track_dist_m jitters.
    n = 300
    rng = np.random.default_rng(7)
    dist_lap = np.linspace(0.0, 3000.0, n)
    track = dist_lap + rng.normal(0.0, 15.0, n)  # heavy GPS jitter
    lap = make_lap_samples(n=n, dist_lap_m=dist_lap, track_dist_m=track)
    out = compute_fused_dist(lap)
    assert _is_monotonic_nondecreasing(out)


def test_output_monotonic_with_smooth_window_one(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — monotonicity holds regardless of the
    #       smoothing window length (cumulative max is applied either way).
    n = 150
    dist_lap = np.linspace(0.0, 1500.0, n)
    track = dist_lap.copy()
    track[::5] -= 50.0  # frequent backward jitter
    lap = make_lap_samples(n=n, dist_lap_m=dist_lap, track_dist_m=track)
    out = compute_fused_dist(lap, smooth_window=1)
    assert _is_monotonic_nondecreasing(out)


# ---------------------------------------------------------------------------
# Row-order alignment
# ---------------------------------------------------------------------------

def test_output_aligned_to_input_row_order(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — output is aligned to the input row
    #       order (rows may be unsorted by t).
    n = 100
    dist_lap = np.linspace(0.0, 1000.0, n)
    track = dist_lap.copy()
    lap = make_lap_samples(n=n, dist_lap_m=dist_lap, track_dist_m=track)
    # Shuffle the rows; the function must return values aligned to THIS order.
    perm = np.random.default_rng(0).permutation(n)
    shuffled = lap.iloc[perm].reset_index(drop=True)
    out = compute_fused_dist(shuffled)
    assert np.asarray(out).shape[0] == n
    # The fused value for each row tracks that row's own dist_lap_m: on clean data
    # (track ~= dist_lap) the fused output equals dist_lap_m in the row's order.
    assert np.allclose(np.asarray(out), shuffled["dist_lap_m"].to_numpy(), atol=1e-6)


# ---------------------------------------------------------------------------
# Teleport robustness
# ---------------------------------------------------------------------------

def test_teleport_does_not_drag_lowpass(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — a sample whose |track_dist_m -
    #       dist_lap_m| >= 50 m (a teleport) does not drag the low-pass; its offset
    #       is interpolated over from good neighbors.
    n = 200
    dist_lap = np.linspace(0.0, 2000.0, n)
    track = dist_lap.copy()  # otherwise clean (offset ~0 everywhere)
    track[100] += 500.0  # a single 500 m teleport (>> 50 m)
    lap = make_lap_samples(n=n, dist_lap_m=dist_lap, track_dist_m=track)
    out = np.asarray(compute_fused_dist(lap))
    # Build the no-teleport reference for the same lap.
    lap_clean = make_lap_samples(n=n, dist_lap_m=dist_lap, track_dist_m=dist_lap)
    ref = np.asarray(compute_fused_dist(lap_clean))
    # The teleport must not pull the fused curve far off the clean reference; if it
    # dragged the lowpass it would shift many neighbors by tens of meters.
    assert np.max(np.abs(out - ref)) < 50.0
    assert _is_monotonic_nondecreasing(out)


# ---------------------------------------------------------------------------
# Clean-data identity
# ---------------------------------------------------------------------------

def test_clean_data_fused_approximates_dist_lap(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — on clean data where track_dist_m ~=
    #       dist_lap_m, fused ~= dist_lap_m.
    n = 250
    dist_lap = np.linspace(0.0, 2500.0, n)
    track = dist_lap.copy()  # exactly equal -> offset is identically 0
    lap = make_lap_samples(n=n, dist_lap_m=dist_lap, track_dist_m=track)
    out = np.asarray(compute_fused_dist(lap))
    assert np.allclose(out, dist_lap, atol=1.0)


def test_fused_base_tracks_dist_lap_with_constant_offset(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — fused = dist_lap_m + lowpass(track - dist_lap).
    #       With a CONSTANT offset, the lowpass equals that constant, so
    #       fused ~= dist_lap_m + constant.
    n = 250
    dist_lap = np.linspace(0.0, 2500.0, n)
    track = dist_lap + 12.0  # constant +12 m offset everywhere
    lap = make_lap_samples(n=n, dist_lap_m=dist_lap, track_dist_m=track)
    out = np.asarray(compute_fused_dist(lap))
    # Allow edge effects of the smoothing window; check the interior.
    interior = slice(40, n - 40)
    assert np.allclose(out[interior], (dist_lap + 12.0)[interior], atol=2.0)
    assert _is_monotonic_nondecreasing(out)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

def test_returns_ndarray_of_correct_length(make_lap_samples):
    # SPEC: fused_axis.compute_fused_dist — returns np.ndarray (Signature) sized to
    #       the input.
    n = 123
    lap = make_lap_samples(n=n)
    out = compute_fused_dist(lap)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == n


# ---------------------------------------------------------------------------
# glitch_runs tests
# ---------------------------------------------------------------------------

from lap_analyzer.fused_axis import GLITCH_OFFSET_M, glitch_runs


def test_glitch_runs_none_when_clean():
    # SPEC: fused_axis.glitch_runs — no run when GPS ~= OBD everywhere.
    td = np.arange(0.0, 100.0, 10.0)
    dl = td.copy()
    fused = td.copy()
    assert glitch_runs(td, dl, fused) == []


def test_glitch_runs_one_contiguous_span():
    # SPEC: fused_axis.glitch_runs — one span covering a contiguous glitched run,
    # reported in FUSED coordinates.
    td = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    dl = td.copy()
    td[2:4] = td[2:4] - 200.0            # samples 2,3 are teleported (offset 200 m >= 50)
    fused = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    runs = glitch_runs(td, dl, fused)
    assert len(runs) == 1
    assert runs[0] == pytest.approx((20.0, 30.0))   # fused span of samples 2..3


def test_glitch_runs_two_separate_spans():
    # SPEC: fused_axis.glitch_runs — separate runs yield separate spans.
    td = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    dl = td.copy()
    td[1] -= 100.0                        # run A: sample 1
    td[4:6] -= 100.0                      # run B: samples 4,5
    fused = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    runs = glitch_runs(td, dl, fused)
    assert len(runs) == 2
    assert runs[0] == pytest.approx((10.0, 10.0))
    assert runs[1] == pytest.approx((40.0, 50.0))


def test_glitch_runs_respects_threshold():
    # SPEC: fused_axis.glitch_runs — offsets below threshold are not flagged.
    td = np.array([0.0, 10.0, 20.0, 30.0])
    dl = td.copy()
    td[1] -= (GLITCH_OFFSET_M - 5.0)      # just under threshold
    fused = td.copy()
    assert glitch_runs(td, dl, fused) == []
