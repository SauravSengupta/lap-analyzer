"""Clean-room tests for lap_analyzer.analysis.

Source of truth: tests/SPEC.md (## analysis.*) + docs/ARCHITECTURE.md +
docs/PIPELINE.md. NO implementation under lap_analyzer/ was read while writing
these. Expectations are derived from documented behavior, not from observed
sample values.

Coverage:
  - section_bounds          (per-corner window math + box-containment invariant)
  - _first_crossing_t       (upward crossing + linear interp + edge cases)
  - find_gear_bands         (ratio clustering → ascending band centers)
  - _confirm_runs           (sub-dwell run suppression to NaN)
  - derive_gear             (THE key contract: gear index 0 = shortest/lowest gear)
  - span_time               (structural emit rule + OBD sanity gate;
                             physical-time accuracy is a KNOWN-LIMITATION → xfail)
  - section_range_bounds    (multi-corner window + ValueError on reversed ids)
  - classify_t8_section     (documented label/exclusion rules on synthetic windows)
  - top_decile_laps         (pace-decile == 0 set, integration)
  - lap_summary             (per-lap dict contract, integration)
  - lap_index               (one row per (session, lap), optional track join)

The gear-index orientation (derive_gear / find_gear_bands) is the headline
contract and gets the most attention below.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.analysis import (
    _confirm_runs,
    _first_crossing_t,
    classify_t8_section,
    derive_gear,
    find_gear_bands,
    section_bounds,
    section_range_bounds,
    span_time,
)
from lap_analyzer.config import tracks_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ridge_track_def():
    """Parse the real tracks/ridge.json dict (allowed: tracks/ is not under
    lap_analyzer/). section_bounds/section_range_bounds/classify_t8_section all
    accept this parsed dict per the spec."""
    return json.loads((tracks_dir() / "ridge.json").read_text(encoding="utf-8"))


def _toy_track_def():
    """A small hand-built track_def with non-contiguous corners and a known
    lap_length, per the spec's documented shape:
    {"corners": [{"id","start_m","end_m"}...], "lap_length_internal_m": ...}.

    Corners are deliberately spaced so the post-cap and next-start clamps are
    both exercisable independently.
    """
    return {
        "lap_length_internal_m": 1000.0,
        "corners": [
            {"id": "C1", "start_m": 100.0, "end_m": 200.0, "apex_m": 150.0},
            {"id": "C2", "start_m": 400.0, "end_m": 450.0, "apex_m": 425.0},
            {"id": "C3", "start_m": 800.0, "end_m": 900.0, "apex_m": 850.0},
        ],
    }


# ---------------------------------------------------------------------------
# section_bounds
# ---------------------------------------------------------------------------

def test_section_bounds_returns_window_per_corner():
    # SPEC: analysis.section_bounds — dict[corner_id] -> (start, end), one per corner
    td = _toy_track_def()
    bounds = section_bounds(td)
    assert set(bounds) == {"C1", "C2", "C3"}
    for win in bounds.values():
        assert isinstance(win, tuple) and len(win) == 2


def test_section_bounds_start_is_pre_m_before_corner_clamped_at_zero():
    # SPEC: analysis.section_bounds — start = max(0, corner.start_m - pre_m)
    td = _toy_track_def()
    bounds = section_bounds(td, pre_m=50.0, post_cap_m=250.0)
    # C1.start_m=100 → 100-50 = 50.
    assert bounds["C1"][0] == pytest.approx(50.0, abs=0.01)
    # A corner whose start_m < pre_m clamps to 0.
    td2 = {"lap_length_internal_m": 1000.0,
           "corners": [{"id": "C0", "start_m": 30.0, "end_m": 120.0, "apex_m": 80.0},
                       {"id": "C1", "start_m": 400.0, "end_m": 450.0, "apex_m": 425.0}]}
    b2 = section_bounds(td2, pre_m=50.0)
    assert b2["C0"][0] == pytest.approx(0.0, abs=0.01)


def test_section_bounds_end_capped_by_post_cap_when_next_corner_far():
    # SPEC: analysis.section_bounds — end = min(next.start_m, corner.end_m + post_cap_m)
    # C1.end_m=200, +250 = 450; next (C2) starts at 400. So end clamps to 400.
    td = _toy_track_def()
    bounds = section_bounds(td, pre_m=50.0, post_cap_m=250.0)
    assert bounds["C1"][1] == pytest.approx(400.0, abs=0.01)
    # C2.end_m=450, +250 = 700; next (C3) starts at 800. post_cap wins → 700.
    assert bounds["C2"][1] == pytest.approx(700.0, abs=0.01)


def test_section_bounds_last_corner_next_start_is_lap_length():
    # SPEC: analysis.section_bounds — for the last corner, "next start" = lap_length_internal_m
    td = _toy_track_def()  # lap_length 1000, C3.end_m=900 (+250 = 1150) → clamp to 1000
    bounds = section_bounds(td, pre_m=50.0, post_cap_m=250.0)
    assert bounds["C3"][1] == pytest.approx(1000.0, abs=0.01)


def test_section_bounds_last_corner_fallback_when_no_lap_length():
    # SPEC: analysis.section_bounds — last corner fallback "next start" = last.end_m + 200
    td = {"corners": [{"id": "C1", "start_m": 100.0, "end_m": 200.0, "apex_m": 150.0}]}
    bounds = section_bounds(td, pre_m=50.0, post_cap_m=250.0)
    # No lap_length_internal_m → next start falls back to 200 + 200 = 400.
    # end = min(400, 200 + 250 = 450) = 400.
    assert bounds["C1"][1] == pytest.approx(400.0, abs=0.01)


def test_section_bounds_window_always_contains_corner_box():
    # SPEC: analysis.section_bounds — invariant: window contains the corner box; start>=0;
    #       end never spills past the next corner's start
    td = _ridge_track_def()
    bounds = section_bounds(td)
    by_id = {c["id"]: c for c in td["corners"]}
    starts_sorted = sorted((c["start_m"], c["id"]) for c in td["corners"])
    for i, (start_m, cid) in enumerate(starts_sorted):
        win_start, win_end = bounds[cid]
        corner = by_id[cid]
        assert win_start >= 0.0
        # box is contained
        assert win_start <= corner["start_m"] + 1e-6
        assert win_end >= corner["end_m"] - 1e-6
        # does not spill past the next corner's start
        if i + 1 < len(starts_sorted):
            next_start = starts_sorted[i + 1][0]
            assert win_end <= next_start + 1e-6


# ---------------------------------------------------------------------------
# _first_crossing_t
# ---------------------------------------------------------------------------

def test_first_crossing_linear_interp_midpoint():
    # SPEC: analysis._first_crossing_t — xs=[0,10], ts=[0,1], target=5 → 0.5
    xs = np.array([0.0, 10.0])
    ts = np.array([0.0, 1.0])
    assert _first_crossing_t(xs, ts, 5.0) == pytest.approx(0.5, abs=1e-9)


def test_first_crossing_no_upward_crossing_returns_none():
    # SPEC: analysis._first_crossing_t — no upward crossing → None
    # Strictly decreasing series never crosses 5 upward.
    xs = np.array([10.0, 8.0, 6.0, 4.0])
    ts = np.array([0.0, 1.0, 2.0, 3.0])
    assert _first_crossing_t(xs, ts, 5.0) is None


def test_first_crossing_is_upward_only():
    # SPEC: analysis._first_crossing_t — first UPWARD crossing (xs[i] < target <= xs[i+1])
    # The series dips below then rises above the target; the downward pass at the
    # start must be ignored; the upward pass is what's reported.
    xs = np.array([6.0, 2.0, 8.0])  # down through 5, then up through 5
    ts = np.array([0.0, 1.0, 2.0])
    t = _first_crossing_t(xs, ts, 5.0)
    # Upward crossing between t=1 (x=2) and t=2 (x=8): 2 + (5-2)/(8-2) = 1.5.
    assert t == pytest.approx(1.5, abs=1e-9)


def test_first_crossing_after_t_restricts_window():
    # SPEC: analysis._first_crossing_t — after_t restricts to crossings at/after that time
    # Two upward crossings of 5: near t=0.5 and near t=2.5.
    xs = np.array([0.0, 10.0, 0.0, 10.0])
    ts = np.array([0.0, 1.0, 2.0, 3.0])
    # No restriction → first crossing (0.5).
    assert _first_crossing_t(xs, ts, 5.0) == pytest.approx(0.5, abs=1e-9)
    # Restricted to >= 1.5 → the second crossing (2.5).
    assert _first_crossing_t(xs, ts, 5.0, after_t=1.5) == pytest.approx(2.5, abs=1e-9)


def test_first_crossing_zero_width_step_returns_ts_i_no_div_zero():
    # SPEC: analysis._first_crossing_t — exact crossing / zero-width step
    #       (xs[i+1] == xs[i]) → returns ts[i] (no division by zero)
    # target lands exactly on the flat step value at the upward transition.
    xs = np.array([0.0, 5.0, 5.0])
    ts = np.array([0.0, 1.0, 2.0])
    t = _first_crossing_t(xs, ts, 5.0)
    assert t is not None
    assert np.isfinite(t)
    # The first sample reaching the target is index 1 (ts=1.0); interpolation of
    # the prior 0->5 rise gives 1.0 and the zero-width step does not blow up.
    assert t == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# find_gear_bands
# ---------------------------------------------------------------------------

def test_find_gear_bands_empty_input_returns_empty():
    # SPEC: analysis.find_gear_bands — empty input → empty array
    out = find_gear_bands(np.array([]))
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_find_gear_bands_all_out_of_range_returns_empty():
    # SPEC: analysis.find_gear_bands — none in [lo, hi] → empty array
    # All ratios well below ratio_lo (default 30) and above ratio_hi (default 260).
    out = find_gear_bands(np.array([5.0, 6.0, 5.5, 1000.0, 1100.0]),
                          ratio_lo=30, ratio_hi=260)
    assert out.size == 0


def test_find_gear_bands_three_clusters_three_ascending_centers():
    # SPEC: analysis.find_gear_bands — 3 well-separated tight clusters → ~3 ascending
    #       centers, each near its cluster mean
    rng = np.random.default_rng(0)
    centers_true = [60.0, 120.0, 200.0]  # well inside [30, 260], > min_separation apart
    parts = [rng.normal(c, 1.0, 400) for c in centers_true]
    ratios = np.concatenate(parts)
    bands = find_gear_bands(ratios)
    # Sorted ascending.
    assert list(bands) == sorted(bands)
    # Approximately three centers recovered (allow the clustering some latitude).
    assert 2 <= bands.size <= 4
    # Each true center has a recovered band near it.
    for c in centers_true:
        assert np.min(np.abs(bands - c)) < 5.0


def test_find_gear_bands_centers_sorted_and_in_range():
    # SPEC: analysis.find_gear_bands — returns sorted-ascending centers within [lo, hi]
    rng = np.random.default_rng(1)
    ratios = np.concatenate([rng.normal(50, 1.0, 300), rng.normal(150, 1.0, 300)])
    bands = find_gear_bands(ratios, ratio_lo=30, ratio_hi=260)
    assert list(bands) == sorted(bands)
    assert np.all(bands >= 30) and np.all(bands <= 260)


# ---------------------------------------------------------------------------
# _confirm_runs
# ---------------------------------------------------------------------------

def test_confirm_runs_short_run_becomes_nan():
    # SPEC: analysis._confirm_runs — [1,1,1,2,1,1,1] n_dwell=2 → the single 2 → NaN
    raw = np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0])
    out = _confirm_runs(raw, n_dwell=2)
    assert np.isnan(out[3])  # the lone 2 is sub-dwell
    # The 1-runs (length 3 each, >= n_dwell) survive.
    assert out[0] == 1.0 and out[2] == 1.0
    assert out[4] == 1.0 and out[6] == 1.0


def test_confirm_runs_preserves_long_runs():
    # SPEC: analysis._confirm_runs — non-NaN runs >= n_dwell are preserved
    raw = np.array([3.0, 3.0, 3.0, 3.0])
    out = _confirm_runs(raw, n_dwell=2)
    assert np.array_equal(out, raw)


def test_confirm_runs_nan_runs_never_modified():
    # SPEC: analysis._confirm_runs — NaN runs are never modified
    raw = np.array([np.nan, np.nan, 5.0, 5.0, np.nan])
    out = _confirm_runs(raw, n_dwell=2)
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[4])
    # The length-2 run of 5 (== n_dwell) survives.
    assert out[2] == 5.0 and out[3] == 5.0


def test_confirm_runs_all_short_runs_all_nan():
    # SPEC: analysis._confirm_runs — every run shorter than n_dwell → all NaN
    raw = np.array([1.0, 2.0, 3.0, 4.0])  # each run length 1
    out = _confirm_runs(raw, n_dwell=2)
    assert np.all(np.isnan(out))


# ---------------------------------------------------------------------------
# derive_gear — THE key contract: gear index orientation
# ---------------------------------------------------------------------------

def _gear_lap(make_lap_samples, ratio, n=200):
    """A lap whose rpm/speed_mph ratio is a constant `ratio` everywhere, with
    rpm and speed both well above the qualifying thresholds (rpm>1200, speed>8).

    speed is held at a fixed value; rpm = ratio * speed so rpm/speed == ratio.
    """
    speed = 60.0
    rpm = ratio * speed
    assert rpm > 1200 and speed > 8  # qualifies
    return make_lap_samples(n=n, speed_mph=speed, rpm=int(round(rpm)))


def test_derive_gear_highest_ratio_is_index_zero(make_lap_samples):
    # SPEC: analysis.derive_gear — gear index 0 = shortest/lowest gear (HIGHEST rpm/speed ratio)
    # Three ascending bands; a lap whose ratio sits in the HIGHEST band → index 0.
    bands = np.array([40.0, 80.0, 120.0])  # ascending: [tallest ... shortest]
    lap = _gear_lap(make_lap_samples, ratio=120.0)  # clearly in the highest band
    gear = derive_gear(lap, bands)
    held = gear.dropna()
    assert not held.empty
    assert int(held.mode().iloc[0]) == 0  # highest ratio → index 0 (shortest gear)


def test_derive_gear_lowest_ratio_is_last_index(make_lap_samples):
    # SPEC: analysis.derive_gear — largest index = tallest gear (LOWEST rpm/speed ratio)
    bands = np.array([40.0, 80.0, 120.0])
    lap = _gear_lap(make_lap_samples, ratio=40.0)  # clearly in the lowest band
    gear = derive_gear(lap, bands)
    held = gear.dropna()
    assert not held.empty
    assert int(held.mode().iloc[0]) == len(bands) - 1  # lowest ratio → tallest gear


def test_derive_gear_middle_ratio_is_middle_index(make_lap_samples):
    # SPEC: analysis.derive_gear — index orientation: middle ratio → middle index
    bands = np.array([40.0, 80.0, 120.0])
    lap = _gear_lap(make_lap_samples, ratio=80.0)
    gear = derive_gear(lap, bands)
    held = gear.dropna()
    assert not held.empty
    # 3 bands, ratio-ascending. argmin|ratio-bands| = band index 1; orientation
    # flips it: (3-1) - 1 = 1. The middle gear stays the middle index.
    assert int(held.mode().iloc[0]) == 1


def test_derive_gear_rpm_zero_is_nan(make_lap_samples):
    # SPEC: analysis.derive_gear — samples with rpm == 0 → NaN gear
    bands = np.array([40.0, 80.0, 120.0])
    n = 200
    rpm = np.full(n, int(120.0 * 60.0))
    rpm[50:60] = 0  # a stretch of rpm==0
    lap = make_lap_samples(n=n, speed_mph=60.0, rpm=rpm)
    gear = derive_gear(lap, bands)
    assert gear.iloc[50:60].isna().all()


def test_derive_gear_aligned_to_samples_index(make_lap_samples):
    # SPEC: analysis.derive_gear — the returned Series is aligned to samples.index
    bands = np.array([40.0, 80.0, 120.0])
    lap = _gear_lap(make_lap_samples, ratio=80.0, n=120)
    # Use a non-default index to confirm alignment isn't positional-by-accident.
    lap.index = np.arange(1000, 1000 + len(lap))
    gear = derive_gear(lap, bands)
    assert list(gear.index) == list(lap.index)


def test_derive_gear_nonqualifying_samples_not_freshly_assigned(make_lap_samples):
    # SPEC: analysis.derive_gear — samples that don't qualify (rpm<=1200 or speed<=8)
    #       get no fresh assignment (NaN unless filled from a held gear)
    bands = np.array([40.0, 80.0, 120.0])
    n = 200
    # Whole lap fails the speed gate (speed <= 8) → no qualifying sample at all,
    # so no gear can be assigned or held forward.
    lap = make_lap_samples(n=n, speed_mph=5.0, rpm=4000)
    gear = derive_gear(lap, bands)
    assert gear.isna().all()


# ---------------------------------------------------------------------------
# span_time
# ---------------------------------------------------------------------------

def _clean_span_lap(make_lap_samples, n=200, total_m=2000.0, total_s=None):
    """A lap where track_dist_m and dist_lap_m advance together linearly with t,
    so the OBD sanity gate passes and crossings are unambiguous."""
    if total_s is None:
        total_s = (n - 1) * 0.1
    t = np.linspace(0.0, total_s, n)
    dist = np.linspace(0.0, total_m, n)
    return make_lap_samples(n=n, t=t, track_dist_m=dist, dist_lap_m=dist)


def test_span_time_uncrossed_bound_returns_none(make_lap_samples):
    # SPEC: analysis.span_time — None if either bound isn't crossed
    lap = _clean_span_lap(make_lap_samples, total_m=2000.0)
    # dist_b beyond the lap's max track_dist_m → never crossed.
    assert span_time(lap, 500.0, 9999.0) is None


def test_span_time_obd_gate_rejects_mismatched_distance(make_lap_samples):
    # SPEC: analysis.span_time — returns None when
    #       |obd_integrated_distance - (dist_b - dist_a)| > tol
    # Make track_dist_m advance far while dist_lap_m (OBD distance) barely moves,
    # so the OBD-distance sanity gate trips.
    n = 200
    t = np.linspace(0.0, 19.9, n)
    track = np.linspace(0.0, 2000.0, n)   # GPS says we covered 2000 m
    dist_lap = np.linspace(0.0, 50.0, n)  # OBD says we covered only 50 m
    lap = make_lap_samples(n=n, t=t, track_dist_m=track, dist_lap_m=dist_lap)
    # Span of ~1000 m by track_dist; OBD integrated distance over that span is
    # nowhere near 1000 m → gate returns None.
    assert span_time(lap, 500.0, 1500.0) is None


def test_span_time_clean_lap_returns_positive_elapsed(make_lap_samples):
    # SPEC: analysis.span_time — a clean lap where track_dist_m and dist_lap_m
    #       advance together returns the true elapsed time
    lap = _clean_span_lap(make_lap_samples, n=201, total_m=2000.0, total_s=20.0)
    # 2000 m over 20 s = 100 m/s. From 500 m to 1500 m is 1000 m → 10 s.
    out = span_time(lap, 500.0, 1500.0)
    assert out is not None
    assert out == pytest.approx(10.0, abs=0.2)


def test_span_time_explicit_tol_is_honored(make_lap_samples):
    # SPEC: analysis.span_time — obd_tol_m gates the OBD-distance sanity check
    # With a generous explicit tolerance, a clean lap still returns a value.
    lap = _clean_span_lap(make_lap_samples, n=201, total_m=2000.0, total_s=20.0)
    out = span_time(lap, 500.0, 1500.0, obd_tol_m=100.0)
    assert out == pytest.approx(10.0, abs=0.2)


@pytest.mark.xfail(reason="KNOWN-LIMITATION: section/span times are GPS-compressed; "
                          "physical-time accuracy of span_time on real laps is not "
                          "guaranteed (see SPEC time_in_corner_s / section_times note).",
                   strict=False)
def test_span_time_physical_accuracy_on_compressed_lap(make_lap_samples):
    # SPEC: KNOWN-LIMITATION — span_time physical-time accuracy under GPS compression.
    # On a GPS-compressed lap, track_dist_m crossings do not correspond to the true
    # physical span, so the elapsed time would be wrong. We assert the IDEAL here
    # only as an xfail to document the limitation — never as a passing guarantee.
    n = 200
    t = np.linspace(0.0, 20.0, n)
    # GPS distance is compressed (only spans 1000 m) vs the true 2000 m physical.
    track = np.linspace(0.0, 1000.0, n)
    dist_lap = np.linspace(0.0, 1000.0, n)
    lap = make_lap_samples(n=n, t=t, track_dist_m=track, dist_lap_m=dist_lap)
    out = span_time(lap, 250.0, 750.0)
    # The "true" physical time for the corresponding real-world span is NOT what a
    # naive crossing-based span_time recovers — this assertion is expected to fail.
    assert out == pytest.approx(20.0, abs=0.5)


# ---------------------------------------------------------------------------
# section_range_bounds
# ---------------------------------------------------------------------------

def test_section_range_bounds_single_corner_equals_section_bounds():
    # SPEC: analysis.section_range_bounds — from_id == to_id returns exactly
    #       section_bounds(track_def)[from_id]
    td = _toy_track_def()
    for cid in ("C1", "C2", "C3"):
        rng = section_range_bounds(td, cid, cid)
        sb = section_bounds(td)[cid]
        assert rng == pytest.approx(sb, abs=0.01)


def test_section_range_bounds_spans_from_to_inclusive():
    # SPEC: analysis.section_range_bounds — a = max(0, from.start_m - pre_m);
    #       b = min(start of corner after to_id, to.end_m + post_cap_m)
    td = _toy_track_def()
    a, b = section_range_bounds(td, "C1", "C2", pre_m=50.0, post_cap_m=250.0)
    # a from C1: 100 - 50 = 50.
    assert a == pytest.approx(50.0, abs=0.01)
    # b for to=C2: corner after C2 starts at 800; C2.end_m + 250 = 700. min → 700.
    assert b == pytest.approx(700.0, abs=0.01)


def test_section_range_bounds_to_is_last_corner_uses_lap_length():
    # SPEC: analysis.section_range_bounds — when to_id is the last corner, the
    #       "next start" falls back to lap_length (mirrors section_bounds)
    td = _toy_track_def()  # lap_length 1000
    a, b = section_range_bounds(td, "C2", "C3", pre_m=50.0, post_cap_m=250.0)
    assert a == pytest.approx(350.0, abs=0.01)  # C2.start_m 400 - 50
    # C3.end_m 900 + 250 = 1150; next-start fallback is lap_length 1000 → 1000.
    assert b == pytest.approx(1000.0, abs=0.01)


def test_section_range_bounds_reversed_order_raises():
    # SPEC: analysis.section_range_bounds — raises ValueError when to_id precedes
    #       from_id in track order
    td = _toy_track_def()
    with pytest.raises(ValueError):
        section_range_bounds(td, "C3", "C1")


def test_section_range_bounds_window_contains_both_corner_boxes():
    # SPEC: analysis.section_range_bounds — window spans from_id..to_id inclusive
    td = _ridge_track_def()
    a, b = section_range_bounds(td, "T1", "T3")
    by_id = {c["id"]: c for c in td["corners"]}
    assert a <= by_id["T1"]["start_m"] + 1e-6
    assert b >= by_id["T3"]["end_m"] - 1e-6
    assert a >= 0.0


# ---------------------------------------------------------------------------
# classify_t8_section — documented label/exclusion rules on synthetic windows
#
# KNOWN-LIMITATION (memory: T8-T11 traffic contamination): we test ONLY the
# documented rule on synthetic windows; we do NOT assert the classifier separates
# traffic from clean downshifts on real data.
# ---------------------------------------------------------------------------

# A minimal track_def carrying T8 and T10 with an apex inside T8's box. Scans
# [T8.start - 50, T10.end].
_T8_TRACK_DEF = {
    "lap_length_internal_m": 3000.0,
    "corners": [
        {"id": "T8", "start_m": 1830.0, "end_m": 2020.0, "apex_m": 1876.0},
        {"id": "T9", "start_m": 2020.0, "end_m": 2297.0, "apex_m": 2168.0},
        {"id": "T10", "start_m": 2297.0, "end_m": 2470.0, "apex_m": 2425.0},
    ],
}

# Two clearly separated gear bands: ratio ~60 = taller gear (higher index),
# ratio ~120 = shorter gear (index 0). A downshift = a transition to a SHORTER
# gear (lower index), i.e. ratio jumps UP from ~60 to ~120.
_T8_BANDS = np.array([60.0, 120.0])


def _t8_window(make_lap_samples, n, ratio_profile, *, rpm_zero=False,
               td_start=1780.0, td_end=2470.0):
    """Build a samples window covering [T8.start-50, T10.end].

    ratio_profile: length-n array of rpm/speed ratios (drives the gear track).
    speed held at 60 mph (qualifies); rpm = ratio*speed. rpm_zero overrides rpm
    to 0 everywhere (the no-obd path).
    """
    td = np.linspace(td_start, td_end, n)
    speed = np.full(n, 60.0)
    if rpm_zero:
        rpm = np.zeros(n, dtype=int)
    else:
        rpm = np.round(np.asarray(ratio_profile) * speed).astype(int)
    return make_lap_samples(n=n, track_dist_m=td, speed_mph=speed, rpm=rpm)


def test_classify_t8_too_few_samples_excluded_ambiguous(make_lap_samples):
    # SPEC: classify_t8_section — < 10 samples in window → label="excluded",
    #       excluded_reason="ambiguous"
    lap = _t8_window(make_lap_samples, n=6, ratio_profile=np.full(6, 60.0))
    out = classify_t8_section(lap, _T8_TRACK_DEF, _T8_BANDS)
    assert out["label"] == "excluded"
    assert out["excluded_reason"] == "ambiguous"


def test_classify_t8_all_rpm_zero_excluded_no_obd(make_lap_samples):
    # SPEC: classify_t8_section — all rpm == 0 in window → excluded_reason="no-obd"
    lap = _t8_window(make_lap_samples, n=80, ratio_profile=np.full(80, 60.0),
                     rpm_zero=True)
    out = classify_t8_section(lap, _T8_TRACK_DEF, _T8_BANDS)
    assert out["excluded_reason"] == "no-obd"


def test_classify_t8_zero_downshifts_is_no_downshift(make_lap_samples):
    # SPEC: classify_t8_section — zero downshifts → label="no-downshift",
    #       excluded_reason=None
    # Constant ratio (one steady gear) the whole window → no gear change at all.
    n = 120
    lap = _t8_window(make_lap_samples, n=n, ratio_profile=np.full(n, 60.0))
    out = classify_t8_section(lap, _T8_TRACK_DEF, _T8_BANDS)
    assert out["label"] == "no-downshift"
    assert out["excluded_reason"] is None


def test_classify_t8_single_downshift_before_apex_is_downshift(make_lap_samples):
    # SPEC: classify_t8_section — exactly one downshift at loc <= T8.apex_m → label="downshift"
    # Start in the taller gear (ratio 60), downshift to the shorter gear (ratio 120)
    # at a track_dist BEFORE T8.apex_m (1876).
    n = 200
    td = np.linspace(1780.0, 2470.0, n)
    apex_m = 1876.0
    # Downshift index: first sample whose track_dist exceeds the apex is the
    # boundary; pick a shift location comfortably before the apex.
    shift_idx = int(np.searchsorted(td, apex_m - 40.0))
    ratio = np.where(np.arange(n) < shift_idx, 60.0, 120.0)
    # Hold each gear long enough to clear min_dwell (>= 0.3 s at 10 Hz = 3+ samples).
    lap = _t8_window(make_lap_samples, n=n, ratio_profile=ratio)
    out = classify_t8_section(lap, _T8_TRACK_DEF, _T8_BANDS)
    assert out["label"] == "downshift"
    assert out["excluded_reason"] is None
    # downshift_dist_m should land before the apex.
    if out["downshift_dist_m"] is not None:
        assert out["downshift_dist_m"] <= apex_m + 1e-6


def test_classify_t8_single_downshift_past_apex_is_traffic(make_lap_samples):
    # SPEC: classify_t8_section — exactly one downshift PAST T8.apex_m (still in
    #       window) → excluded_reason="traffic"
    n = 200
    td = np.linspace(1780.0, 2470.0, n)
    apex_m = 1876.0
    # Shift well AFTER the apex but still inside the window (< T10.end 2470).
    shift_idx = int(np.searchsorted(td, apex_m + 200.0))
    ratio = np.where(np.arange(n) < shift_idx, 60.0, 120.0)
    lap = _t8_window(make_lap_samples, n=n, ratio_profile=ratio)
    out = classify_t8_section(lap, _T8_TRACK_DEF, _T8_BANDS)
    assert out["excluded_reason"] == "traffic"


def test_classify_t8_multiple_downshifts_excluded(make_lap_samples):
    # SPEC: classify_t8_section — more than one downshift → excluded
    n = 240
    td = np.linspace(1780.0, 2470.0, n)
    # Two distinct downshifts: 60 -> 120, back up to 60, then 120 again.
    # (Each held segment is long enough to clear dwell.)
    q = n // 4
    ratio = np.empty(n)
    ratio[:q] = 60.0
    ratio[q:2 * q] = 120.0   # downshift 1
    ratio[2 * q:3 * q] = 60.0  # upshift (not a downshift)
    ratio[3 * q:] = 120.0    # downshift 2
    lap = _t8_window(make_lap_samples, n=n, ratio_profile=ratio)
    out = classify_t8_section(lap, _T8_TRACK_DEF, _T8_BANDS)
    assert out["label"] == "excluded"


def test_classify_t8_returns_documented_keys(make_lap_samples):
    # SPEC: classify_t8_section — returns {label, excluded_reason, downshift_dist_m, t8_apex_gear}
    n = 120
    lap = _t8_window(make_lap_samples, n=n, ratio_profile=np.full(n, 60.0))
    out = classify_t8_section(lap, _T8_TRACK_DEF, _T8_BANDS)
    assert set(out.keys()) >= {"label", "excluded_reason", "downshift_dist_m",
                               "t8_apex_gear"}


# ---------------------------------------------------------------------------
# top_decile_laps / lap_summary / lap_index — integration tier (corpus-backed)
# ---------------------------------------------------------------------------

@pytest.fixture
def ridge_corpus(sample_data_root):
    """Load the committed Ridge corpus via the analysis library's loader.

    The spec says to "load_corpus(track) first". We import it lazily so the
    module's other tests don't depend on this symbol's exact name resolving at
    import time; if the loader is named differently this fixture will surface a
    clear ImportError rather than silently skip.
    """
    from lap_analyzer.analysis import load_corpus
    return load_corpus("ridge")


def test_top_decile_laps_returns_set_of_session_lap_pairs(ridge_corpus):
    # SPEC: analysis.top_decile_laps — set of (session_id, int lap) whose rows have
    #       lap_pace_decile == 0
    from lap_analyzer.analysis import top_decile_laps
    out = top_decile_laps(ridge_corpus)
    assert isinstance(out, set)
    # Each element is a (session_id, lap) pair; lap is an int.
    for elem in out:
        assert len(elem) == 2
        _sid, lap = elem
        assert isinstance(lap, (int, np.integer))


def test_top_decile_laps_are_exactly_the_decile_zero_rows(ridge_corpus):
    # SPEC: analysis.top_decile_laps — EXACTLY the (session, lap) pairs with decile 0
    from lap_analyzer.analysis import top_decile_laps
    corpus = ridge_corpus
    if "lap_pace_decile" not in corpus.columns:
        pytest.skip("corpus has no lap_pace_decile column")
    out = top_decile_laps(corpus)
    d0 = corpus[corpus["lap_pace_decile"] == 0]
    expected = {(s, int(l)) for s, l in zip(d0["session_id"], d0["lap"])}
    assert out == expected


def test_lap_summary_missing_lap_returns_empty_dict(ridge_corpus):
    # SPEC: analysis.lap_summary — empty {} when that lap isn't in the corpus
    from lap_analyzer.analysis import lap_summary
    out = lap_summary(ridge_corpus, "nonexistent-session", 99999)
    assert out == {}


def test_lap_summary_present_lap_has_documented_keys(ridge_corpus):
    # SPEC: analysis.lap_summary — includes n_transits, n_reliable_transits,
    #       lap_reliable, max_speed_mph for a present lap
    from lap_analyzer.analysis import lap_summary
    corpus = ridge_corpus
    # Pick any real (session, lap) present in the corpus.
    first = corpus.iloc[0]
    sid, lap = first["session_id"], int(first["lap"])
    out = lap_summary(corpus, sid, lap)
    assert out  # non-empty
    for key in ("n_transits", "n_reliable_transits", "lap_reliable", "max_speed_mph"):
        assert key in out
    # n_transits is a positive count.
    assert out["n_transits"] >= 1
    # reliable transits never exceed total transits.
    assert out["n_reliable_transits"] <= out["n_transits"]


def test_lap_index_one_row_per_session_lap(ridge_corpus):
    # SPEC: analysis.lap_index — one row per (session_id, lap)
    from lap_analyzer.analysis import lap_index
    idx = lap_index(ridge_corpus)
    assert isinstance(idx, pd.DataFrame)
    assert "session_id" in idx.columns and "lap" in idx.columns
    dupes = idx.duplicated(subset=["session_id", "lap"]).sum()
    assert dupes == 0
    # Row count equals the number of distinct (session, lap) pairs in the corpus.
    distinct = ridge_corpus.drop_duplicates(subset=["session_id", "lap"]).shape[0]
    assert len(idx) == distinct


def test_lap_index_with_track_joins_lap_time_and_clean(sample_data_root, ridge_corpus):
    # SPEC: analysis.lap_index — when track is given, joins lap_time_s/is_clean
    #       from each session's laps.csv
    from lap_analyzer.analysis import lap_index
    idx = lap_index(ridge_corpus, track="ridge")
    assert "lap_time_s" in idx.columns
    assert "is_clean" in idx.columns
