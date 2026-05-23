"""Clean-room tests for lap_analyzer.corners.

Source of truth: tests/SPEC.md (## corners.*) + docs/ARCHITECTURE.md (§2.5
corner-candidate scan, positive=right convention) + docs/PIPELINE.md. NO
implementation under lap_analyzer/ was read while writing these. Expectations are
derived from documented behavior, not from observed sample values.

Coverage:
  - extract_lap_candidates       (lat-G peak detection → candidate dicts;
                                  <50-sample short-circuit; direction convention;
                                  per-candidate offsets relative to entry_dist_m)
  - extract_session_candidates   (warmup/cooldown skip; empty-frame columns ==
                                  CANDIDATE_COLUMNS — integration tier)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.corners import extract_lap_candidates, extract_session_candidates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lap_with_latg_hump(make_lap_samples, *, n=120, peak_g=0.8, sign=+1.0,
                        peak_frac=0.5, width_frac=0.08):
    """A one-lap frame with a single clean Gaussian |lat_g| hump.

    sign=+1 → right turn (positive lat_g per the convention); sign=-1 → left.
    peak placed at peak_frac of the lap; width set so the peak is isolated.
    dist_lap_m advances linearly so entry/peak/exit distances are well-defined,
    sampled at 10 Hz (t spacing 0.1 s → peak spacing of 3 s = 30 samples).
    """
    i = np.arange(n)
    center = peak_frac * (n - 1)
    width = width_frac * n
    hump = peak_g * np.exp(-0.5 * ((i - center) / width) ** 2)
    lat_g = sign * hump
    return make_lap_samples(
        n=n,
        t=np.arange(n) * 0.1,
        lat_g=lat_g,
        long_g=np.zeros(n),
        speed_mph=np.full(n, 70.0),
        throttle_norm=np.full(n, 1.0),
        dist_lap_m=np.linspace(0.0, 2000.0, n),
        lat=np.full(n, 45.0),
        long=np.full(n, -122.0),
    )


# ---------------------------------------------------------------------------
# extract_lap_candidates
# ---------------------------------------------------------------------------

def test_extract_lap_candidates_too_few_samples_returns_empty(make_lap_samples):
    # SPEC: corners.extract_lap_candidates — < 50 samples → []
    lap = _lap_with_latg_hump(make_lap_samples, n=40)
    out = extract_lap_candidates(lap, sample_rate_hz=10.0)
    assert out == []


def test_extract_lap_candidates_returns_list_of_dicts(make_lap_samples):
    # SPEC: corners.extract_lap_candidates — emits one candidate dict per |lat_g| peak
    lap = _lap_with_latg_hump(make_lap_samples, n=120, peak_g=0.8)
    out = extract_lap_candidates(lap, sample_rate_hz=10.0)
    assert isinstance(out, list)
    assert all(isinstance(c, dict) for c in out)


def test_extract_lap_candidates_single_right_hump_one_right_candidate(make_lap_samples):
    # SPEC: corners.extract_lap_candidates — direction == "right" when the SMOOTHED
    #       peak lat_g > 0 (positive=right convention); one clear hump → one candidate
    lap = _lap_with_latg_hump(make_lap_samples, n=120, peak_g=0.8, sign=+1.0)
    out = extract_lap_candidates(lap, sample_rate_hz=10.0)
    # Exactly one peak above the 0.4 g height threshold on this lap.
    assert len(out) == 1
    assert out[0]["direction"] == "right"


def test_extract_lap_candidates_left_hump_is_left(make_lap_samples):
    # SPEC: corners.extract_lap_candidates — smoothed peak lat_g <= 0 → "left"
    lap = _lap_with_latg_hump(make_lap_samples, n=120, peak_g=0.8, sign=-1.0)
    out = extract_lap_candidates(lap, sample_rate_hz=10.0)
    assert len(out) == 1
    assert out[0]["direction"] == "left"


def test_extract_lap_candidates_below_height_threshold_no_candidate(make_lap_samples):
    # SPEC: corners.extract_lap_candidates — peaks require height >= 0.4 g
    # A hump that never reaches 0.4 g should yield no candidate.
    lap = _lap_with_latg_hump(make_lap_samples, n=120, peak_g=0.25, sign=+1.0)
    out = extract_lap_candidates(lap, sample_rate_hz=10.0)
    assert out == []


def test_extract_lap_candidates_min_spacing_3s_merges_close_peaks(make_lap_samples):
    # SPEC: corners.extract_lap_candidates — min peak spacing 3 s
    # Two |lat_g| humps spaced only ~1 s apart should not both survive as separate
    # candidates (peak spacing < 3 s). At 10 Hz, 1 s = 10 samples apart.
    n = 120
    i = np.arange(n)
    width = 0.03 * n
    h1 = 0.8 * np.exp(-0.5 * ((i - 50) / width) ** 2)
    h2 = 0.8 * np.exp(-0.5 * ((i - 60) / width) ** 2)  # only 10 samples (~1 s) later
    lap = make_lap_samples(
        n=n, t=np.arange(n) * 0.1, lat_g=h1 + h2, long_g=np.zeros(n),
        speed_mph=np.full(n, 70.0), throttle_norm=np.full(n, 1.0),
        dist_lap_m=np.linspace(0.0, 2000.0, n),
        lat=np.full(n, 45.0), long=np.full(n, -122.0),
    )
    out = extract_lap_candidates(lap, sample_rate_hz=10.0)
    # The 3 s min-spacing rule collapses the pair to a single candidate.
    assert len(out) == 1


def test_extract_lap_candidates_has_distance_fields(make_lap_samples):
    # SPEC: corners.extract_lap_candidates — candidate carries peak/entry/exit distances
    lap = _lap_with_latg_hump(make_lap_samples, n=120, peak_g=0.8)
    out = extract_lap_candidates(lap, sample_rate_hz=10.0)
    assert len(out) == 1
    cand = out[0]
    # Peak/entry/exit distances must be present; entry precedes peak precedes exit.
    assert "entry_dist_m" in cand
    assert "peak_dist_m" in cand
    assert "exit_dist_m" in cand
    assert cand["entry_dist_m"] <= cand["peak_dist_m"] <= cand["exit_dist_m"]


def test_extract_lap_candidates_offsets_relative_to_entry_dist(make_lap_samples):
    # SPEC: corners.extract_lap_candidates — offsets (brake_on_offset_m, ...) are
    #       relative to the candidate's entry_dist_m, or None when the event doesn't occur
    n = 120
    i = np.arange(n)
    width = 0.05 * n
    lat_g = 0.8 * np.exp(-0.5 * ((i - 60) / width) ** 2)  # right hump mid-lap
    # Brake engages in a known window approaching the hump.
    brake = np.zeros(n, dtype=int)
    brake[40:55] = 1
    long_g = np.zeros(n)
    long_g[40:55] = -0.6  # braking deceleration accompanies brake-on
    lap = make_lap_samples(
        n=n, t=np.arange(n) * 0.1, lat_g=lat_g, long_g=long_g, brake=brake,
        speed_mph=np.full(n, 70.0), throttle_norm=np.full(n, 1.0),
        dist_lap_m=np.linspace(0.0, 2000.0, n),
        lat=np.full(n, 45.0), long=np.full(n, -122.0),
    )
    out = extract_lap_candidates(lap, sample_rate_hz=10.0)
    assert len(out) == 1
    cand = out[0]
    assert "brake_on_offset_m" in cand
    if cand["brake_on_offset_m"] is not None:
        # Offset is the brake-on distance minus the candidate's entry distance.
        dist = np.linspace(0.0, 2000.0, n)
        brake_on_dist = dist[40]
        assert cand["brake_on_offset_m"] == pytest.approx(
            brake_on_dist - cand["entry_dist_m"], abs=20.0
        )


def test_extract_lap_candidates_absent_event_offset_is_none(make_lap_samples):
    # SPEC: corners.extract_lap_candidates — offset is None when the event doesn't occur
    # No braking anywhere on this lap → brake_on_offset_m should be None.
    lap = _lap_with_latg_hump(make_lap_samples, n=120, peak_g=0.8)
    # _lap_with_latg_hump leaves brake at the make_lap_samples default (0).
    out = extract_lap_candidates(lap, sample_rate_hz=10.0)
    assert len(out) == 1
    assert out[0]["brake_on_offset_m"] is None


# ---------------------------------------------------------------------------
# extract_session_candidates — integration tier
# ---------------------------------------------------------------------------

def test_extract_session_candidates_returns_dataframe(sample_data_root):
    # SPEC: corners.extract_session_candidates — (session_dir) -> DataFrame
    session_dir = sample_data_root / "sessions" / "ridge" / "20260517-100304"
    out = extract_session_candidates(session_dir)
    assert isinstance(out, pd.DataFrame)


def test_extract_session_candidates_columns_are_candidate_columns(sample_data_root):
    # SPEC: corners.extract_session_candidates — output columns are CANDIDATE_COLUMNS
    # (the spec pins exact column equality for the EMPTY case; for a non-empty
    # frame we assert the columns are a subset of the contract, no extras).
    from lap_analyzer.corners import CANDIDATE_COLUMNS
    session_dir = sample_data_root / "sessions" / "ridge" / "20260517-100304"
    out = extract_session_candidates(session_dir)
    if out.empty:
        # SPEC pins exact column equality for the empty case.
        assert list(out.columns) == list(CANDIDATE_COLUMNS)
    else:
        assert set(out.columns).issubset(set(CANDIDATE_COLUMNS))


def test_extract_session_candidates_skips_warmup_cooldown(sample_data_root):
    # SPEC: corners.extract_session_candidates — skips warmup/cooldown laps (uses clean_reason)
    session_dir = sample_data_root / "sessions" / "ridge" / "20260517-100304"
    laps = pd.read_csv(session_dir / "laps.csv")
    # Laps flagged warmup/cooldown (non-empty clean_reason) must not contribute
    # candidates.
    non_clean = set(
        laps[laps["clean_reason"].fillna("").astype(str) != ""]["lap"].astype(int)
    )
    out = extract_session_candidates(session_dir)
    if not out.empty and "lap" in out.columns and non_clean:
        emitted = set(out["lap"].astype(int).unique())
        assert emitted.isdisjoint(non_clean)
