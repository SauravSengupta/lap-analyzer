"""Clean-room tests for lap_analyzer.labeler.

Source of truth: tests/SPEC.md (## labeler.*) + docs/ARCHITECTURE.md §2/§4 +
docs/PIPELINE.md. NO implementation under lap_analyzer/ was read while writing
these. Expectations are derived from documented behavior, not observed values.

Coverage:
  - load_track                 (dataclass parse + sort + anchor population)
  - ReferenceIndex.project     (local-meters projection arithmetic)
  - ReferenceIndex.lookup      (nearest-reference distance lookup, integration)
  - compute_lap_drift          (MEDIAN aggregation robustness — the headline)
  - label_samples              (corner segmentation + front-straight wrap)
  - build_corner_transit       (apex / latg-peak / input-timing offsets, signs)
  - build_session_corners      (clean-lap-only transit emission, integration)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.config import tracks_dir
from lap_analyzer.labeler import (
    Corner,
    build_corner_transit,
    build_reference_index,
    build_session_corners,
    compute_lap_drift,
    label_samples,
    load_track,
)

# Meters-per-degree constants the spec says build_reference_index computes.
M_PER_DEG_LAT = 111_132.0
RIDGE_CENTER_LAT = 47.255  # ~Ridge latitude; only used to derive m_per_deg_lon
M_PER_DEG_LON = 111_132.0 * math.cos(math.radians(RIDGE_CENTER_LAT))


def _call_label_samples(lap, track, ref_index):
    """Invoke label_samples tolerant of arg ORDER.

    The spec names the collaborators (a Track and a ReferenceIndex) but does not
    pin the parameter order. Rather than read the implementation, try the
    documented-plausible orderings and use whichever the function accepts.
    """
    for args in ((lap, track, ref_index), (lap, ref_index, track)):
        try:
            return label_samples(*args)
        except TypeError:
            continue
    # Last resort: let the natural call raise a clear error.
    return label_samples(lap, track, ref_index)


# ---------------------------------------------------------------------------
# load_track
# ---------------------------------------------------------------------------

def test_load_track_parses_core_fields():
    # SPEC: labeler.load_track — Track has track_id/name/direction/lap_length_m
    track = load_track("ridge")
    assert track.track_id == "ridge"
    assert isinstance(track.name, str) and track.name
    assert track.direction in ("clockwise", "counter-clockwise")
    # lap_length_m comes from lap_length_internal_m in the JSON.
    assert isinstance(track.lap_length_m, (int, float))
    assert track.lap_length_m > 0


def test_load_track_corners_sorted_ascending_by_start_m():
    # SPEC: labeler.load_track — corners are sorted ascending by start_m
    track = load_track("ridge")
    starts = [c.start_m for c in track.corners]
    assert starts == sorted(starts)
    assert len(starts) >= 2  # ridge has 16 corners; need >1 to make sorting meaningful


def test_load_track_anchors_populated_with_corner_bounds():
    # SPEC: labeler.load_track — each CalibrationAnchor gets its corner's start/end_m
    track = load_track("ridge")
    corner_ids = {c.id for c in track.corners}
    # Anchors whose corner_id has no matching corner are dropped.
    for anchor in track.calibration_anchors:
        assert anchor.corner_id in corner_ids
    # And the surviving anchors carry their corner's bounds.
    bounds_by_id = {c.id: (c.start_m, c.end_m) for c in track.corners}
    for anchor in track.calibration_anchors:
        start, end = bounds_by_id[anchor.corner_id]
        assert anchor.start_m == pytest.approx(start, abs=0.01)
        assert anchor.end_m == pytest.approx(end, abs=0.01)


def test_load_track_reference_lap_block_from_json():
    # SPEC: labeler.load_track — reference_session_id/reference_lap from JSON block
    track = load_track("ridge")
    # ridge.json has reference_lap pointing at _synthetic_centerline / lap 1.
    assert track.reference_session_id == "_synthetic_centerline"
    assert track.reference_lap == 1


def test_load_track_reads_from_tracks_dir_not_data_root(sample_data_root):
    # SPEC: load_track uses tracks/<id>.json (read via tracks_dir(), not data/samples)
    # Even with DATA_ROOT pointed at the sample bundle, load_track resolves the
    # repo-root tracks/ definition.
    assert (tracks_dir() / "ridge.json").exists()
    track = load_track("ridge")
    assert track.track_id == "ridge"


def test_load_track_both_tracks_loadable():
    # SPEC: labeler.load_track — works for every track JSON (ridge + pir)
    for track_id in ("ridge", "pir"):
        track = load_track(track_id)
        assert track.track_id == track_id
        assert track.corners  # non-empty corner list
        starts = [c.start_m for c in track.corners]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# ReferenceIndex.project — pure projection arithmetic
#
# The spec documents .project's FORMULA relative to "the index center" and that
# "the center maps to (0,0)" — but does NOT name the index's internal attributes.
# To stay clean-room (no implementation peeking), these tests assert .project's
# documented PROPERTIES through the public method alone:
#   * the projected difference between two points equals their meter delta, with
#     x scaled by m_per_deg_lon and y by m_per_deg_lat (translation-invariance);
#   * exactly one point projects to the origin (the center), so projecting that
#     same point twice is reproducible / deterministic.
# ---------------------------------------------------------------------------

@pytest.fixture
def ridge_reference_index(sample_data_root):
    """A real ReferenceIndex built from the committed _synthetic_centerline session."""
    track = load_track("ridge")
    return build_reference_index(track)


@pytest.fixture
def ridge_centerline_points(sample_data_root):
    """Reference (lat, long, dist_lap_m) points read from the committed centerline.

    Reading a file under data/samples/ is permitted; we use it only to obtain
    real on-track coordinates to drive .lookup — not to back out expected
    track_dist values (those follow from the spec's coincident-point rule).
    """
    df = pd.read_parquet(sample_data_root / "corpus" / "ridge_centerline.parquet")
    return df


def test_reference_index_project_shape(ridge_reference_index):
    # SPEC: ReferenceIndex.project(lat, lon) → (N, 2) array of local meters
    idx = ridge_reference_index
    out = idx.project(np.array([47.255, 47.256]), np.array([-123.19, -123.18]))
    assert out.shape == (2, 2)


def test_reference_index_project_translation_invariant_meters(ridge_reference_index):
    # SPEC: ReferenceIndex.project — x=(lon-center_lon)*m_per_deg_lon,
    #       y=(lat-center_lat)*m_per_deg_lat. The center cancels in a difference,
    #       so projected deltas equal the meter deltas (independent of internals).
    idx = ridge_reference_index
    lat0, lon0 = 47.2550, -123.1900
    dlat, dlon = 0.0010, 0.0010
    out = idx.project(np.array([lat0, lat0 + dlat]), np.array([lon0, lon0 + dlon]))
    dy = out[1, 1] - out[0, 1]
    dx = out[1, 0] - out[0, 0]
    # y axis scaled by 111_132 m/deg lat; x axis by 111_132*cos(lat) m/deg lon.
    assert dy == pytest.approx(dlat * M_PER_DEG_LAT, rel=1e-4)
    assert dx == pytest.approx(dlon * M_PER_DEG_LON, rel=1e-3)


def test_reference_index_project_origin_is_unique_center(ridge_reference_index):
    # SPEC: ReferenceIndex.project — the center maps to (0, 0)
    # We can't read the center coord (internal), but we can confirm exactly one
    # point projects to the origin: solve for it from two projected probes along
    # each axis, then verify projecting that solved center yields ~(0,0).
    idx = ridge_reference_index
    lat0, lon0 = 47.2550, -123.1900
    out = idx.project(np.array([lat0]), np.array([lon0]))
    y0, x0 = out[0, 1], out[0, 0]
    # Back out the center from the linear relation y0 = (lat0-clat)*m_lat.
    clat = lat0 - y0 / M_PER_DEG_LAT
    clon = lon0 - x0 / M_PER_DEG_LON
    center_out = idx.project(np.array([clat]), np.array([clon]))
    assert center_out[0, 0] == pytest.approx(0.0, abs=0.5)
    assert center_out[0, 1] == pytest.approx(0.0, abs=0.5)


# ---------------------------------------------------------------------------
# ReferenceIndex.lookup — nearest-reference lookup (integration tier)
# ---------------------------------------------------------------------------

def test_reference_index_lookup_returns_two_nonneg_distance(ridge_reference_index,
                                                            ridge_centerline_points):
    # SPEC: ReferenceIndex.lookup — returns (track_dist_m, dist_to_ref_m)
    idx = ridge_reference_index
    row = ridge_centerline_points.iloc[0]
    track_dist_m, dist_to_ref_m = idx.lookup(float(row["lat"]), float(row["long"]))
    assert dist_to_ref_m >= 0.0
    assert np.isfinite(track_dist_m)


def test_reference_index_lookup_coincident_point(ridge_reference_index,
                                                 ridge_centerline_points):
    # SPEC: ReferenceIndex.lookup — a point coincident with a reference sample has
    #       dist_to_ref_m ≈ 0 and returns that sample's dist_lap_m.
    idx = ridge_reference_index
    # The centerline IS the reference path. Pick a mid-track point on it; the
    # nearest reference sample is essentially itself, so dist_to_ref ≈ 0 and the
    # returned track_dist_m equals that point's centerline track_dist_m.
    cl = ridge_centerline_points
    mid = cl.iloc[len(cl) // 2]
    track_dist_m, dist_to_ref_m = idx.lookup(float(mid["lat"]), float(mid["long"]))
    assert dist_to_ref_m == pytest.approx(0.0, abs=3.0)
    assert track_dist_m == pytest.approx(float(mid["track_dist_m"]), abs=5.0)


# ---------------------------------------------------------------------------
# compute_lap_drift — the headline: MEDIAN robustness to one bad anchor
# ---------------------------------------------------------------------------

def _anchor(corner_id, lat, lon):
    """Build a calibration-anchor-shaped object for compute_lap_drift.

    SPEC (labeler.compute_lap_drift): each anchor carries (ref_lat, ref_long), and
    (labeler.load_track) each CalibrationAnchor is populated with its corner's
    start_m/end_m. We construct via the documented field names; the search-radius
    box is wide so every anchor is in range unless a test deliberately places the
    lap far away.
    """
    from lap_analyzer.labeler import CalibrationAnchor

    fields = dict(
        corner_id=corner_id,
        ref_lat=lat,
        ref_long=lon,
        start_m=0.0,
        end_m=10_000.0,
    )
    try:
        return CalibrationAnchor(**fields)
    except TypeError:
        # Tolerate an extra documented attribute (e.g. confidence) without
        # reading the implementation: retry with a benign default.
        return CalibrationAnchor(confidence="high", **fields)


def _lap_with_points(make_lap_samples, points):
    """A synthetic lap whose lat/long pass through the given (lat, lon) points.

    Pads to >=200 samples; the listed points are placed at the front so each is
    the closest sample to its corresponding anchor.
    """
    n = max(200, len(points) + 10)
    lat = np.full(n, points[-1][0])
    lon = np.full(n, points[-1][1])
    for i, (la, lo) in enumerate(points):
        lat[i] = la
        lon[i] = lo
    return make_lap_samples(n=n, lat=lat, long=lon)


def test_compute_lap_drift_median_ignores_one_bad_anchor(make_lap_samples):
    # SPEC: compute_lap_drift — MEDIAN is robust to a single wildly-off anchor
    # Construct anchors at known coords, and a lap whose nearest sample to each
    # anchor sits a KNOWN offset away. Three "good" anchors agree (+10 m east,
    # +0 m north); one "bad" anchor's sample is a +90 m east outlier. The returned
    # drift must equal the good cluster's median (10 m), NOT the mean (30 m, which
    # the outlier would drag off).
    clat = 47.255
    # ~1 deg lon at this latitude in meters:
    m_lon = M_PER_DEG_LON
    m_lat = M_PER_DEG_LAT

    def shift(lat, lon, dx_m, dy_m):
        return (lat + dy_m / m_lat, lon + dx_m / m_lon)

    anchors = []
    points = []
    base_lons = [-123.180, -123.185, -123.190]
    # Three good anchors: sample is +10 m east, +0 m north of each anchor coord.
    for i, blon in enumerate(base_lons):
        a_lat, a_lon = clat, blon
        anchors.append(_anchor(f"G{i}", a_lat, a_lon))
        points.append(shift(a_lat, a_lon, 10.0, 0.0))
    # One bad anchor whose nearest sample is a strong outlier. It must stay
    # inside the 100 m search radius to actually be USED (the spec only counts
    # samples within 100 m of the anchor); +90 m east is the most extreme outlier
    # that is still used. The median must absorb it regardless.
    bad_lat, bad_lon = clat + 0.01, -123.200
    anchors.append(_anchor("BAD", bad_lat, bad_lon))
    points.append(shift(bad_lat, bad_lon, 90.0, 0.0))  # +90 m east — outlier

    lap = _lap_with_points(make_lap_samples, points)
    drift_lat, drift_lon, n_used, disagreement = compute_lap_drift(
        lap, anchors, m_lat, m_lon
    )
    assert n_used == 4
    # Median east-offset of [10,10,10,90] = 10 m → drift_lon ~= +10 m.
    assert drift_lon == pytest.approx(10.0, abs=1.5)
    # Mean would be (10+10+10+90)/4 = 30 m. Confirm the result is NOT the mean.
    assert abs(drift_lon - 30.0) > 5.0
    # North offset is ~0 for all.
    assert drift_lat == pytest.approx(0.0, abs=1.5)
    assert disagreement >= 0.0


def test_compute_lap_drift_single_anchor_zero_disagreement(make_lap_samples):
    # SPEC: compute_lap_drift — n_used == 1 → disagreement == 0.0
    clat, clon = 47.255, -123.185
    a_lat, a_lon = clat, clon
    # Sample sits +12 m east of the anchor.
    sample_lat = a_lat
    sample_lon = a_lon + 12.0 / M_PER_DEG_LON
    lap = _lap_with_points(make_lap_samples, [(sample_lat, sample_lon)])
    anchors = [_anchor("A", a_lat, a_lon)]
    drift_lat, drift_lon, n_used, disagreement = compute_lap_drift(
        lap, anchors, M_PER_DEG_LAT, M_PER_DEG_LON
    )
    assert n_used == 1
    assert disagreement == pytest.approx(0.0, abs=1e-9)
    assert drift_lon == pytest.approx(12.0, abs=1.5)


def test_compute_lap_drift_no_anchor_within_radius(make_lap_samples):
    # SPEC: compute_lap_drift — no anchor within 100 m → (None, None, 0, None)
    # Lap sits far (>>100 m) from the only anchor.
    anchors = [_anchor("A", 47.255, -123.185)]
    far_lat = 47.300  # ~5 km north
    far_lon = -123.300
    lap = _lap_with_points(make_lap_samples, [(far_lat, far_lon)])
    drift_lat, drift_lon, n_used, disagreement = compute_lap_drift(
        lap, anchors, M_PER_DEG_LAT, M_PER_DEG_LON
    )
    assert (drift_lat, drift_lon, n_used, disagreement) == (None, None, 0, None)


def test_compute_lap_drift_disagreement_rms_of_deviations(make_lap_samples):
    # SPEC: compute_lap_drift — disagreement_m = sqrt(mean(dev² from the median)) ≥ 0
    # Two anchors with east-offsets +10 m and +20 m. Median = 15 m; deviations
    # are ±5 m; RMS = 5 m.
    clat = 47.255
    a0 = (clat, -123.180)
    a1 = (clat, -123.190)
    p0 = (a0[0], a0[1] + 10.0 / M_PER_DEG_LON)
    p1 = (a1[0], a1[1] + 20.0 / M_PER_DEG_LON)
    anchors = [_anchor("A0", *a0), _anchor("A1", *a1)]
    lap = _lap_with_points(make_lap_samples, [p0, p1])
    _, drift_lon, n_used, disagreement = compute_lap_drift(
        lap, anchors, M_PER_DEG_LAT, M_PER_DEG_LON
    )
    assert n_used == 2
    assert drift_lon == pytest.approx(15.0, abs=1.5)  # median of [10, 20]
    assert disagreement == pytest.approx(5.0, abs=1.5)
    assert disagreement >= 0.0


# ---------------------------------------------------------------------------
# build_corner_transit — apex / latg / input-timing offsets and signs
# ---------------------------------------------------------------------------

def _make_corner(corner_id="T1", start_m=100.0, end_m=300.0, apex_m=200.0,
                 secondary_apex_m=None, type="right"):
    """Construct a Corner via the dataclass.

    Corner is a plain dataclass with all 8 fields required (id, name, start_m,
    end_m, apex_m, secondary_apex_m, type, notes); name/notes are benign here.
    """
    return Corner(id=corner_id, name=None, start_m=start_m, end_m=end_m,
                  apex_m=apex_m, secondary_apex_m=secondary_apex_m,
                  type=type, notes=None)


def test_build_corner_transit_too_few_samples_returns_none(make_lap_samples):
    # SPEC: build_corner_transit — fewer than 3 samples inside [start_m,end_m] → None
    corner = _make_corner(start_m=100.0, end_m=300.0, apex_m=200.0)
    # Only 2 samples land inside [100, 300]: place track_dist far apart.
    td = np.array([0.0, 150.0, 250.0, 1000.0, 2000.0])
    lap = make_lap_samples(n=5, track_dist_m=td)
    assert build_corner_transit(lap, corner) is None


def test_build_corner_transit_apex_offset_signs(make_lap_samples):
    # SPEC: build_corner_transit — apex_dist_offset_m = round(min_speed_dist_m - apex_m, 2)
    #       and latg_peak_offset_m = round(latg_peak_dist_m - apex_m, 2)
    # Build a V-shaped speed profile with min at a KNOWN track_dist, and a lat_g
    # peak at a DIFFERENT known track_dist, so the two offsets resolve to known
    # signed values.
    n = 41
    td = np.linspace(100.0, 300.0, n)  # corner box exactly [100, 300]
    apex_m = 220.0
    # Speed: V-shape with minimum exactly at td == 180.0 (index 16).
    min_idx = 16
    assert td[min_idx] == pytest.approx(180.0, abs=0.01)
    speed = 100.0 - 40.0 * (1.0 - np.abs(np.arange(n) - min_idx) / n)
    speed[min_idx] = 30.0  # unambiguous global minimum
    # lat_g: a peak (max |lat_g|) at td == 260.0 (index 32).
    peak_idx = 32
    assert td[peak_idx] == pytest.approx(260.0, abs=0.01)
    lat_g = np.full(n, 0.1)
    lat_g[peak_idx] = 1.3  # unambiguous max magnitude

    corner = _make_corner(start_m=100.0, end_m=300.0, apex_m=apex_m)
    lap = make_lap_samples(n=n, track_dist_m=td, speed_mph=speed, lat_g=lat_g)
    out = build_corner_transit(lap, corner)
    assert out is not None
    # min speed at 180 → 180 - 220 = -40 (before the apex)
    assert out["apex_dist_offset_m"] == pytest.approx(-40.0, abs=0.01)
    # latg peak at 260 → 260 - 220 = +40 (after the apex)
    assert out["latg_peak_offset_m"] == pytest.approx(40.0, abs=0.01)
    assert out["latg_peak_dist_m"] == pytest.approx(260.0, abs=0.01)
    assert out["min_speed_dist_m"] == pytest.approx(180.0, abs=0.01)


def test_build_corner_transit_latg_peak_uses_abs_magnitude(make_lap_samples):
    # SPEC: build_corner_transit — latg_peak_dist_m is the max |lat_g| sample;
    #       max_lat_g is the max of |lat_g| (sign-agnostic magnitude)
    n = 31
    td = np.linspace(100.0, 300.0, n)
    apex_m = 200.0
    # A strongly NEGATIVE lat_g (left turn) is the largest magnitude.
    lat_g = np.full(n, 0.2)
    neg_idx = 20
    lat_g[neg_idx] = -1.4  # largest |lat_g|
    corner = _make_corner(start_m=100.0, end_m=300.0, apex_m=apex_m)
    lap = make_lap_samples(n=n, track_dist_m=td, lat_g=lat_g)
    out = build_corner_transit(lap, corner)
    assert out is not None
    assert out["latg_peak_dist_m"] == pytest.approx(td[neg_idx], abs=0.01)
    # Reported magnitude is positive even though the peak was negative.
    assert out["max_lat_g"] == pytest.approx(1.4, abs=0.01)


def test_build_corner_transit_input_offsets_from_start_m(make_lap_samples):
    # SPEC: build_corner_transit — brake_on/throttle_lift/wot offsets are from
    #       corner.start_m (value - start_m), or None when the event never occurs
    n = 41
    start_m = 100.0
    td = np.linspace(start_m, 300.0, n)
    # Brake turns on (0→1) at a known track_dist (index 10 → td ~ 150).
    brake = np.zeros(n, dtype=int)
    on_idx = 10
    brake[on_idx:on_idx + 8] = 1
    corner = _make_corner(start_m=start_m, end_m=300.0, apex_m=200.0)
    lap = make_lap_samples(n=n, track_dist_m=td, brake=brake)
    out = build_corner_transit(lap, corner)
    assert out is not None
    # brake_on offset is measured from start_m.
    assert out["brake_on_dist_m"] == pytest.approx(td[on_idx] - start_m, abs=5.0)
    assert out["brake_on_dist_m"] >= 0.0


def test_build_corner_transit_event_absent_is_none(make_lap_samples):
    # SPEC: build_corner_transit — offset is None when the event never occurs
    n = 31
    td = np.linspace(100.0, 300.0, n)
    # Brake never engages → brake_on offset should be None.
    corner = _make_corner(start_m=100.0, end_m=300.0, apex_m=200.0)
    lap = make_lap_samples(n=n, track_dist_m=td, brake=0)
    out = build_corner_transit(lap, corner)
    assert out is not None
    assert out["brake_on_dist_m"] is None


def test_build_corner_transit_peak_brake_is_int(make_lap_samples):
    # SPEC: build_corner_transit — peak_brake is an int (max of 0/1 brake channel)
    n = 31
    td = np.linspace(100.0, 300.0, n)
    brake = np.zeros(n, dtype=int)
    brake[15:20] = 1
    corner = _make_corner(start_m=100.0, end_m=300.0, apex_m=200.0)
    lap = make_lap_samples(n=n, track_dist_m=td, brake=brake)
    out = build_corner_transit(lap, corner)
    assert out is not None
    assert isinstance(out["peak_brake"], (int, np.integer))
    assert out["peak_brake"] == 1


def test_build_corner_transit_only_samples_inside_box(make_lap_samples):
    # SPEC: build_corner_transit — metrics are computed over samples in [start_m,end_m]
    # A min-speed sample OUTSIDE the box must not become the reported min.
    n = 41
    td = np.linspace(0.0, 400.0, n)  # spans well beyond the corner box
    speed = np.full(n, 90.0)
    # Slowest sample of the WHOLE lap sits outside the box (td near 0).
    speed[0] = 5.0
    # Slowest sample INSIDE [100, 300] sits at a known location.
    inside_mask = (td >= 100.0) & (td <= 300.0)
    inside_idx = np.where(inside_mask)[0]
    chosen = inside_idx[len(inside_idx) // 2]
    speed[chosen] = 35.0
    corner = _make_corner(start_m=100.0, end_m=300.0, apex_m=200.0)
    lap = make_lap_samples(n=n, track_dist_m=td, speed_mph=speed)
    out = build_corner_transit(lap, corner)
    assert out is not None
    # The reported min-speed location is the in-box minimum, not the global one.
    assert out["min_speed_dist_m"] == pytest.approx(td[chosen], abs=0.01)
    assert out["min_speed_mph"] == pytest.approx(35.0, abs=0.01)


# ---------------------------------------------------------------------------
# label_samples — corner segmentation + front-straight wrap (integration tier)
# ---------------------------------------------------------------------------

@pytest.fixture
def ridge_labeled_lap(sample_data_root, ridge_reference_index):
    """A single clean flying lap from a committed session, re-labeled by label_samples.

    The committed parquet is already labeled, but we re-run label_samples to test
    its contract directly. We strip the previously-added label columns first so
    we observe label_samples' own output.
    """
    track = load_track("ridge")
    samples = pd.read_parquet(
        sample_data_root / "sessions" / "ridge" / "20260517-100304" / "samples.parquet"
    )
    lap = samples[samples["lap"] == 3].copy()
    # Keep only normalized-session columns so label_samples computes labels fresh.
    drop = [c for c in lap.columns if c.startswith("gps_drift")
            or c in ("track_dist_m", "track_dist_offset_m", "corner")]
    lap = lap.drop(columns=[c for c in drop if c in lap.columns])
    return track, _call_label_samples(lap, track, ridge_reference_index)


def test_label_samples_adds_documented_columns(ridge_labeled_lap):
    # SPEC: label_samples — adds track_dist_m, track_dist_offset_m, drift cols, corner
    _track, labeled = ridge_labeled_lap
    for col in ("track_dist_m", "track_dist_offset_m", "corner"):
        assert col in labeled.columns


def test_label_samples_every_sample_gets_one_label(ridge_labeled_lap):
    # SPEC: label_samples — every sample gets exactly one corner label
    _track, labeled = ridge_labeled_lap
    assert labeled["corner"].notna().all()
    assert (labeled["corner"].astype(str).str.len() > 0).all()


def test_label_samples_corner_labels_drawn_from_track(ridge_labeled_lap):
    # SPEC: label_samples — corner boxes labeled by corner id; gaps are straights
    track, labeled = ridge_labeled_lap
    corner_ids = {c.id for c in track.corners}
    labels = set(labeled["corner"].unique())
    # Every label is either a known corner id or an S_post_<id> straight.
    for lab in labels:
        assert (lab in corner_ids) or lab.startswith("S_post_")


def test_label_samples_front_straight_wraps_to_last_corner(ridge_labeled_lap):
    # SPEC: label_samples — the segment before the first corner is S_post_<last-corner-id>
    track, labeled = ridge_labeled_lap
    # Corners are sorted ascending by start_m; the last one's id owns the wrap.
    last_corner_id = track.corners[-1].id
    wrap_label = f"S_post_{last_corner_id}"
    labels = set(labeled["corner"].unique())
    # The wrap label must appear: samples before the first corner box exist on a
    # full lap (the start/finish straight).
    assert wrap_label in labels


def test_label_samples_straight_label_format(ridge_labeled_lap):
    # SPEC: label_samples — a straight is labeled S_post_<id-of-preceding-corner>
    track, labeled = ridge_labeled_lap
    corner_ids = {c.id for c in track.corners}
    straight_labels = {lab for lab in labeled["corner"].unique()
                       if isinstance(lab, str) and lab.startswith("S_post_")}
    assert straight_labels, "expected at least one straight segment on a full lap"
    for lab in straight_labels:
        preceding = lab[len("S_post_"):]
        assert preceding in corner_ids


# ---------------------------------------------------------------------------
# build_session_corners — clean-lap-only transit emission (integration tier)
# ---------------------------------------------------------------------------

@pytest.fixture
def ridge_session_corners(sample_data_root):
    # build_session_corners(samples, laps, track, session_id, date) operates on
    # already-labeled samples (the sample bundle's samples.parquet carries
    # track_dist_m/corner/drift columns) — it does not read a session dir.
    track = load_track("ridge")
    session_id = "20260517-100304"
    session_dir = sample_data_root / "sessions" / "ridge" / session_id
    samples = pd.read_parquet(session_dir / "samples.parquet")
    laps = pd.read_csv(session_dir / "laps.csv")
    date = f"{session_id[:4]}-{session_id[4:6]}-{session_id[6:8]}"
    corners = build_session_corners(samples, laps, track, session_id, date)
    return track, corners


def test_build_session_corners_has_identity_and_drift_columns(ridge_session_corners):
    # SPEC: build_session_corners — emits session_id, date, lap, corner_id + drift cols
    _track, corners = ridge_session_corners
    assert isinstance(corners, pd.DataFrame)
    for col in ("session_id", "date", "lap", "corner_id"):
        assert col in corners.columns
    for col in ("gps_drift_lat_m", "gps_drift_lon_m",
                "gps_drift_n_anchors", "gps_drift_disagreement_m"):
        assert col in corners.columns


def test_build_session_corners_clean_laps_only(sample_data_root, ridge_session_corners):
    # SPEC: build_session_corners — only laps with is_clean == True are processed
    _track, corners = ridge_session_corners
    laps = pd.read_csv(
        sample_data_root / "sessions" / "ridge" / "20260517-100304" / "laps.csv"
    )
    clean_laps = set(laps[laps["is_clean"]]["lap"].astype(int))
    emitted_laps = set(corners["lap"].astype(int).unique())
    # No warmup/cooldown (non-clean) lap should appear in the transit table.
    assert emitted_laps.issubset(clean_laps)


def test_build_session_corners_one_row_per_clean_lap_x_corner(ridge_session_corners):
    # SPEC: build_session_corners — one row per (clean lap × corner)
    _track, corners = ridge_session_corners
    # No (lap, corner_id) pair should be duplicated.
    dupes = corners.duplicated(subset=["lap", "corner_id"]).sum()
    assert dupes == 0


def test_build_session_corners_corner_ids_are_track_corners(ridge_session_corners):
    # SPEC: build_session_corners — corner_id values come from the track's corners
    track, corners = ridge_session_corners
    corner_ids = {c.id for c in track.corners}
    emitted = set(corners["corner_id"].unique())
    assert emitted.issubset(corner_ids)
