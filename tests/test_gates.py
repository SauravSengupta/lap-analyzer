import numpy as np
import pandas as pd
import pytest

from lap_analyzer.gates import TrackFrame, build_gate, gate_crossing_time


def _straight_centerline(n=1001, lat=47.0, lon0=-123.0, length_deg=0.02):
    """East-west centerline at constant latitude; track_dist_m is cumulative meters."""
    lon = np.linspace(lon0, lon0 + length_deg, n)
    lat_arr = np.full(n, lat)
    frame_m_per_deg_lon = 111_132.0 * np.cos(np.radians(lat))
    dist = (lon - lon0) * frame_m_per_deg_lon
    return pd.DataFrame({"track_dist_m": dist, "lat": lat_arr, "long": lon})


def test_trackframe_to_xy_maps_degrees_to_meters():
    cl = _straight_centerline()
    frame = TrackFrame.from_centerline(cl)
    # A point one lon-step east of frame origin maps to a positive x (east) in meters,
    # and same-latitude points map to y == 0.
    x, y = frame.to_xy(np.array([47.0, 47.0]), np.array([frame.lon0, frame.lon0 + 0.001]))
    assert y[0] == pytest.approx(0.0, abs=1e-6)
    assert x[0] == pytest.approx(0.0, abs=1e-6)
    assert x[1] == pytest.approx(0.001 * frame.m_per_deg_lon, abs=1e-6)
    assert x[1] > 0


def test_build_gate_is_perpendicular_and_centered():
    cl = _straight_centerline()          # east-west line, tangent points +x
    frame = TrackFrame.from_centerline(cl)
    i = len(cl) // 2
    mid = float(cl["track_dist_m"].iloc[i])
    gate = build_gate(cl, mid, frame, half_width_m=40.0)
    # Gate is perpendicular to an east-west track => a north-south segment:
    # its two endpoints share x and differ in y by 2*half_width.
    assert gate.p1[0] == pytest.approx(gate.p2[0], abs=1e-6)
    assert abs(gate.p1[1] - gate.p2[1]) == pytest.approx(80.0, abs=1e-3)
    # Centre of the gate is the centerline point at `mid`, expressed in the SAME
    # frame the crossing code uses (frame.to_xy) — NOT the raw track_dist_m value.
    cx_exp, cy_exp = frame.to_xy(np.array([cl["lat"].iloc[i]]), np.array([cl["long"].iloc[i]]))
    assert (gate.p1[0] + gate.p2[0]) / 2 == pytest.approx(cx_exp[0], abs=0.5)
    assert (gate.p1[1] + gate.p2[1]) / 2 == pytest.approx(cy_exp[0], abs=0.5)


def _run_lap_along(cl, lat_offset_deg=0.0, n=400, speed_mps=40.0):
    """A lap driving east along the centerline (optionally shifted north by
    lat_offset_deg), sampled at 10 Hz. Returns arrays (lat, lon, t, track_dist_m).
    track_dist_m is the *centerline* projection distance (independent of lat_offset)."""
    lon = np.linspace(cl["long"].iloc[0], cl["long"].iloc[-1], n)
    lat = np.full(n, cl["lat"].iloc[0] + lat_offset_deg)
    frame = TrackFrame.from_centerline(cl)
    m_per_deg_lon = frame.m_per_deg_lon
    track_dist = (lon - cl["long"].iloc[0]) * m_per_deg_lon
    t = np.arange(n) / 10.0
    return lat, lon, t, track_dist


def test_gate_crossing_time_straight_matches_expected():
    cl = _straight_centerline()
    frame = TrackFrame.from_centerline(cl)
    lat, lon, t, td = _run_lap_along(cl)
    gate_dist = 400.0
    gate = build_gate(cl, gate_dist, frame)
    ct = gate_crossing_time(lat, lon, t, td, gate, frame, seed_dist_m=gate_dist)
    assert ct is not None
    # Crossing time == time when track_dist_m == gate_dist (interp), since the lap
    # drives straight down the centerline.
    expected = float(np.interp(gate_dist, td, t))
    assert ct == pytest.approx(expected, abs=0.05)


def test_gate_crossing_time_is_immune_to_lateral_offset():
    # The whole point: a lap shifted laterally (but same longitudinal progress)
    # crosses the perpendicular gate at the SAME time as the on-centerline lap.
    cl = _straight_centerline()
    frame = TrackFrame.from_centerline(cl)
    gate = build_gate(cl, 400.0, frame, half_width_m=40.0)
    on = gate_crossing_time(*_run_lap_along(cl, lat_offset_deg=0.0), gate, frame, 400.0)
    # Shift north by ~15 m (well within the 40 m half-width).
    off_deg = 15.0 / frame.m_per_deg_lat
    off = gate_crossing_time(*_run_lap_along(cl, lat_offset_deg=off_deg), gate, frame, 400.0)
    assert on is not None and off is not None
    assert off == pytest.approx(on, abs=0.02)


def test_gate_crossing_time_miss_returns_none():
    cl = _straight_centerline()
    frame = TrackFrame.from_centerline(cl)
    gate = build_gate(cl, 400.0, frame, half_width_m=40.0)
    # Lap shifted north by 100 m — beyond the 40 m half-width, so it never crosses.
    off_deg = 100.0 / frame.m_per_deg_lat
    ct = gate_crossing_time(*_run_lap_along(cl, lat_offset_deg=off_deg), gate, frame, 400.0)
    assert ct is None


def test_gate_crossing_time_picks_crossing_nearest_seed():
    # SPEC: when the path crosses a gate twice within the seed window, choose the
    # crossing whose track_dist_m is nearest seed_dist_m.
    cl = _straight_centerline()
    frame = TrackFrame.from_centerline(cl)
    gate = build_gate(cl, 400.0, frame, half_width_m=40.0)   # N-S line at x = gx
    gx = gate.p1[0]                                           # gate centre x in the frame
    lat0 = cl["lat"].iloc[0]
    xs = np.array([gx - 20.0, gx + 20.0, gx - 20.0])         # weave across gx twice
    lon = frame.lon0 + xs / frame.m_per_deg_lon              # back to lon so to_xy(lon).x == xs
    lat = np.full(3, lat0)
    t = np.array([0.0, 1.0, 2.0])
    td = np.array([380.0, 420.0, 460.0])                     # track_dist advances forward
    # First crossing at track_dist 400 (t=0.5), second at 440 (t=1.5).
    near_first = gate_crossing_time(lat, lon, t, td, gate, frame, seed_dist_m=400.0)
    near_second = gate_crossing_time(lat, lon, t, td, gate, frame, seed_dist_m=460.0)
    assert near_first == pytest.approx(0.5, abs=0.05)
    assert near_second == pytest.approx(1.5, abs=0.05)
