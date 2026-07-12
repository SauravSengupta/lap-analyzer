import math

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.centerline import M_PER_DEG_LAT
from lap_analyzer.gates import TrackFrame, build_gate, gate_crossing_time


def _straight_centerline(n=1001, lat=47.0, lon0=-123.0, length_deg=0.02):
    """East-west centerline at constant latitude; track_dist_m is cumulative meters."""
    lon = np.linspace(lon0, lon0 + length_deg, n)
    lat_arr = np.full(n, lat)
    frame_m_per_deg_lon = 111_132.0 * np.cos(np.radians(lat))
    dist = (lon - lon0) * frame_m_per_deg_lon
    return pd.DataFrame({"track_dist_m": dist, "lat": lat_arr, "long": lon})


def _snaking_centerline(n=2001, lat0=47.0, lon0=-123.0, length_m=2000.0,
                        amp_m=5.0, wavelength_m=18.0):
    """A dead-straight (due-east) track carrying the centerline's transverse
    snaking artifact — the ~5 m amplitude / ~18 m wavelength oscillation the
    synthetic centerline is known to have (design spec R3, 2026-07-11).

    ``track_dist_m`` is the underlying straight-line distance, so the TRUE track
    tangent is due east at every point and a correct gate is always a
    north-south segment. Any rotation of the gate away from north-south is pure
    artifact — exactly what the smoothed tangent must reject.
    """
    m_per_deg_lat = M_PER_DEG_LAT
    m_per_deg_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))
    s = np.linspace(0.0, length_m, n)
    x = s                                              # east, metres
    y = amp_m * np.sin(2.0 * np.pi * s / wavelength_m)  # north snaking, metres
    lat = lat0 + y / m_per_deg_lat
    lon = lon0 + x / m_per_deg_lon
    return pd.DataFrame({"track_dist_m": s, "lat": lat, "long": lon})


def _acute_angle_deg(vx, vy, wx=1.0, wy=0.0):
    """Acute angle (degrees) between (vx,vy) and (wx,wy), ignoring 180° flips."""
    vn = math.hypot(vx, vy)
    wn = math.hypot(wx, wy)
    c = abs((vx * wx + vy * wy) / (vn * wn))
    return math.degrees(math.acos(min(1.0, c)))


def _gate_tangent_rotation_deg(gate, true_tangent=(1.0, 0.0)):
    """How far the gate's implied track tangent is rotated from ``true_tangent``.

    The gate segment is laid perpendicular to the tangent, so rotating the
    segment direction by 90° recovers the tangent the gate was built from.
    """
    d = gate.p1 - gate.p2
    return _acute_angle_deg(d[1], -d[0], *true_tangent)


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


def test_build_gate_tangent_survives_centerline_snaking():
    # The gate tangent must come from a σ=10 m smoothed fit over a ±20 m window,
    # not a 2-sample baseline (design spec R3 / PR 0, 2026-07-11). Evaluated at a
    # snake-slope MAXIMUM (a whole number of 18 m wavelengths in), where the
    # centerline artifact is worst, the gate must still point true (north-south)
    # within 3° so it never converts a lateral line offset into crossing-time error.
    cl = _snaking_centerline()
    frame = TrackFrame.from_centerline(cl)
    dist_m = 990.0  # 55 * 18 m — cos(2πs/λ) = 1, the steepest point of the snake
    gate = build_gate(cl, dist_m, frame, half_width_m=40.0)
    assert _gate_tangent_rotation_deg(gate) < 3.0


def test_two_sample_tangent_is_the_bug_being_fixed():
    # Documents WHY the smoothed tangent is required: the retired 2-sample tangent
    # (centerline i-1 .. i+1) is rotated tens of degrees off true at the same
    # snake-slope peak — the >3° failure the test above now guards against.
    cl = _snaking_centerline()
    frame = TrackFrame.from_centerline(cl)
    dist_m = 990.0
    cd = cl["track_dist_m"].to_numpy()
    x, y = frame.to_xy(cl["lat"].to_numpy(), cl["long"].to_numpy())
    i = int(np.argmin(np.abs(cd - dist_m)))
    i0, i1 = max(0, i - 1), min(len(cd) - 1, i + 1)
    tx, ty = x[i1] - x[i0], y[i1] - y[i0]  # the old, retired tangent
    assert _acute_angle_deg(tx, ty) > 40.0


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
