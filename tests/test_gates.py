import numpy as np
import pandas as pd
import pytest

from lap_analyzer.gates import Gate, TrackFrame, build_gate


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
    mid = float(cl["track_dist_m"].iloc[len(cl) // 2])
    gate = build_gate(cl, mid, frame, half_width_m=40.0)
    # Gate is perpendicular to an east-west track => a north-south segment:
    # its two endpoints share x and differ in y by 2*half_width.
    assert gate.p1[0] == pytest.approx(gate.p2[0], abs=1e-6)
    assert abs(gate.p1[1] - gate.p2[1]) == pytest.approx(80.0, abs=1e-3)
    # Centre of the gate sits on the centerline point at `mid` (x == mid meters here).
    assert (gate.p1[0] + gate.p2[0]) / 2 == pytest.approx(mid, abs=0.5)
