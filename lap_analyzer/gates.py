"""Transponder-style gate crossing for section timing.

Section boundaries are physical gates laid across the track (perpendicular to the
centerline tangent), not 1-D `track_dist_m` thresholds. A lap's section time is
the elapsed time between where its (lat,long) path crosses the entry gate and the
exit gate. A purely lateral GPS offset — which mis-times the centerline
projection on a curved corner — does not move a perpendicular gate crossing, so
genuine racing-line variation is preserved and GPS arc-compression is rejected.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .centerline import M_PER_DEG_LAT


@dataclass
class TrackFrame:
    """Equirectangular lat/long -> local meters, centred on the centerline."""

    lat0: float
    lon0: float
    m_per_deg_lat: float
    m_per_deg_lon: float

    @classmethod
    def from_centerline(cls, centerline: pd.DataFrame) -> "TrackFrame":
        lat0 = float(centerline["lat"].mean())
        lon0 = float(centerline["long"].mean())
        return cls(
            lat0=lat0,
            lon0=lon0,
            m_per_deg_lat=M_PER_DEG_LAT,
            m_per_deg_lon=M_PER_DEG_LAT * math.cos(math.radians(lat0)),
        )

    def to_xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        x = (np.asarray(lon, dtype=float) - self.lon0) * self.m_per_deg_lon
        y = (np.asarray(lat, dtype=float) - self.lat0) * self.m_per_deg_lat
        return x, y


@dataclass
class Gate:
    """A finite line segment across the track, in local meters."""

    p1: np.ndarray  # [x, y]
    p2: np.ndarray  # [x, y]


# 2026-07-11, gps-trust PR 0 (design spec R3): the tangent that orients a gate is
# a Gaussian-weighted linear regression of centerline position on track_dist_m
# over a +/-TANGENT_HALF_WINDOW_M window with kernel scale TANGENT_SIGMA_M. The
# retired 2-sample tangent read the synthetic centerline's ~5 m / ~18 m snaking
# artifact as real heading and rotated gates up to 59 deg (median 9, p90 46), which
# turns an ordinary 3-5 m lateral line offset into ~0.1-0.17 s of spurious crossing
# time. A +/-20 m window spans >2 snake wavelengths, so the oscillation cancels.
TANGENT_SIGMA_M = 10.0
TANGENT_HALF_WINDOW_M = 20.0


def _smoothed_tangent(
    cd: np.ndarray, x: np.ndarray, y: np.ndarray, dist_m: float,
    sigma_m: float = TANGENT_SIGMA_M, half_window_m: float = TANGENT_HALF_WINDOW_M,
) -> tuple[float, float] | None:
    """Unit track tangent (dx/ds, dy/ds) at `dist_m` from a Gaussian-weighted
    local linear regression of position on track_dist_m. Returns None if the
    window is too sparse or degenerate to fit a direction."""
    win = np.abs(cd - dist_m) <= half_window_m
    s = cd[win]
    if s.size < 2:
        return None
    w = np.exp(-0.5 * ((s - dist_m) / sigma_m) ** 2)
    sw = w.sum()
    if sw <= 0.0:
        return None
    s_c = s - np.sum(w * s) / sw
    ss = np.sum(w * s_c * s_c)
    if ss <= 0.0:                               # all evidence at one distance
        return None
    tx = np.sum(w * s_c * (x[win] - np.sum(w * x[win]) / sw)) / ss   # slope dx/ds
    ty = np.sum(w * s_c * (y[win] - np.sum(w * y[win]) / sw)) / ss   # slope dy/ds
    tnorm = math.hypot(tx, ty)
    if tnorm == 0.0:
        return None
    return tx / tnorm, ty / tnorm


def build_gate(
    centerline: pd.DataFrame,
    dist_m: float,
    frame: TrackFrame,
    half_width_m: float = 40.0,
) -> Gate:
    """Gate perpendicular to the centerline tangent at `dist_m`, +/- half_width_m wide."""
    cd = centerline["track_dist_m"].to_numpy()
    x, y = frame.to_xy(centerline["lat"].to_numpy(), centerline["long"].to_numpy())
    i = int(np.argmin(np.abs(cd - dist_m)))
    cx, cy = x[i], y[i]
    tangent = _smoothed_tangent(cd, x, y, dist_m)
    if tangent is None:
        # The ±20m regression window was too sparse to fit a direction. Unreachable
        # on the ridge centerline (all 32 gates sit ≥219m inside coverage) but
        # reachable on a sparse bootstrap centerline for a new track — warn rather
        # than silently lay a gate, and fall back to the nearest-two-point tangent
        # (best available; a sparse centerline has no snaking artifact to smooth).
        warnings.warn(
            f"build_gate: no smoothed tangent at dist_m={dist_m:.1f} "
            f"(centerline sparse within ±{TANGENT_HALF_WINDOW_M:.0f}m); "
            "falling back to a 2-point tangent",
            stacklevel=2,
        )
        i0, i1 = max(0, i - 1), min(len(cd) - 1, i + 1)
        tx, ty = x[i1] - x[i0], y[i1] - y[i0]
        tnorm = math.hypot(tx, ty) or 1.0
        tx, ty = tx / tnorm, ty / tnorm
    else:
        tx, ty = tangent
    px, py = -ty, tx                            # unit perpendicular (tangent is unit)
    return Gate(
        p1=np.array([cx + px * half_width_m, cy + py * half_width_m]),
        p2=np.array([cx - px * half_width_m, cy - py * half_width_m]),
    )


def _segment_cross_frac(p, r, a, b) -> float | None:
    """Fraction along path step p->p+r where it crosses gate segment a->b, or None."""
    s = b - a
    denom = r[0] * s[1] - r[1] * s[0]
    if denom == 0.0:
        return None
    qp = a - p
    tf = (qp[0] * s[1] - qp[1] * s[0]) / denom   # along the path step
    u = (qp[0] * r[1] - qp[1] * r[0]) / denom     # along the gate
    if 0.0 <= tf <= 1.0 and 0.0 <= u <= 1.0:
        return float(tf)
    return None


def gate_crossing_time(
    lat, lon, t, track_dist_m,
    gate: Gate,
    frame: TrackFrame,
    seed_dist_m: float,
    seed_window_m: float = 120.0,
) -> float | None:
    """Interpolated time the (lat,long) path crosses `gate`, near `seed_dist_m`."""
    x, y = frame.to_xy(lat, lon)
    t = np.asarray(t, dtype=float)
    td = np.asarray(track_dist_m, dtype=float)
    n = len(x)
    if n < 2:
        return None
    in_win = np.abs(td - seed_dist_m) <= seed_window_m
    a, b = gate.p1, gate.p2
    best_key = None
    best_time = None
    for i in range(n - 1):
        if not (in_win[i] or in_win[i + 1]):
            continue
        p = np.array([x[i], y[i]])
        r = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])
        frac = _segment_cross_frac(p, r, a, b)
        if frac is None:
            continue
        cross_time = float(t[i] + frac * (t[i + 1] - t[i]))
        cross_dist = float(td[i] + frac * (td[i + 1] - td[i]))
        key = abs(cross_dist - seed_dist_m)
        if best_key is None or key < best_key:
            best_key, best_time = key, cross_time
    return best_time
