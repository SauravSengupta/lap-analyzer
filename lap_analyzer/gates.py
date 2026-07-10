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
