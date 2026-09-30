from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SessionMeta(BaseModel):
    session_id: str
    track: str
    vehicle: str
    date_utc: datetime
    n_laps: int
    n_clean_laps: int
    sample_rate_hz: float
    duration_s: float
    best_lap: int | None = None
    best_lap_time_s: float | None = None
    # OBD-derived fields are None for GPS-only (OBD-dropout) sessions.
    has_obd: bool = True
    throttle_max_observed: float | None = None
    speed_max_obd_mph: float | None = None
    rpm_max: int | None = None
    coolant_min_f: float | None = None
    coolant_max_f: float | None = None
    iat_first_f: float | None = None
    iat_max_f: float | None = None
    trackaddict_start_finish: dict | None = None
    trackaddict_split_points: list[dict] = []
    raw_csv_path: str
    # GPS lag correction provenance (gps_lag.summarize_lags). None = normalized before
    # gps-lag-v1 (stale); method_version is the staleness marker.
    gps_lag: dict | None = None
