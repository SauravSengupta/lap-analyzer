from __future__ import annotations

from datetime import datetime
from typing import Optional

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
    best_lap: Optional[int] = None
    best_lap_time_s: Optional[float] = None
    throttle_max_observed: float
    speed_max_obd_mph: float
    rpm_max: int
    coolant_min_f: Optional[float] = None
    coolant_max_f: Optional[float] = None
    iat_first_f: Optional[float] = None
    iat_max_f: Optional[float] = None
    trackaddict_start_finish: Optional[dict] = None
    trackaddict_split_points: list[dict] = []
    raw_csv_path: str
