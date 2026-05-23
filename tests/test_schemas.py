"""Clean-room tests for lap_analyzer.schemas.SessionMeta.

Source of truth: tests/SPEC.md "## schemas.SessionMeta". SessionMeta is a
pydantic v2 BaseModel. Behavior derived from the spec only.
"""
from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from lap_analyzer.schemas import SessionMeta


# The 12 required fields per SPEC, with spec-typed sample values.
REQUIRED = dict(
    session_id="20260101-120000",
    track="ridge",
    vehicle="Porsche 981 Cayman",
    date_utc=datetime(2026, 1, 1, 12, 0, 0),
    n_laps=5,
    n_clean_laps=3,
    sample_rate_hz=21.0,
    duration_s=600.0,
    throttle_max_observed=98.5,
    speed_max_obd_mph=120.0,
    rpm_max=7200,
    raw_csv_path="data/raw/ridge/Log-20260101-120000.csv",
)

# The optional fields that SPEC says default to None.
OPTIONAL_NONE_FIELDS = [
    "best_lap",
    "best_lap_time_s",
    "coolant_min_f",
    "coolant_max_f",
    "iat_first_f",
    "iat_max_f",
    "trackaddict_start_finish",
]


# SPEC: schemas.SessionMeta — required-only construction succeeds
def test_required_only_construction_succeeds():
    meta = SessionMeta(**REQUIRED)
    assert meta.session_id == "20260101-120000"
    assert meta.track == "ridge"
    assert meta.n_laps == 5


# SPEC: schemas.SessionMeta — optionals default to None when omitted
@pytest.mark.parametrize("field", OPTIONAL_NONE_FIELDS)
def test_optional_fields_default_none(field):
    meta = SessionMeta(**REQUIRED)
    assert getattr(meta, field) is None


# SPEC: schemas.SessionMeta — trackaddict_split_points defaults to []
def test_split_points_defaults_to_empty_list():
    meta = SessionMeta(**REQUIRED)
    assert meta.trackaddict_split_points == []


# SPEC: schemas.SessionMeta — model_dump_json -> model_validate_json round-trips
def test_json_round_trip():
    meta = SessionMeta(**REQUIRED)
    restored = SessionMeta.model_validate_json(meta.model_dump_json())
    assert restored == meta


# SPEC: schemas.SessionMeta — round-trips with optionals populated too
def test_json_round_trip_with_optionals():
    meta = SessionMeta(
        **REQUIRED,
        best_lap=2,
        best_lap_time_s=95.123,
        coolant_min_f=180.0,
        coolant_max_f=210.0,
        iat_first_f=70.0,
        iat_max_f=110.0,
        trackaddict_start_finish={"lat": 45.0, "long": -122.0},
        trackaddict_split_points=[{"index": 1, "lat": 45.1, "long": -122.1}],
    )
    restored = SessionMeta.model_validate_json(meta.model_dump_json())
    assert restored == meta
    assert restored.trackaddict_split_points == [
        {"index": 1, "lat": 45.1, "long": -122.1}
    ]


# SPEC: schemas.SessionMeta — omitting a required field raises ValidationError
@pytest.mark.parametrize("missing", list(REQUIRED.keys()))
def test_missing_required_field_raises(missing):
    kwargs = {k: v for k, v in REQUIRED.items() if k != missing}
    with pytest.raises(ValidationError):
        SessionMeta(**kwargs)


# SPEC: schemas.SessionMeta — trackaddict_start_finish accepts a dict
def test_start_finish_accepts_dict():
    meta = SessionMeta(**REQUIRED, trackaddict_start_finish={"lat": 1.0, "long": 2.0})
    assert meta.trackaddict_start_finish == {"lat": 1.0, "long": 2.0}
