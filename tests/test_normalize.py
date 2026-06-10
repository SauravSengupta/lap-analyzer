"""Clean-room tests for the pure functions in lap_analyzer.normalize.

Source of truth: tests/SPEC.md "## normalize.*" (everything EXCEPT the
"## normalize CLI" section, which belongs to another author). Behavior derived
from the spec + docs/PIPELINE.md + docs/ARCHITECTURE.md only — no inspection of
lap_analyzer source.

Covered: session_id_from_filename, _parse_coords, parse_metadata_header,
normalize_dataframe, compute_lap_times, lap_summary, reference_session_ids,
and the MissingOBDError path of normalize_session.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.normalize import (
    MissingOBDError,
    _parse_coords,
    compute_lap_times,
    lap_summary,
    normalize_dataframe,
    normalize_session,
    parse_metadata_header,
    reference_session_ids,
    session_id_from_filename,
)


# ---------------------------------------------------------------------------
# session_id_from_filename
# ---------------------------------------------------------------------------


# SPEC: normalize.session_id_from_filename — extracts YYYYMMDD-HHMMSS from Log-<id>
def test_session_id_basic():
    assert session_id_from_filename("Log-20260517-100304 Ridge.csv") == "20260517-100304"


# SPEC: normalize.session_id_from_filename — regex search anywhere in basename
def test_session_id_anywhere_in_name():
    assert session_id_from_filename("prefix Log-20240829-103904 stuff.csv") == "20240829-103904"


# SPEC: normalize.session_id_from_filename — works on a full path (uses basename)
def test_session_id_from_full_path():
    p = Path("data") / "raw" / "ridge" / "Log-20251018-105948 Session.csv"
    assert session_id_from_filename(str(p)) == "20251018-105948"


# SPEC: normalize.session_id_from_filename — no Log-<8>-<6> pattern raises ValueError
def test_session_id_no_pattern_raises():
    with pytest.raises(ValueError):
        session_id_from_filename("some_random_file.csv")


# SPEC: normalize.session_id_from_filename — partial/malformed digit groups raise
def test_session_id_wrong_digit_count_raises():
    with pytest.raises(ValueError):
        session_id_from_filename("Log-2026-100304 Ridge.csv")


# ---------------------------------------------------------------------------
# _parse_coords
# ---------------------------------------------------------------------------


# SPEC: normalize._parse_coords — parses "<lat>, <long>" into dict
def test_parse_coords_basic():
    assert _parse_coords("45.5, -122.6") == {"lat": 45.5, "long": -122.6}


# SPEC: normalize._parse_coords — ignores anything after an @ and a trailing comma
def test_parse_coords_ignores_after_at():
    assert _parse_coords("45.5, -122.6 @ 2026-01-01 12:00:00") == {
        "lat": 45.5,
        "long": -122.6,
    }


# SPEC: normalize._parse_coords — fewer than 2 comma-parts -> None
def test_parse_coords_too_few_parts():
    assert _parse_coords("45.5") is None


# SPEC: normalize._parse_coords — a non-float part -> None
def test_parse_coords_non_float():
    assert _parse_coords("north, -122.6") is None


# ---------------------------------------------------------------------------
# parse_metadata_header
# ---------------------------------------------------------------------------


# SPEC: normalize.parse_metadata_header — recognizes Vehicle / End Point / Split Point
def test_parse_metadata_recognized_lines(tmp_path):
    p = tmp_path / "Log-20260101-120000.csv"
    p.write_text(
        "# Vehicle: Porsche 981 Cayman\n"
        "# End Point: 45.5, -122.6 @ 2026-01-01 12:00:00\n"
        "# Split Point 1: 45.1, -122.1 @ 2026-01-01 12:00:01\n"
        "# Split Point 2: 45.2, -122.2 @ 2026-01-01 12:00:02\n"
        "Time,Lap\n0,1\n",
        encoding="utf-8",
    )
    meta = parse_metadata_header(p)
    assert meta["vehicle"] == "Porsche 981 Cayman"
    assert meta["start_finish"] == {"lat": 45.5, "long": -122.6}
    assert meta["split_points"] == [
        {"index": 1, "lat": 45.1, "long": -122.1},
        {"index": 2, "lat": 45.2, "long": -122.2},
    ]


# SPEC: normalize.parse_metadata_header — always carries raw_csv_path and split_points list
def test_parse_metadata_always_has_keys(tmp_path):
    p = tmp_path / "Log-20260101-120000.csv"
    p.write_text("Time,Lap\n0,1\n", encoding="utf-8")  # no comment block at all
    meta = parse_metadata_header(p)
    assert "raw_csv_path" in meta
    assert isinstance(meta["raw_csv_path"], str)
    assert meta["split_points"] == []


# SPEC: normalize.parse_metadata_header — stops at first non-# line
def test_parse_metadata_stops_at_first_data_line(tmp_path):
    p = tmp_path / "Log-20260101-120000.csv"
    p.write_text(
        "# Vehicle: Real Car\n"
        "Time,Lap\n"
        "# Vehicle: Should Be Ignored\n",  # past the comment block — must be ignored
        encoding="utf-8",
    )
    meta = parse_metadata_header(p)
    assert meta["vehicle"] == "Real Car"


# SPEC: normalize.parse_metadata_header — unrecognized comment lines are ignored
def test_parse_metadata_ignores_unknown_comments(tmp_path):
    p = tmp_path / "Log-20260101-120000.csv"
    p.write_text(
        "# Some Unknown Header: value\n"
        "# Device: phone\n"
        "# Vehicle: Known Car\n"
        "Time,Lap\n0,1\n",
        encoding="utf-8",
    )
    meta = parse_metadata_header(p)
    assert meta["vehicle"] == "Known Car"
    assert meta["split_points"] == []


# ---------------------------------------------------------------------------
# normalize_dataframe — helpers
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS = [
    "session_id", "t", "lap", "dist_m", "dist_lap_m", "speed_mph",
    "speed_mph_gps", "throttle_norm", "brake", "rpm", "lat_g", "long_g",
    "coolant_f", "iat_f", "lat", "long", "altitude_m", "gps_accuracy_m",
]


def _raw_frame(n=20, **overrides):
    """A canonical-named raw frame (the shape read_csv hands normalize_dataframe).

    Single lap by default. Channels overridable. Not derived from sample data.
    """
    t = np.arange(n) * 0.1
    base = {
        "t": t,
        "lap": np.ones(n, dtype=int),
        "lat": np.full(n, 45.0),
        "long": np.full(n, -122.0),
        "speed_mph": np.full(n, 60.0),
        "speed_mph_gps": np.full(n, 60.0),
        "lat_g": np.zeros(n),
        "long_g": np.zeros(n),
        "brake": np.zeros(n, dtype=int),
        "rpm": np.full(n, 4000.0),
        "throttle_raw": np.full(n, 50.0),
        "coolant_f": np.full(n, 200.0),
        "iat_f": np.full(n, 90.0),
        # manifold_psi is dropped from the output, but normalize_dataframe still
        # forward-fills all 6 OBD channels, so it must be present in the input.
        "manifold_psi": np.full(n, 14.0),
        "altitude_m": np.full(n, 100.0),
        "gps_accuracy_m": np.full(n, 3.0),
    }
    for k, v in overrides.items():
        base[k] = np.asarray(v) if np.ndim(v) else np.full(n, v)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# normalize_dataframe — contract / columns
# ---------------------------------------------------------------------------


# SPEC: normalize.normalize_dataframe — returns (df, {"throttle_max_observed": float})
def test_normalize_returns_tuple_with_throttle_max():
    out, info = normalize_dataframe(_raw_frame(), "20260101-120000")
    assert isinstance(out, pd.DataFrame)
    assert isinstance(info, dict)
    assert "throttle_max_observed" in info
    assert isinstance(float(info["throttle_max_observed"]), float)


# SPEC: normalize.normalize_dataframe — output has exactly the documented columns
def test_normalize_output_columns_exact():
    out, _ = normalize_dataframe(_raw_frame(), "20260101-120000")
    assert set(out.columns) == set(OUTPUT_COLUMNS)


# SPEC: normalize.normalize_dataframe — session_id is stamped onto every row
def test_normalize_stamps_session_id():
    out, _ = normalize_dataframe(_raw_frame(), "20260101-120000")
    assert (out["session_id"] == "20260101-120000").all()


# SPEC: normalize.normalize_dataframe — throttle_max_observed reports the raw max
def test_normalize_throttle_max_observed_is_raw_max():
    raw = _raw_frame(throttle_raw=np.linspace(0, 80, 20))
    _, info = normalize_dataframe(raw, "20260101-120000")
    assert info["throttle_max_observed"] == pytest.approx(80.0, abs=0.01)


# ---------------------------------------------------------------------------
# normalize_dataframe — lat_g / long_g negation (the key axis invariants)
# ---------------------------------------------------------------------------


# SPEC: normalize.normalize_dataframe — canonical lat_g == -raw["lat_g"] (exact)
def test_normalize_lat_g_negated_exactly():
    raw = _raw_frame(lat_g=np.array([0.3, -0.5, 0.0, 0.8] + [0.0] * 16))
    out, _ = normalize_dataframe(raw, "20260101-120000")
    expected = -raw["lat_g"].to_numpy()
    assert np.array_equal(out["lat_g"].to_numpy(), expected)


# SPEC: normalize.normalize_dataframe — raw left-turn (raw>0) becomes negative; +lat_g = right
def test_normalize_lat_g_right_turn_positive():
    raw = _raw_frame(lat_g=-0.7)  # raw right-turn convention -> canonical positive
    out, _ = normalize_dataframe(raw, "20260101-120000")
    assert (out["lat_g"] > 0).all()  # positive lat_g = right turn


# SPEC: normalize.normalize_dataframe — canonical long_g == -raw["long_g"] (exact)
def test_normalize_long_g_negated_exactly():
    raw = _raw_frame(long_g=np.array([0.9, -0.4, 0.0, 0.2] + [0.0] * 16))
    out, _ = normalize_dataframe(raw, "20260101-120000")
    expected = -raw["long_g"].to_numpy()
    assert np.array_equal(out["long_g"].to_numpy(), expected)


# SPEC: normalize.normalize_dataframe — raw braking (raw>0) becomes negative; +long_g = accel
def test_normalize_long_g_braking_negative():
    raw = _raw_frame(long_g=0.9)  # raw braking convention -> canonical negative
    out, _ = normalize_dataframe(raw, "20260101-120000")
    assert (out["long_g"] < 0).all()  # negative long_g = braking


# ---------------------------------------------------------------------------
# normalize_dataframe — throttle_norm
# ---------------------------------------------------------------------------


# SPEC: normalize.normalize_dataframe — throttle_norm = throttle_raw / max, in [0,1], max -> 1.0
def test_normalize_throttle_norm_range_and_max():
    raw = _raw_frame(throttle_raw=np.linspace(0, 100, 20))
    out, _ = normalize_dataframe(raw, "20260101-120000")
    assert out["throttle_norm"].min() >= 0.0
    assert out["throttle_norm"].max() == pytest.approx(1.0, abs=0.001)


# SPEC: normalize.normalize_dataframe — throttle_norm value equals raw/max
def test_normalize_throttle_norm_ratio():
    raw = _raw_frame(throttle_raw=np.array([25.0, 50.0, 100.0] + [50.0] * 17))
    out, _ = normalize_dataframe(raw, "20260101-120000")
    assert out["throttle_norm"].iloc[0] == pytest.approx(0.25, abs=0.001)
    assert out["throttle_norm"].iloc[1] == pytest.approx(0.5, abs=0.001)
    assert out["throttle_norm"].iloc[2] == pytest.approx(1.0, abs=0.001)


# SPEC: normalize.normalize_dataframe — max(throttle_raw)==0 -> 0.0 everywhere (no div0)
def test_normalize_throttle_norm_all_zero():
    raw = _raw_frame(throttle_raw=0.0)
    out, _ = normalize_dataframe(raw, "20260101-120000")
    assert (out["throttle_norm"] == 0.0).all()
    assert not out["throttle_norm"].isna().any()


# ---------------------------------------------------------------------------
# normalize_dataframe — OBD forward/back fill
# ---------------------------------------------------------------------------


# SPEC: normalize.normalize_dataframe — OBD channels ffill/bfill -> no NaN given >=1 value
def test_normalize_obd_fill_no_nan():
    n = 20
    rpm = np.full(n, 4000.0)
    rpm[0] = np.nan      # leading NaN -> back-filled
    rpm[-1] = np.nan     # trailing NaN -> forward-filled
    coolant = np.full(n, 200.0)
    coolant[5] = np.nan  # interior NaN -> filled
    raw = _raw_frame(n=n, rpm=rpm, coolant_f=coolant)
    out, _ = normalize_dataframe(raw, "20260101-120000")
    for col in ("rpm", "speed_mph", "throttle_norm", "coolant_f", "iat_f"):
        assert not out[col].isna().any(), col


# ---------------------------------------------------------------------------
# normalize_dataframe — dist_m (trapezoidal integration)
# ---------------------------------------------------------------------------


# SPEC: normalize.normalize_dataframe — dist_m starts at 0
def test_normalize_dist_m_starts_zero():
    out, _ = normalize_dataframe(_raw_frame(), "20260101-120000")
    assert out["dist_m"].iloc[0] == pytest.approx(0.0, abs=0.01)


# SPEC: normalize.normalize_dataframe — dist_m monotonic non-decreasing for non-neg speeds
def test_normalize_dist_m_monotonic():
    raw = _raw_frame(speed_mph=np.abs(np.sin(np.linspace(0, 3, 20))) * 50.0)
    out, _ = normalize_dataframe(raw, "20260101-120000")
    diffs = np.diff(out["dist_m"].to_numpy())
    assert (diffs >= -1e-9).all()


# SPEC: normalize.normalize_dataframe — dist_m is trapezoidal integral of speed*0.44704 over t
def test_normalize_dist_m_constant_speed():
    # constant 60 mph for 1.9 s (20 samples @ 0.1 s) -> 60*0.44704*1.9 m at the end
    raw = _raw_frame(n=20, speed_mph=60.0)
    out, _ = normalize_dataframe(raw, "20260101-120000")
    expected_end = 60.0 * 0.44704 * (raw["t"].iloc[-1] - raw["t"].iloc[0])
    assert out["dist_m"].iloc[-1] == pytest.approx(expected_end, abs=0.01)


# SPEC: normalize.normalize_dataframe — single-row frame -> dist_m == 0 (no integration step)
def test_normalize_single_row_dist_zero():
    raw = _raw_frame(n=1, speed_mph=80.0)
    out, _ = normalize_dataframe(raw, "20260101-120000")
    assert len(out) == 1
    assert out["dist_m"].iloc[0] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# normalize_dataframe — dist_lap_m
# ---------------------------------------------------------------------------


# SPEC: normalize.normalize_dataframe — dist_lap_m starts at 0 at each lap's first sample
def test_normalize_dist_lap_m_starts_zero_per_lap():
    n = 40
    lap = np.array([1] * 20 + [2] * 20)
    raw = _raw_frame(n=n, lap=lap, speed_mph=60.0)
    out, _ = normalize_dataframe(raw, "20260101-120000")
    for lp in (1, 2):
        first = out[out["lap"] == lp]["dist_lap_m"].iloc[0]
        assert first == pytest.approx(0.0, abs=0.01)


# SPEC: normalize.normalize_dataframe — canonical_lap_length_m rescales each lap's max exactly
def test_normalize_dist_lap_m_rescaled():
    n = 40
    lap = np.array([1] * 20 + [2] * 20)
    raw = _raw_frame(n=n, lap=lap, speed_mph=60.0)
    out, _ = normalize_dataframe(raw, "20260101-120000", canonical_lap_length_m=3000.0)
    for lp in (1, 2):
        mx = out[out["lap"] == lp]["dist_lap_m"].max()
        assert mx == pytest.approx(3000.0, abs=0.01)


# SPEC: normalize.normalize_dataframe — with canonical_lap_length_m None, no rescale
def test_normalize_dist_lap_m_no_rescale_when_none():
    raw = _raw_frame(n=20, speed_mph=60.0)
    out_none, _ = normalize_dataframe(raw, "20260101-120000", canonical_lap_length_m=None)
    out_scaled, _ = normalize_dataframe(
        raw, "20260101-120000", canonical_lap_length_m=3000.0
    )
    # rescaling forces the lap's max to the canonical value; not rescaling does not.
    assert out_scaled["dist_lap_m"].max() == pytest.approx(3000.0, abs=0.01)
    assert out_none["dist_lap_m"].max() != pytest.approx(3000.0, abs=0.01)
    assert out_none["dist_lap_m"].iloc[0] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# normalize_dataframe — dtypes
# ---------------------------------------------------------------------------


# SPEC: normalize.normalize_dataframe — lap int32, brake int8, rpm int32
def test_normalize_dtypes():
    out, _ = normalize_dataframe(_raw_frame(), "20260101-120000")
    assert out["lap"].dtype == np.int32
    assert out["brake"].dtype == np.int8
    assert out["rpm"].dtype == np.int32


# ---------------------------------------------------------------------------
# compute_lap_times
# ---------------------------------------------------------------------------


# SPEC: normalize.compute_lap_times — columns are lap, lap_time_s
def test_compute_lap_times_columns():
    df = _raw_frame(n=30, lap=np.array([1] * 10 + [2] * 10 + [3] * 10))
    res = compute_lap_times(df)
    assert set(res.columns) >= {"lap", "lap_time_s"}


# SPEC: normalize.compute_lap_times — one row per distinct lap
def test_compute_lap_times_one_row_per_lap():
    df = _raw_frame(n=30, lap=np.array([1] * 10 + [2] * 10 + [3] * 10))
    res = compute_lap_times(df)
    assert sorted(res["lap"].tolist()) == [1, 2, 3]
    assert len(res) == 3


# SPEC: normalize.compute_lap_times — documented golden: starts t=0,100,205, last=305 -> 100,105,100
def test_compute_lap_times_golden():
    # three laps starting at t = 0, 100, 205; final sample at t = 305
    t = np.array([0.0, 50.0, 100.0, 150.0, 205.0, 305.0])
    lap = np.array([1, 1, 2, 2, 3, 3])
    df = pd.DataFrame({"t": t, "lap": lap})
    res = compute_lap_times(df).sort_values("lap").reset_index(drop=True)
    times = dict(zip(res["lap"], res["lap_time_s"]))
    assert times[1] == pytest.approx(100.0, abs=0.001)
    assert times[2] == pytest.approx(105.0, abs=0.001)
    assert times[3] == pytest.approx(100.0, abs=0.001)  # final lap = last_t - start_t


# SPEC: normalize.compute_lap_times — final lap uses end-of-data minus its start
def test_compute_lap_times_final_lap_is_inlap():
    t = np.array([0.0, 10.0, 20.0, 33.0])
    lap = np.array([1, 1, 2, 2])
    df = pd.DataFrame({"t": t, "lap": lap})
    res = compute_lap_times(df).sort_values("lap").reset_index(drop=True)
    times = dict(zip(res["lap"], res["lap_time_s"]))
    assert times[1] == pytest.approx(20.0, abs=0.001)  # next start - this start
    assert times[2] == pytest.approx(13.0, abs=0.001)  # 33 - 20 (incomplete in-lap)


# SPEC: normalize.compute_lap_times — rounded to 3 dp
def test_compute_lap_times_rounded_3dp():
    t = np.array([0.0, 1.23456, 2.5, 3.99999])
    lap = np.array([1, 1, 2, 2])
    df = pd.DataFrame({"t": t, "lap": lap})
    res = compute_lap_times(df)
    for v in res["lap_time_s"]:
        assert v == pytest.approx(round(v, 3), abs=1e-9)


# ---------------------------------------------------------------------------
# lap_summary
# ---------------------------------------------------------------------------


def _two_lap_normalized(n_per=60):
    """A normalized-shape frame with 3 laps so first/last cleanness is exercised."""
    n = n_per * 3
    t = np.arange(n) * 0.1
    lap = np.repeat([1, 2, 3], n_per)
    df = pd.DataFrame({
        "session_id": "20260101-120000",
        "t": t,
        "lap": lap.astype(np.int32),
        "dist_m": np.linspace(0, 6000, n),
        "dist_lap_m": np.tile(np.linspace(0, 2000, n_per), 3),
        "speed_mph": np.full(n, 60.0),
        "speed_mph_gps": np.full(n, 60.0),
        "throttle_norm": np.full(n, 0.5),
        "brake": np.zeros(n, dtype=np.int8),
        "rpm": np.full(n, 4000, dtype=np.int32),
        "lat_g": np.zeros(n),
        "long_g": np.zeros(n),
        "coolant_f": np.full(n, 200.0),
        "iat_f": np.full(n, 90.0),
        "lat": np.full(n, 45.0),
        "long": np.full(n, -122.0),
        "altitude_m": np.full(n, 100.0),
        "gps_accuracy_m": np.full(n, 3.0),
    })
    return df


# SPEC: normalize.lap_summary — one row per lap
def test_lap_summary_one_row_per_lap():
    df = _two_lap_normalized()
    lt = compute_lap_times(df)
    summ = lap_summary(df, lt)
    assert sorted(summ["lap"].tolist()) == [1, 2, 3]
    assert len(summ) == 3


# SPEC: normalize.lap_summary — is_clean is False only for first and last lap
def test_lap_summary_is_clean_shape_only():
    df = _two_lap_normalized()
    lt = compute_lap_times(df)
    summ = lap_summary(df, lt).sort_values("lap").reset_index(drop=True)
    clean = dict(zip(summ["lap"], summ["is_clean"]))
    assert not clean[1]  # warmup
    assert clean[2]      # interior
    assert not clean[3]  # cooldown


# SPEC: normalize.lap_summary — clean_reason warmup/cooldown/"" and is_clean==(reason=="")
def test_lap_summary_clean_reason():
    df = _two_lap_normalized()
    lt = compute_lap_times(df)
    summ = lap_summary(df, lt).sort_values("lap").reset_index(drop=True)
    reason = dict(zip(summ["lap"], summ["clean_reason"]))
    assert reason[1] == "warmup"
    assert reason[2] == ""
    assert reason[3] == "cooldown"
    for _, row in summ.iterrows():
        assert row["is_clean"] == (row["clean_reason"] == "")


# SPEC: normalize.lap_summary — pct_wot = fraction of samples with throttle_norm >= 0.95
def test_lap_summary_pct_wot():
    df = _two_lap_normalized(n_per=100)
    # make lap 2 exactly 40% WOT
    mask = df["lap"] == 2
    idx = df[mask].index
    thr = df["throttle_norm"].to_numpy().copy()
    thr[idx[:40]] = 1.0   # >= 0.95
    thr[idx[40:]] = 0.5   # below
    df["throttle_norm"] = thr
    lt = compute_lap_times(df)
    summ = lap_summary(df, lt)
    row2 = summ[summ["lap"] == 2].iloc[0]
    assert row2["pct_wot"] == pytest.approx(0.40, abs=0.01)


# SPEC: normalize.lap_summary — pct_braking = fraction with brake == 1
def test_lap_summary_pct_braking():
    df = _two_lap_normalized(n_per=100)
    mask = df["lap"] == 2
    idx = df[mask].index
    brake = df["brake"].to_numpy().copy()
    brake[idx[:25]] = 1  # 25% braking
    df["brake"] = brake.astype(np.int8)
    lt = compute_lap_times(df)
    summ = lap_summary(df, lt)
    row2 = summ[summ["lap"] == 2].iloc[0]
    assert row2["pct_braking"] == pytest.approx(0.25, abs=0.01)


# SPEC: normalize.lap_summary — max_decel_g = max(-long_g); max_accel_g = max(long_g)
def test_lap_summary_decel_accel():
    df = _two_lap_normalized(n_per=100)
    mask = df["lap"] == 2
    idx = df[mask].index
    lg = df["long_g"].to_numpy().copy()
    lg[idx[0]] = -1.2   # hard braking -> most-negative long_g
    lg[idx[1]] = 0.8    # accel
    df["long_g"] = lg
    lt = compute_lap_times(df)
    summ = lap_summary(df, lt)
    row2 = summ[summ["lap"] == 2].iloc[0]
    assert row2["max_decel_g"] == pytest.approx(1.2, abs=0.01)   # surfaced positive
    assert row2["max_accel_g"] == pytest.approx(0.8, abs=0.01)


# SPEC: normalize.lap_summary — lap_dist_m = max(dist_lap_m) for the lap
def test_lap_summary_lap_dist_m():
    df = _two_lap_normalized(n_per=100)
    lt = compute_lap_times(df)
    summ = lap_summary(df, lt)
    row2 = summ[summ["lap"] == 2].iloc[0]
    expected = df[df["lap"] == 2]["dist_lap_m"].max()
    assert row2["lap_dist_m"] == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# reference_session_ids
# ---------------------------------------------------------------------------


# SPEC: normalize.reference_session_ids — returns a set of reference-marked ids
def test_reference_session_ids_returns_set(sample_data_root):
    result = reference_session_ids("ridge")
    assert isinstance(result, set)


# SPEC: normalize.reference_session_ids — a session NOT marked reference is absent
def test_reference_session_ids_excludes_non_reference(sample_data_root):
    result = reference_session_ids("ridge")
    # 20260517-100304 is a normal session in the bundle — not marked reference
    assert "20260517-100304" not in result
    # excluded/flagged-only sessions are not "reference" either
    assert "20251017-092131" not in result   # exclude: wet
    assert "20250518-100457" not in result   # flag: gps_unreliable


# SPEC: normalize.reference_session_ids — missing notes file -> empty set
def test_reference_session_ids_missing_notes(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))  # no notes/<track>.json here
    assert reference_session_ids("ridge") == set()


# ---------------------------------------------------------------------------
# normalize_session — MissingOBDError path
# ---------------------------------------------------------------------------


def _no_obd_csv_columns(n=30):
    """The required GPS/accel/timing columns, but NO *OBD columns (per PIPELINE)."""
    return {
        "Time": np.arange(n) * 0.05,
        "UTC Time": ["2026-01-01 12:00:00"] * n,
        "Lap": np.ones(n, dtype=int),
        "Latitude": np.full(n, 45.0),
        "Longitude": np.full(n, -122.0),
        "Altitude (m)": np.full(n, 100.0),
        "Speed (MPH)": np.full(n, 60.0),
        "Accuracy (m)": np.full(n, 3.0),
        "Accel X": np.zeros(n),
        "Accel Y": np.zeros(n),
        "Accel Z": np.ones(n),
        "Brake (calculated)": np.zeros(n, dtype=int),
    }


# SPEC: normalize.normalize_session — CSV missing OBD columns raises MissingOBDError.
# DATA_ROOT is pointed at an empty tmp_path so the documented "writes no output"
# behavior cannot touch the real bundle; the CSV itself comes from the fixture.
def test_normalize_session_missing_obd_raises(make_trackaddict_csv, monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    csv = make_trackaddict_csv(_no_obd_csv_columns())
    with pytest.raises(MissingOBDError):
        normalize_session(csv, "ridge", tmp_path / "out")


# SPEC: normalize.normalize_session — no output is written when OBD is missing
def test_normalize_session_missing_obd_writes_nothing(make_trackaddict_csv, monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    csv = make_trackaddict_csv(_no_obd_csv_columns())
    with pytest.raises(MissingOBDError):
        normalize_session(csv, "ridge", data_root / "sessions" / "ridge")
    # no normalized session artifact should have been produced for this run
    assert not data_root.exists() or not any(data_root.rglob("*.parquet"))
