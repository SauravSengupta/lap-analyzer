"""Clean-room tests for the normalize CLI (lap_analyzer.cli.normalize.main).

Source of truth: tests/SPEC.md (## normalize CLI — status lines & exit codes) +
docs/PIPELINE.md (status-line vocabulary, exit semantics, column contract). NO
implementation under lap_analyzer/ was read while writing these.

We call main(argv: list[str]) and capture stdout with capsys. DATA_ROOT points at
a tmp_path; for --all, raw CSVs are placed under <DATA_ROOT>/raw/<track>/. Raw
TrackAddict column names come from PIPELINE.md's column-contract table (mirrored
in conftest.RAW_OBD_COLUMNS for the OBD subset).

Documented status-line prefixes: ok / skip / excl / noobd / FAIL.
Documented exit codes: valid -> 0, noobd -> 0 (not a failure), FAIL -> 2.

Coverage:
  - ok + exit 0 on a valid CSV
  - output files actually written for ok
  - noobd + exit 0 on a CSV missing OBD columns
  - FAIL + exit 2 on a malformed CSV (missing required GPS/timing columns)
  - skip on re-run without --force; ok again with --force
  - excl on a session listed with exclude in notes
  - final summary line reports counts
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.cli.normalize import main


# ---------------------------------------------------------------------------
# Raw TrackAddict CSV builders (from PIPELINE.md's column-contract table)
# ---------------------------------------------------------------------------

# Two GPS coordinates that straddle the End Point lap-cut so the session
# segments into >1 lap. Values are arbitrary plausible WGS84 around a track.
_END_POINT = (45.6200, -123.1800)


def _raw_columns(n=400, *, with_obd=True, with_gps=True):
    """A full raw TrackAddict column dict keyed by the RAW header names.

    The car loops past the End Point a few times so the session has several laps.
    OBD and GPS/timing column groups can be omitted to exercise noobd / FAIL.
    """
    t = np.arange(n) * 0.05  # ~20 Hz
    # TrackAddict's "UTC Time" column is a numeric Unix epoch (seconds), not an
    # ISO string — normalize does float(raw["utc"]) and datetime.fromtimestamp().
    utc0 = pd.Timestamp("2026-01-01T20:00:00Z").timestamp()
    # Drive lat/long in a repeating ramp that crosses the end point several times.
    cycles = 4
    phase = np.linspace(0, cycles * 2 * np.pi, n)
    lat = _END_POINT[0] + 0.004 * np.sin(phase)
    lon = _END_POINT[1] + 0.004 * np.cos(phase)
    lap_counter = (phase // (2 * np.pi)).astype(int)
    speed_gps = 60.0 + 20.0 * np.abs(np.sin(phase))

    cols: dict[str, np.ndarray] = {}
    if with_gps:
        cols.update({
            "Time": t,
            "UTC Time": utc0 + t,
            "Lap": lap_counter,
            "Sector": np.zeros(n, dtype=int),
            "Latitude": lat,
            "Longitude": lon,
            "Altitude (m)": np.full(n, 100.0),
            "Speed (MPH)": speed_gps,
            "Heading": (np.degrees(phase) % 360.0),
            "Accuracy (m)": np.full(n, 3.0),
            "Accel X": 0.1 * np.sin(phase),   # longitudinal on this phone
            "Accel Y": 0.5 * np.sin(phase),   # lateral on this phone
            "Accel Z": np.full(n, 1.0),
            "Brake (calculated)": (np.sin(phase) < -0.5).astype(int),
        })
    if with_obd:
        cols.update({
            "Engine Speed (RPM) *OBD": (3000 + 2000 * np.abs(np.sin(phase))).astype(int),
            "Vehicle Speed (mph) *OBD": speed_gps * 0.94,  # ~6% tire offset
            "Throttle Position (%) *OBD": 50.0 + 50.0 * np.abs(np.sin(phase)),
            "Engine Coolant Temp (F) *OBD": np.full(n, 200.0),
            "Intake Air Temp (F) *OBD": np.full(n, 90.0),
            "Intake Manifold Pressure (PSI) *OBD": np.full(n, 14.0),
        })
    return cols


def _write_raw_csv(path: Path, columns: dict, *, vehicle="Porsche 981 Cayman",
                   end_point=_END_POINT):
    """Write a TrackAddict-shaped CSV (comment header + data) at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns)
    header = (f"# Vehicle: {vehicle}\n"
              f"# End Point: {end_point[0]}, {end_point[1]} @ 2026-01-01 20:00:00\n")
    path.write_text(header + df.to_csv(index=False), encoding="utf-8")
    return path


def _raw_dir(data_root: Path, track="ridge") -> Path:
    d = data_root / "raw" / track
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# ok + exit 0
# ---------------------------------------------------------------------------

def test_valid_csv_prints_ok_and_returns_zero(tmp_path, monkeypatch, capsys):
    # SPEC: normalize CLI — normalizing a valid CSV prints a line starting "ok "
    #       and returns 0.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    csv = _write_raw_csv(raw / "Log-20260101-200000 Ridge.csv", _raw_columns())

    rc = main([str(csv), "--track", "ridge"])
    out = capsys.readouterr().out
    assert rc == 0
    assert any(line.startswith("ok ") for line in out.splitlines()), out


def test_valid_csv_writes_session_outputs(tmp_path, monkeypatch, capsys):
    # SPEC: PIPELINE — a normalized session writes samples.parquet, laps.csv,
    #       meta.json under data/sessions/<track>/<sid>/.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    _write_raw_csv(raw / "Log-20260101-200000 Ridge.csv", _raw_columns())

    rc = main(["--track", "ridge", "--all"])
    capsys.readouterr()
    assert rc == 0
    sid_dir = tmp_path / "sessions" / "ridge" / "20260101-200000"
    assert (sid_dir / "samples.parquet").exists()
    assert (sid_dir / "laps.csv").exists()
    assert (sid_dir / "meta.json").exists()


# ---------------------------------------------------------------------------
# noobd + exit 0
# ---------------------------------------------------------------------------

def test_missing_obd_prints_noobd_and_returns_zero(tmp_path, monkeypatch, capsys):
    # SPEC: normalize CLI — a CSV missing OBD columns prints "noobd " and the run
    #       still returns 0 (no OBD is not a failure).
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    csv = _write_raw_csv(raw / "Log-20260101-201000 NoOBD.csv",
                         _raw_columns(with_obd=False))

    rc = main([str(csv), "--track", "ridge"])
    out = capsys.readouterr().out
    assert rc == 0
    assert any(line.startswith("noobd ") for line in out.splitlines()), out


def test_missing_obd_writes_no_output(tmp_path, monkeypatch, capsys):
    # SPEC: PIPELINE / normalize.normalize_session — MissingOBDError path writes no
    #       output; the session is reported noobd and skipped.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    _write_raw_csv(raw / "Log-20260101-201000 NoOBD.csv", _raw_columns(with_obd=False))

    main(["--track", "ridge", "--all"])
    capsys.readouterr()
    sid_dir = tmp_path / "sessions" / "ridge" / "20260101-201000"
    assert not (sid_dir / "samples.parquet").exists()


# ---------------------------------------------------------------------------
# FAIL + exit 2
# ---------------------------------------------------------------------------

def test_malformed_csv_prints_fail_and_returns_two(tmp_path, monkeypatch, capsys):
    # SPEC: normalize CLI — a malformed/failing CSV prints "FAIL " and the run
    #       returns 2. Omit the required GPS/timing columns (but keep OBD) so the
    #       failure is NOT the noobd path.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    # Has OBD but no GPS/timing columns -> a non-noobd hard failure.
    bad_cols = _raw_columns(with_gps=False)  # only the 6 OBD columns present
    csv = _write_raw_csv(raw / "Log-20260101-202000 Bad.csv", bad_cols)

    rc = main([str(csv), "--track", "ridge"])
    out = capsys.readouterr().out
    assert rc == 2
    assert any(line.startswith("FAIL ") for line in out.splitlines()), out


# ---------------------------------------------------------------------------
# skip (no --force) / ok again (--force)
# ---------------------------------------------------------------------------

def test_rerun_without_force_prints_skip(tmp_path, monkeypatch, capsys):
    # SPEC: normalize CLI — re-running without --force on already-normalized output
    #       prints "skip ".
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    csv = _write_raw_csv(raw / "Log-20260101-200000 Ridge.csv", _raw_columns())

    rc1 = main([str(csv), "--track", "ridge"])
    capsys.readouterr()
    assert rc1 == 0

    rc2 = main([str(csv), "--track", "ridge"])
    out2 = capsys.readouterr().out
    assert rc2 == 0
    assert any(line.startswith("skip ") for line in out2.splitlines()), out2


def test_rerun_with_force_renormalizes_ok(tmp_path, monkeypatch, capsys):
    # SPEC: normalize CLI — --force re-normalizes even if outputs exist (prints ok).
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    csv = _write_raw_csv(raw / "Log-20260101-200000 Ridge.csv", _raw_columns())

    main([str(csv), "--track", "ridge"])
    capsys.readouterr()
    rc = main([str(csv), "--track", "ridge", "--force"])
    out = capsys.readouterr().out
    assert rc == 0
    assert any(line.startswith("ok ") for line in out.splitlines()), out


# ---------------------------------------------------------------------------
# excl (notes exclude)
# ---------------------------------------------------------------------------

def test_excluded_session_prints_excl(tmp_path, monkeypatch, capsys):
    # SPEC: normalize CLI — a session listed with "exclude" in notes prints "excl ".
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    # session id 20260101-203000 is excluded via notes.
    csv = _write_raw_csv(raw / "Log-20260101-203000 Wet.csv", _raw_columns())
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / "ridge.json").write_text(
        '{"20260101-203000": {"exclude": "wet"}}', encoding="utf-8")

    rc = main([str(csv), "--track", "ridge"])
    out = capsys.readouterr().out
    assert rc == 0
    assert any(line.startswith("excl ") for line in out.splitlines()), out


# ---------------------------------------------------------------------------
# summary line
# ---------------------------------------------------------------------------

def test_all_run_emits_summary_line(tmp_path, monkeypatch, capsys):
    # SPEC: normalize CLI — the final summary line reports the
    #       processed/skipped/excluded/no-OBD/failed counts.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    _write_raw_csv(raw / "Log-20260101-200000 Ok.csv", _raw_columns())
    _write_raw_csv(raw / "Log-20260101-201000 NoOBD.csv", _raw_columns(with_obd=False))

    rc = main(["--track", "ridge", "--all"])
    out = capsys.readouterr().out
    # noobd does not flip the run to failure.
    assert rc == 0
    # A per-CSV ok and a per-CSV noobd line both appear.
    lines = out.splitlines()
    assert any(l.startswith("ok ") for l in lines), out
    assert any(l.startswith("noobd ") for l in lines), out
    # Some trailing summary line carries count digits.
    assert any(any(ch.isdigit() for ch in l) for l in lines[-3:]), out


def test_all_run_with_failure_returns_two(tmp_path, monkeypatch, capsys):
    # SPEC: normalize CLI — when any CSV fails (FAIL), the --all run returns 2 even
    #       if other CSVs were ok/noobd.
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    raw = _raw_dir(tmp_path)
    _write_raw_csv(raw / "Log-20260101-200000 Ok.csv", _raw_columns())
    _write_raw_csv(raw / "Log-20260101-202000 Bad.csv", _raw_columns(with_gps=False))

    rc = main(["--track", "ridge", "--all"])
    out = capsys.readouterr().out
    assert rc == 2
    lines = out.splitlines()
    assert any(l.startswith("ok ") for l in lines), out
    assert any(l.startswith("FAIL ") for l in lines), out
