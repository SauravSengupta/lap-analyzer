"""Clean-room tests for lap_analyzer.centerline.

Source of truth: tests/SPEC.md (## centerline.build_centerline,
## centerline.install_centerline) + docs/ARCHITECTURE.md §3 (synthetic
centerline / lap-wrap refinement) + docs/PIPELINE.md. NO implementation under
lap_analyzer/ was read while writing these.

build_centerline is INTEGRATION tier and heavy: asserting exact coordinates is
out of scope. We assert OUTPUT SCHEMA, the protected-range / monotone-ruler
invariants, and DETERMINISM. install_centerline is exercised against a tmp
synth_dir using a real committed centerline parquet as input.

Coverage (build_centerline):
  - output schema: columns track_dist_m, lat, long, n, spread_m
  - ~lap_length_m rows (1 m grid)
  - track_dist_m increasing and within [0, lap_length_m]
  - determinism (same inputs -> identical output)

Coverage (install_centerline):
  - writes samples.parquet / laps.csv / meta.json into synth_dir
  - samples.parquet has normalized-session columns; lat/long from centerline
  - gps_accuracy_m == 0.0 sentinel; unused kinematic channels are NaN
  - laps.csv: single lap 1, is_clean == True
  - meta.json: synthetic == true + source path recorded
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.centerline import build_centerline, install_centerline

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_ROOT = REPO_ROOT / "data" / "samples"

CENTERLINE_COLUMNS = ["track_dist_m", "lat", "long", "n", "spread_m"]


# ---------------------------------------------------------------------------
# build_centerline — schema + ruler invariants + determinism
# ---------------------------------------------------------------------------

@pytest.fixture
def copied_ridge_root(tmp_path, monkeypatch):
    """Copy the committed Ridge sessions + notes tree into a tmp DATA_ROOT.

    build_centerline aggregates clean drift-corrected laps; running it against a
    copy keeps the committed sample corpus untouched.
    """
    shutil.copytree(SAMPLES_ROOT / "sessions" / "ridge", tmp_path / "sessions" / "ridge")
    if (SAMPLES_ROOT / "notes").exists():
        shutil.copytree(SAMPLES_ROOT / "notes", tmp_path / "notes")
    # Copy the existing corpus so any centerline-adjacent reads resolve.
    shutil.copytree(SAMPLES_ROOT / "corpus", tmp_path / "corpus")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    return tmp_path


def _build_ridge_centerline(root, out_name="ridge_centerline_test.parquet"):
    """Call build_centerline with its documented signature against the copied
    Ridge tree, returning the output DataFrame.

    Signature: build_centerline(sessions_dir, notes_path, lap_length_m,
    track_lat_deg, out_path, protected_ranges=None). lap_length / center latitude
    / corner protected-ranges come from tracks/ridge.json.
    """
    track_json = json.loads((REPO_ROOT / "tracks" / "ridge.json").read_text(encoding="utf-8"))
    protected = [(float(c["start_m"]), float(c["end_m"])) for c in track_json["corners"]]
    return build_centerline(
        sessions_dir=root / "sessions" / "ridge",
        notes_path=root / "notes" / "ridge.json",
        lap_length_m=float(track_json["lap_length_internal_m"]),
        track_lat_deg=float(track_json["start_finish"]["lat"]),
        out_path=root / "corpus" / out_name,
        protected_ranges=protected,
    )


def test_build_centerline_schema(copied_ridge_root):
    # SPEC: centerline.build_centerline — output columns track_dist_m, lat, long,
    #       n, spread_m (one row per 1 m grid point).
    df = _build_ridge_centerline(copied_ridge_root)
    for col in CENTERLINE_COLUMNS:
        assert col in df.columns


def test_build_centerline_track_dist_increasing_and_in_range(copied_ridge_root):
    # SPEC: centerline.build_centerline — track_dist_m is increasing and within
    #       [0, lap_length_m].
    df = _build_ridge_centerline(copied_ridge_root).sort_index()
    td = df["track_dist_m"].to_numpy()
    assert np.all(np.diff(td) > 0)  # strictly increasing 1 m grid
    assert td.min() >= 0.0
    # Roughly lap_length rows at 1 m spacing -> span ~= row count.
    assert td.max() <= len(df) + 1.0


def test_build_centerline_roughly_one_row_per_meter(copied_ridge_root):
    # SPEC: centerline.build_centerline — ~lap_length_m rows on a 1 m grid; grid
    #       spacing is ~1 m so consecutive track_dist_m differ by ~1.
    df = _build_ridge_centerline(copied_ridge_root).sort_values("track_dist_m")
    steps = np.diff(df["track_dist_m"].to_numpy())
    assert np.median(steps) == pytest.approx(1.0, abs=0.5)


def test_build_centerline_protected_corner_bins_present(copied_ridge_root):
    # SPEC: ARCHITECTURE §3 — bins inside a corner box are never rejected; the
    #       grid covers the full lap so every protected (corner) range has rows.
    track_json = json.loads((REPO_ROOT / "tracks" / "ridge.json").read_text(encoding="utf-8"))
    df = _build_ridge_centerline(copied_ridge_root)
    td = df["track_dist_m"].to_numpy()
    for c in track_json["corners"]:
        in_corner = (td >= float(c["start_m"])) & (td <= float(c["end_m"]))
        assert in_corner.sum() > 0, f"corner {c['id']} range has no centerline rows"


def test_build_centerline_is_deterministic(copied_ridge_root):
    # SPEC: ARCHITECTURE design principle 1 — deterministic; same inputs ->
    #       identical centerline.
    a = _build_ridge_centerline(copied_ridge_root, "a.parquet").reset_index(drop=True)
    b = _build_ridge_centerline(copied_ridge_root, "b.parquet").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# install_centerline — session-shaped output
# ---------------------------------------------------------------------------

NORMALIZED_SAMPLE_COLUMNS = [
    "session_id", "t", "lap", "dist_m", "dist_lap_m", "speed_mph",
    "speed_mph_gps", "throttle_norm", "brake", "rpm", "lat_g", "long_g",
    "coolant_f", "iat_f", "lat", "long", "altitude_m", "gps_accuracy_m",
]


@pytest.fixture
def installed_synth(tmp_path):
    """Run install_centerline into a tmp synth_dir from the committed Ridge
    centerline parquet. Returns (synth_dir, centerline_path)."""
    centerline_path = SAMPLES_ROOT / "corpus" / "ridge_centerline.parquet"
    synth_dir = tmp_path / "_synthetic_centerline"
    install_centerline("ridge", centerline_path, synth_dir)
    return synth_dir, centerline_path


def test_install_writes_three_session_files(installed_synth):
    # SPEC: centerline.install_centerline — writes samples.parquet, laps.csv,
    #       meta.json into synth_dir, shaped like a normalized session.
    synth_dir, _ = installed_synth
    assert (synth_dir / "samples.parquet").exists()
    assert (synth_dir / "laps.csv").exists()
    assert (synth_dir / "meta.json").exists()


def test_install_samples_have_normalized_columns(installed_synth):
    # SPEC: centerline.install_centerline — samples.parquet has the
    #       normalized-session columns.
    synth_dir, _ = installed_synth
    s = pd.read_parquet(synth_dir / "samples.parquet")
    for col in NORMALIZED_SAMPLE_COLUMNS:
        assert col in s.columns


def test_install_latlong_come_from_centerline(installed_synth):
    # SPEC: centerline.install_centerline — lat/long come from the centerline.
    synth_dir, centerline_path = installed_synth
    s = pd.read_parquet(synth_dir / "samples.parquet")
    cl = pd.read_parquet(centerline_path)
    # One sample per grid point: row count matches the centerline grid.
    assert len(s) == len(cl)
    # The lat/long sets equal the centerline's (order-aligned along the polyline).
    assert np.allclose(np.sort(s["lat"].to_numpy()),
                       np.sort(cl["lat"].to_numpy()), atol=1e-9)
    assert np.allclose(np.sort(s["long"].to_numpy()),
                       np.sort(cl["long"].to_numpy()), atol=1e-9)


def test_install_gps_accuracy_is_perfect_sentinel(installed_synth):
    # SPEC: centerline.install_centerline — gps_accuracy_m is 0.0 (perfect-accuracy
    #       sentinel).
    synth_dir, _ = installed_synth
    s = pd.read_parquet(synth_dir / "samples.parquet")
    assert (s["gps_accuracy_m"].to_numpy() == 0.0).all()


def test_install_unused_channels_are_nan(installed_synth):
    # SPEC: centerline.install_centerline — unused channels (speed_mph,
    #       throttle_norm, ...) are NaN sentinels.
    synth_dir, _ = installed_synth
    s = pd.read_parquet(synth_dir / "samples.parquet")
    for col in ("speed_mph", "throttle_norm"):
        assert s[col].isna().all(), f"{col} should be NaN sentinel"


def test_install_laps_single_clean_lap(installed_synth):
    # SPEC: centerline.install_centerline — laps.csv has a single lap 1 marked
    #       is_clean == True.
    synth_dir, _ = installed_synth
    laps = pd.read_csv(synth_dir / "laps.csv")
    assert len(laps) == 1
    assert int(laps.iloc[0]["lap"]) == 1
    assert bool(laps.iloc[0]["is_clean"]) is True


def test_install_meta_records_synthetic_and_source(installed_synth):
    # SPEC: centerline.install_centerline — meta.json records synthetic: true and
    #       the source path.
    synth_dir, centerline_path = installed_synth
    meta = json.loads((synth_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("synthetic") is True
    # The source path references the centerline parquet that was installed.
    source = str(meta.get("source", ""))
    assert source, "meta.json must record a source path"
    assert "centerline" in source.lower()
