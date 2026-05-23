"""Shared fixtures for the lap_analyzer test suite.

These are test INFRASTRUCTURE (not clean-room test logic): a way to point the
package at the committed sample bundle, to synthesize TrackAddict CSVs from the
documented column contract, and to build one-lap sample frames for pure-function
tests. Clean-room test authors consume these but must still derive expected
BEHAVIOR from tests/SPEC.md, never from observed sample values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_ROOT = REPO_ROOT / "data" / "samples"

# Raw TrackAddict column names (the contract documented in docs/PIPELINE.md).
RAW_OBD_COLUMNS = [
    "Engine Speed (RPM) *OBD",
    "Vehicle Speed (mph) *OBD",
    "Throttle Position (%) *OBD",
    "Engine Coolant Temp (F) *OBD",
    "Intake Air Temp (F) *OBD",
    "Intake Manifold Pressure (PSI) *OBD",
]


@pytest.fixture
def sample_data_root(monkeypatch):
    """Point lap_analyzer.config.data_root() at the committed sample bundle.

    config.data_root() reads os.environ['DATA_ROOT'] on every call, so setting
    the env var is enough — no module reload needed. An absolute path wins over
    the APP_ROOT join.
    """
    monkeypatch.setenv("DATA_ROOT", str(SAMPLES_ROOT))
    return SAMPLES_ROOT


@pytest.fixture
def make_trackaddict_csv(tmp_path):
    """Factory: write a minimal TrackAddict CSV from a dict of raw columns.

    `columns` maps RAW TrackAddict column names -> array-likes (equal length).
    Omit the OBD columns to exercise the MissingOBDError path. The filename
    matches the Log-YYYYMMDD-HHMMSS pattern normalize expects.
    """
    def _make(columns: dict, *, session_id="20260101-120000",
              vehicle="Porsche 981 Cayman", end_point=(45.0, -122.0),
              filename=None):
        df = pd.DataFrame(columns)
        path = tmp_path / (filename or f"Log-{session_id} Test Session.csv")
        header = (f"# Vehicle: {vehicle}\n"
                  f"# End Point: {end_point[0]}, {end_point[1]} @ 2026-01-01 12:00:00\n")
        path.write_text(header + df.to_csv(index=False), encoding="utf-8")
        return path
    return _make


@pytest.fixture
def make_lap_samples():
    """Factory: a one-lap, normalized+labeled samples DataFrame with overridable channels.

    Defaults describe a plausible ~10 Hz flying lap. Pass length-n arrays or
    scalars to override any channel; scalars are broadcast to length n.
    """
    def _make(n=200, **overrides):
        t = np.arange(n) * 0.1
        base = {
            "session_id": "20260101-120000",
            "t": t,
            "lap": 1,
            "dist_m": np.linspace(0, 2000, n),
            "dist_lap_m": np.linspace(0, 2000, n),
            "track_dist_m": np.linspace(0, 2000, n),
            "track_dist_offset_m": 1.0,
            "speed_mph": 80.0,
            "speed_mph_gps": 80.0,
            "throttle_norm": 1.0,
            "brake": 0,
            "rpm": 4000,
            "lat_g": 0.0,
            "long_g": 0.0,
            "coolant_f": 200.0,
            "iat_f": 90.0,
            "lat": 45.0,
            "long": -122.0,
            "altitude_m": 100.0,
            "gps_accuracy_m": 3.0,
        }
        base.update(overrides)
        data = {k: (np.asarray(v) if np.ndim(v) else np.full(n, v)) for k, v in base.items()}
        return pd.DataFrame(data)
    return _make
