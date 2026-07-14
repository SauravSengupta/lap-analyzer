"""Shared cached loaders and helpers for the visualizer pages."""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from lap_analyzer.analysis import lap_index, load_corpus, load_samples
from lap_analyzer.config import tracks_dir


# Single source of truth for the section-timing cache version. Both app.py and
# every page that computes section times (via section_times / span_time /
# range_section_times) pass it EXPLICITLY as version= into their @st.cache_data
# functions — st.cache_data keys on the function's own source + PASSED args, not
# its callees, so a change in gates.py / analysis.py only reaches every cache when
# this constant bumps AND is passed. It must be a plain (non-underscore) name and
# actually passed at the call site: Streamlit drops underscore-prefixed args and
# unpassed defaults from the key (cache_utils.py), so a `_version=` default is a
# no-op. Lives here (not app.py) so pages import it without executing app.py.
# v10 (2026-07-11, gps-trust PR 0): build_gate now uses a σ=10m smoothed tangent,
# which shifts gate-crossing section times at rotation-exposed corners.
# v11 (2026-07-12, gps-trust PR 2): section_times / span_time / range_section_times
# now read the trajectory layer (estimate_trajectory + section_timing) — value AND
# confidence from one pass, always-emit rows, new sigma_t_s/status/rank_eligible/
# checks_json columns; timing_reliable is a compat alias of rank_eligible.
_SECTION_TIMES_VERSION = 11


def current_track() -> str:
    """The track the visualizer is currently showing.

    Priority: explicit user pick in st.session_state -> LAP_ANALYZER_TRACK env var -> 'ridge'.
    The sidebar picker in app.py writes to st.session_state['track'].
    """
    return st.session_state.get("track") or os.environ.get("LAP_ANALYZER_TRACK", "ridge")


def available_tracks() -> list[str]:
    """List of track slugs found under tracks/.

    Excludes dotfiles and stems containing a dot (e.g. `ridge.pre-apex-pin-update.json`
    is a backup, not a real track). Clean track slugs are simple identifiers.
    """
    return sorted(
        p.stem for p in tracks_dir().glob("*.json")
        if not p.stem.startswith(".") and "." not in p.stem
    )


@st.cache_data(show_spinner=False)
def corpus(track: str) -> pd.DataFrame:
    return load_corpus(track)


@st.cache_data(show_spinner=False)
def laps(track: str) -> pd.DataFrame:
    return lap_index(corpus(track), track=track)


@st.cache_data(show_spinner=False)
def track_def(track: str) -> dict:
    return json.loads((tracks_dir() / f"{track}.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner="loading samples")
def samples(track: str, session_id: str, lap: int) -> pd.DataFrame:
    """One lap's samples (long_g is already canonical: + = accel, − = brake).

    `rpm` is included so the downshift page can derive gear; app.py ignores it.
    `track_dist_m`/`dist_lap_m` + `lat_g`/`speed_mph_gps` + the drift-correction
    columns feed the trajectory estimator (estimate_trajectory), which the
    visualizer runs directly for the along-track axis and σ shading. The drift
    columns are load-bearing: estimate_trajectory drift-corrects lat/long (design
    R5) only when they are present, so omitting them here would make the plotted
    s_hat/σ disagree with the section-times table (both come from the same
    estimator, and analysis loads these columns via _read_traj_samples).
    `lat`/`long` are the raw GPS path — every gate-crossing consumer of this
    loader (analysis.span_time on the downshift page, app.py's section-time
    tooltip) reads them, so they must be here or those callers KeyError.
    """
    df = load_samples(track, session_id, lap)
    cols = ["t", "lap", "track_dist_m", "dist_lap_m", "speed_mph", "speed_mph_gps",
            "throttle_norm", "long_g", "lat_g", "rpm", "lat", "long"]
    s = df[cols].copy()
    # Drift-correction offsets (design R5). Present on every real session; default
    # to 0.0 (no correction) when a session lacks them, matching _read_traj_samples.
    for c in ("gps_drift_lat_m", "gps_drift_lon_m"):
        s[c] = df[c].to_numpy() if c in df.columns else 0.0
    return s


def session_hhmm(sid: str) -> str:
    return f"{sid[9:11]}:{sid[11:13]}"


def format_lap_time(s: float) -> str:
    """Lap time as M:SS.SS (e.g. 132.62s -> '2:12.62')."""
    m, rem = divmod(s, 60)
    return f"{int(m)}:{rem:05.2f}"
