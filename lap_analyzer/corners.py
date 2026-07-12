from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


SMOOTH_SECONDS = 0.3
MIN_PEAK_SPACING_S = 3.0
PEAK_HEIGHT_G = 0.4
BOUNDARY_THRESHOLD_G = 0.2
THROTTLE_LIFT_THRESHOLD = 0.8
THROTTLE_RETURN_THRESHOLD = 0.8
BRAKE_LONG_G_THRESHOLD = -0.3
# Braking / throttle-lift for a corner routinely begins before the lat-G boundary
# (turn-in). Look back this far (in dist_lap_m) from the candidate's entry when
# detecting those onsets, so the reported offsets capture pre-turn-in braking —
# matching labeler.build_corner_transit's LOOKBACK_M. Offsets stay relative to
# entry, so an onset before entry is a negative offset.
LOOKBACK_M = 150.0


def _first_dist_where(df: pd.DataFrame, mask: pd.Series) -> float | None:
    hits = df.index[mask]
    if len(hits) == 0:
        return None
    return float(df.loc[hits[0], "dist_lap_m"])


def extract_lap_candidates(lap_df: pd.DataFrame, sample_rate_hz: float) -> list[dict]:
    lap_df = lap_df.reset_index(drop=True)
    if len(lap_df) < 50:
        return []

    # GPS-only (OBD-dropout) sessions have all-NaN speed_mph; fall back to GPS
    # speed for the min-speed apex pick so idxmin() has real values to sort,
    # mirroring labeler.build_corner_transit (~12% of sessions log without OBD).
    obd_present = bool(lap_df["speed_mph"].notna().any())
    speed_col = "speed_mph" if obd_present else "speed_mph_gps"

    window = max(3, int(round(SMOOTH_SECONDS * sample_rate_hz)))
    distance = int(round(MIN_PEAK_SPACING_S * sample_rate_hz))

    lat_g = lap_df["lat_g"].to_numpy()
    lat_g_smooth = pd.Series(lat_g).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    abs_smooth = np.abs(lat_g_smooth)

    peaks, _ = find_peaks(abs_smooth, height=PEAK_HEIGHT_G, distance=distance)

    rows = []
    for i, peak_idx in enumerate(peaks, start=1):
        entry_idx = peak_idx
        while entry_idx > 0 and abs_smooth[entry_idx] >= BOUNDARY_THRESHOLD_G:
            entry_idx -= 1
        exit_idx = peak_idx
        while exit_idx < len(abs_smooth) - 1 and abs_smooth[exit_idx] >= BOUNDARY_THRESHOLD_G:
            exit_idx += 1

        window_df = lap_df.iloc[entry_idx:exit_idx + 1]
        min_idx = window_df[speed_col].idxmin()

        entry_dist = float(lap_df.iloc[entry_idx]["dist_lap_m"])
        post_apex = window_df.loc[min_idx:]

        # Approach window: entry minus LOOKBACK_M through the corner exit, so the
        # brake / lift onsets that precede turn-in are seen (see LOOKBACK_M).
        exit_dist = float(window_df.iloc[-1]["dist_lap_m"])
        approach = lap_df[
            (lap_df["dist_lap_m"] >= entry_dist - LOOKBACK_M)
            & (lap_df["dist_lap_m"] <= exit_dist)
        ]

        brake_dist = _first_dist_where(approach, approach["long_g"] < BRAKE_LONG_G_THRESHOLD)
        lift_dist = _first_dist_where(approach, approach["throttle_norm"] < THROTTLE_LIFT_THRESHOLD)
        ret_dist = _first_dist_where(post_apex, post_apex["throttle_norm"] > THROTTLE_RETURN_THRESHOLD)

        rows.append({
            "candidate_idx": i,
            "peak_dist_m": round(float(lap_df.iloc[peak_idx]["dist_lap_m"]), 2),
            "peak_lat_g": round(float(lat_g_smooth[peak_idx]), 3),
            "direction": "right" if lat_g_smooth[peak_idx] > 0 else "left",
            "entry_dist_m": round(entry_dist, 2),
            "exit_dist_m": round(float(lap_df.iloc[exit_idx]["dist_lap_m"]), 2),
            "duration_s": round(float(lap_df.iloc[exit_idx]["t"] - lap_df.iloc[entry_idx]["t"]), 3),
            "min_speed_mph": round(float(lap_df.loc[min_idx, speed_col]), 2),
            "min_speed_dist_m": round(float(lap_df.loc[min_idx, "dist_lap_m"]), 2),
            "brake_on_offset_m": round(brake_dist - entry_dist, 2) if brake_dist is not None else None,
            "throttle_lift_offset_m": round(lift_dist - entry_dist, 2) if lift_dist is not None else None,
            "throttle_return_offset_m": round(ret_dist - entry_dist, 2) if ret_dist is not None else None,
            "apex_lat": round(float(lap_df.loc[min_idx, "lat"]), 7),
            "apex_long": round(float(lap_df.loc[min_idx, "long"]), 7),
            # Provenance for min_speed_mph (mirrors labeler.build_corner_transit):
            # obd_present=False means the speeds above came from GPS, not OBD.
            "obd_present": obd_present,
            "speed_source": "obd" if obd_present else "gps",
        })
    return rows


CANDIDATE_COLUMNS = [
    "session_id", "date", "lap", "lap_time_s", "candidate_idx",
    "peak_dist_m", "peak_lat_g", "direction",
    "entry_dist_m", "exit_dist_m", "duration_s",
    "min_speed_mph", "min_speed_dist_m",
    "brake_on_offset_m", "throttle_lift_offset_m", "throttle_return_offset_m",
    "apex_lat", "apex_long",
    "obd_present", "speed_source",
]


def extract_session_candidates(session_dir: Path) -> pd.DataFrame:
    session_id = session_dir.name
    samples = pd.read_parquet(session_dir / "samples.parquet")
    laps = pd.read_csv(session_dir / "laps.csv")

    duration = samples["t"].iloc[-1] - samples["t"].iloc[0]
    sample_rate_hz = (len(samples) - 1) / duration if duration > 0 else 20.0

    flying_laps = laps[~laps["clean_reason"].isin(["warmup", "cooldown"])]
    date = f"{session_id[:4]}-{session_id[4:6]}-{session_id[6:8]}"

    rows = []
    for _, lap_row in flying_laps.iterrows():
        lap_num = int(lap_row["lap"])
        lap_samples = samples[samples["lap"] == lap_num]
        for c in extract_lap_candidates(lap_samples, sample_rate_hz):
            c["session_id"] = session_id
            c["date"] = date
            c["lap"] = lap_num
            c["lap_time_s"] = float(lap_row["lap_time_s"])
            rows.append(c)

    if not rows:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    return pd.DataFrame(rows)[CANDIDATE_COLUMNS]
