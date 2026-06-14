"""Per-transit quality metrics.

Computes corpus-wide stats and joins back into each session's corners.parquet:

- `lap_pace_decile`: 0-9, where 0 = fastest 10% of clean laps for the track.
- `latg_peak_offset_z`: per-corner z-score of latg_peak_offset_m. Outliers indicate
  unusual line position relative to the corpus median for that corner.
- `entry_speed_z`: per-corner z-score of entry_speed_mph. Strong negative values
  flag suspiciously slow corner entries (traffic, off, recovery from prior corner).
- `gps_drift_mag_m`: convenience = sqrt(gps_drift_lat_m^2 + gps_drift_lon_m^2).
- `neighborhood_offset_max_m`: max of this transit's track_dist_offset_max_m and
  its two corner-sequence neighbors. Catches glitch-in-adjacent-corner spillover.
- `transit_reliable`: passes the STANDARD reliability filter (drift disagreement
  <= 20m, transit offset <= 40m, neighborhood offset <= 40m).
- `lap_reliable`: every corner on this lap is `transit_reliable`. Use this when
  cross-corner consistency matters (e.g., time-delta across the lap).

Filters are signals for the analysis layer to apply (or override with custom
thresholds via the raw metric columns), not exclusions baked into the data.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import sessions_dir, tracks_dir
from .normalize import reference_session_ids


# STANDARD reliability tier — see notebooks/reliability_tiers.py for the analysis
# that produced these thresholds.
DRIFT_DISAGREEMENT_LIMIT_M = 20.0
TRANSIT_OFFSET_LIMIT_M = 40.0
NEIGHBORHOOD_OFFSET_LIMIT_M = 40.0

# Columns this module adds to each transit row. Dropped from the input before
# recomputation so the step is idempotent: re-running flag_quality on an already
# flagged corners.parquet otherwise makes the merge below suffix the existing
# lap_pace_decile to lap_pace_decile_x/_y, dropping the documented column name.
_COMPUTED_COLUMNS = [
    "lap_pace_decile", "latg_peak_offset_z", "entry_speed_z", "gps_drift_mag_m",
    "neighborhood_offset_max_m", "transit_reliable", "lap_reliable",
]


def compute_quality(track: str) -> pd.DataFrame:
    """Read all session corners.parquet + laps.csv, compute per-transit quality columns,
    return a DataFrame keyed by (session_id, lap, corner_id)."""
    root = sessions_dir(track)
    # Reference sessions are fully labeled but excluded from every corpus-wide
    # stat here (pace deciles, per-corner z-score medians).
    ref = reference_session_ids(track)
    transits = pd.concat(
        [pd.read_parquet(p) for p in sorted(root.rglob("corners.parquet"))
         if p.parent.name not in ref],
        ignore_index=True,
    )
    laps = pd.concat(
        [pd.read_csv(p) for p in sorted(root.rglob("laps.csv"))
         if p.parent.name not in ref],
        ignore_index=True,
    )

    # Idempotency: strip any columns we are about to recompute, so re-running on
    # already-flagged corners.parquet doesn't collide on the merge below.
    transits = transits.drop(columns=[c for c in _COMPUTED_COLUMNS if c in transits.columns])

    clean = laps[laps["is_clean"].astype(bool)][["session_id", "lap", "lap_time_s"]].copy()
    clean["lap_pace_decile"] = pd.qcut(
        clean["lap_time_s"], 10, labels=False, duplicates="drop"
    ).astype("int8")

    out = transits.merge(clean[["session_id", "lap", "lap_pace_decile"]],
                         on=["session_id", "lap"], how="left")

    # Per-corner z-scores using corpus median + MAD-derived scale (more robust to outliers
    # than mean/std). `baseline` selects which rows DEFINE the per-corner median/MAD;
    # all rows are still SCORED against it. Speed metrics use an OBD-only baseline so
    # GPS-only sessions (GPS speed reads ~1-2 mph low) don't shift the references.
    def add_corner_z(col, name, baseline=None):
        base = out if baseline is None else out[baseline]
        med_by_corner = base.groupby("corner_id")[col].median()
        dev = (base[col] - base["corner_id"].map(med_by_corner)).abs()
        mad_by_corner = dev.groupby(base["corner_id"]).median() * 1.4826  # ≈ std
        med = out["corner_id"].map(med_by_corner)
        mad = out["corner_id"].map(mad_by_corner)
        mad_safe = mad.where(mad > 0.1, np.nan)  # avoid divide-by-zero on degenerate corners
        out[name] = ((out[col] - med) / mad_safe).round(2)

    # lat-G is present regardless of OBD -> all sessions define its baseline.
    add_corner_z("latg_peak_offset_m", "latg_peak_offset_z")
    # entry_speed baseline = OBD sessions only (default True if the column is absent,
    # e.g. a corpus labeled before obd_present existed).
    obd_mask = out["obd_present"] if "obd_present" in out.columns else pd.Series(True, index=out.index)
    add_corner_z("entry_speed_mph", "entry_speed_z", baseline=obd_mask.astype(bool))

    out["gps_drift_mag_m"] = np.sqrt(
        out["gps_drift_lat_m"] ** 2 + out["gps_drift_lon_m"] ** 2
    ).round(2)

    # --- Neighborhood offset + STANDARD reliability flags ---
    # Build corner-sequence neighbors from the track def. Wraps around at start/finish.
    track_path = tracks_dir() / f"{track}.json"
    track_def = json.loads(track_path.read_text(encoding="utf-8"))
    corner_order = [c["id"] for c in sorted(track_def["corners"], key=lambda c: c["start_m"])]
    n = len(corner_order)
    prev_neighbor = {corner_order[i]: corner_order[(i - 1) % n] for i in range(n)}
    next_neighbor = {corner_order[i]: corner_order[(i + 1) % n] for i in range(n)}

    offset_idx = out.set_index(["session_id", "lap", "corner_id"])["track_dist_offset_max_m"].to_dict()

    def neighborhood_offset(row) -> float:
        own = row["track_dist_offset_max_m"]
        prev_off = offset_idx.get((row["session_id"], int(row["lap"]), prev_neighbor[row["corner_id"]]), np.nan)
        next_off = offset_idx.get((row["session_id"], int(row["lap"]), next_neighbor[row["corner_id"]]), np.nan)
        vals = [v for v in (own, prev_off, next_off) if not pd.isna(v)]
        return max(vals) if vals else np.nan

    out["neighborhood_offset_max_m"] = out.apply(neighborhood_offset, axis=1).round(2)

    out["transit_reliable"] = (
        (out["gps_drift_disagreement_m"] <= DRIFT_DISAGREEMENT_LIMIT_M)
        & (out["track_dist_offset_max_m"] <= TRANSIT_OFFSET_LIMIT_M)
        & (out["neighborhood_offset_max_m"] <= NEIGHBORHOOD_OFFSET_LIMIT_M)
    )
    # lap_reliable: every transit on this (session, lap) is transit_reliable
    out["lap_reliable"] = out.groupby(["session_id", "lap"])["transit_reliable"].transform("min").astype(bool)

    return out


def write_quality(track: str) -> dict:
    """Update each session's corners.parquet with quality columns. Returns counts dict."""
    out = compute_quality(track)
    written = 0
    for session_id, group in out.groupby("session_id"):
        path = sessions_dir(track) / session_id / "corners.parquet"
        if not path.exists():
            continue
        # Replace the whole file with the joined version
        group.to_parquet(path, index=False)
        written += 1
    return {"sessions_updated": written, "n_transits": len(out)}
