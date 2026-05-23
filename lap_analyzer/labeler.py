"""Corner labeling and per-corner-transit metrics.

Reads tracks/<track>.json + a normalized session, adds a `corner` column to
each sample, and produces a per-(lap × corner) transit table for clean laps.

Cross-lap position alignment uses a GPS-based reference index:
  - The track JSON nominates a reference lap (session_id + lap) as the
    canonical "ruler" for the track.
  - For each sample in any lap, we find the spatially nearest sample in the
    reference lap and use that reference sample's dist_lap_m as the sample's
    `track_dist_m`. This makes track_dist_m a stable cross-lap coordinate even
    when individual laps' OBD-integrated dist_lap_m has non-uniform error.
  - All corner ranges, anchor positions, etc. are interpreted in track_dist_m.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import sessions_dir, tracks_dir


THROTTLE_LIFT_THRESHOLD = 0.8
THROTTLE_RETURN_THRESHOLD = 0.8
THROTTLE_RETURN_SUSTAIN_S = 0.3
WOT_THRESHOLD = 0.95
LOOKBACK_M = 150.0

# Meters per degree latitude (kept as a constant — varies <1% globally and is
# below drift-correction noise). Longitude m-per-deg is lat-dependent and lives
# on ReferenceIndex (computed from each track's actual center latitude).
_M_PER_DEG_LAT = 111_132.0


@dataclass
class Corner:
    id: str
    name: str | None
    start_m: float
    end_m: float
    apex_m: float
    secondary_apex_m: float | None
    type: str
    notes: str | None


@dataclass
class CalibrationAnchor:
    corner_id: str
    ref_lat: float
    ref_long: float
    start_m: float  # filled in from corresponding Corner during load
    end_m: float


@dataclass
class Track:
    track_id: str
    name: str
    direction: str
    lap_length_m: float
    corners: list[Corner]
    calibration_anchors: list[CalibrationAnchor]
    reference_session_id: str
    reference_lap: int


def load_track(track_id: str) -> Track:
    path = tracks_dir() / f"{track_id}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    corners = [
        Corner(
            id=c["id"],
            name=c.get("name"),
            start_m=float(c["start_m"]),
            end_m=float(c["end_m"]),
            apex_m=float(c["apex_m"]),
            secondary_apex_m=float(c["secondary_apex_m"]) if c.get("secondary_apex_m") else None,
            type=c["type"],
            notes=c.get("notes"),
        )
        for c in raw["corners"]
    ]
    corners.sort(key=lambda c: c.start_m)
    corner_by_id = {c.id: c for c in corners}
    anchors = []
    for a in raw.get("calibration_anchors", []):
        c = corner_by_id.get(a["corner_id"])
        if c is None:
            continue
        anchors.append(CalibrationAnchor(
            corner_id=a["corner_id"],
            ref_lat=float(a["ref_lat"]),
            ref_long=float(a["ref_long"]),
            start_m=c.start_m,
            end_m=c.end_m,
        ))
    ref_lap_info = raw.get("reference_lap", {})
    return Track(
        track_id=raw["track_id"],
        name=raw["name"],
        direction=raw["direction"],
        lap_length_m=float(raw["lap_length_internal_m"]),
        corners=corners,
        calibration_anchors=anchors,
        reference_session_id=ref_lap_info.get("session_id", ""),
        reference_lap=int(ref_lap_info.get("lap", 0)),
    )


@dataclass
class ReferenceIndex:
    """KD-tree over the reference lap's GPS positions.

    Used to remap any sample's (lat, long) to a track_dist_m by finding the
    spatially nearest reference sample and using its dist_lap_m.
    """
    tree: cKDTree
    ref_dist: np.ndarray  # parallel to tree points; ref_dist[i] = dist_lap_m of ref sample i
    ref_lat: np.ndarray   # parallel to tree points; ref_lat[i] = lat of ref sample i
    ref_lon: np.ndarray   # parallel to tree points; ref_lon[i] = long of ref sample i
    center_lat: float
    center_lon: float
    m_per_deg_lat: float
    m_per_deg_lon: float

    def project(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Project (lat, lon) arrays into local meters relative to center."""
        x = (lon - self.center_lon) * self.m_per_deg_lon
        y = (lat - self.center_lat) * self.m_per_deg_lat
        return np.column_stack([x, y])

    def lookup(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each (lat, lon), return (track_dist_m, dist_to_ref_m)."""
        xy = self.project(lat, lon)
        dists, idxs = self.tree.query(xy, k=1)
        return self.ref_dist[idxs], dists

    def lookup_full(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """For each (lat, lon), return (track_dist_m, ref_lat, ref_lon, dist_to_ref_m)."""
        xy = self.project(lat, lon)
        dists, idxs = self.tree.query(xy, k=1)
        return self.ref_dist[idxs], self.ref_lat[idxs], self.ref_lon[idxs], dists


def build_reference_index(track: Track) -> ReferenceIndex:
    if not track.reference_session_id:
        raise ValueError(f"Track {track.track_id} has no reference_lap defined in JSON.")
    ref_dir = sessions_dir(track.track_id) / track.reference_session_id
    samples = pd.read_parquet(ref_dir / "samples.parquet")
    ref = samples[samples["lap"] == track.reference_lap]
    if len(ref) < 50:
        raise ValueError(f"Reference lap has only {len(ref)} samples; need a complete lap.")

    center_lat = float(ref["lat"].mean())
    center_lon = float(ref["long"].mean())
    m_per_deg_lat = _M_PER_DEG_LAT
    m_per_deg_lon = _M_PER_DEG_LAT * np.cos(np.radians(center_lat))
    x = (ref["long"].to_numpy() - center_lon) * m_per_deg_lon
    y = (ref["lat"].to_numpy() - center_lat) * m_per_deg_lat
    pts = np.column_stack([x, y])
    return ReferenceIndex(
        tree=cKDTree(pts),
        ref_dist=ref["dist_lap_m"].to_numpy(),
        ref_lat=ref["lat"].to_numpy(),
        ref_lon=ref["long"].to_numpy(),
        center_lat=center_lat,
        center_lon=center_lon,
        m_per_deg_lat=m_per_deg_lat,
        m_per_deg_lon=m_per_deg_lon,
    )


def label_samples(samples: pd.DataFrame, track: Track, ref: ReferenceIndex | None = None) -> pd.DataFrame:
    """Add `track_dist_m`, `track_dist_offset_m`, drift columns, and `corner`.

    Two-pass:
      Pass 1 (per lap): compute GPS drift by spatial-radius anchor search.
      Pass 2 (per sample): apply per-lap drift translation to GPS, then look up
        track_dist_m via kd-tree against the reference lap.
    """
    out = samples.copy()
    if ref is None:
        ref = build_reference_index(track)

    # Pass 1: per-lap drift estimate. ACTIVE correction is anchor-based (median of
    # 17 Maps-pin anchor offsets); centerline-based is computed alongside as a
    # passive sanity-check diagnostic. See lines 221-228 for the active assignment.
    drift_lat_cl: dict = {}
    drift_lon_cl: dict = {}
    drift_n_cl: dict = {}
    drift_region_disagree: dict = {}
    drift_n_filtered: dict = {}
    drift_lat_anchor: dict = {}
    drift_lon_anchor: dict = {}
    drift_n_anchor: dict = {}
    drift_anchor_disagree: dict = {}
    for lap_num in out["lap"].unique():
        lap_s = out[out["lap"] == lap_num]
        # Centerline-based (passive diagnostic)
        cl_lat, cl_lon, cl_n, cl_region, cl_filt = compute_lap_drift_centerline(lap_s, ref)
        drift_lat_cl[lap_num] = cl_lat
        drift_lon_cl[lap_num] = cl_lon
        drift_n_cl[lap_num] = cl_n
        drift_region_disagree[lap_num] = cl_region
        drift_n_filtered[lap_num] = cl_filt
        # Anchor-based (ACTIVE; written to gps_drift_{lat,lon}_m below)
        d_lat, d_lon, n, disagree = compute_lap_drift(
            lap_s, track.calibration_anchors, ref.m_per_deg_lat, ref.m_per_deg_lon
        )
        drift_lat_anchor[lap_num] = d_lat if d_lat is not None else 0.0
        drift_lon_anchor[lap_num] = d_lon if d_lon is not None else 0.0
        drift_n_anchor[lap_num] = n
        drift_anchor_disagree[lap_num] = disagree if disagree is not None else 0.0

    # Active correction: anchor-based (17 Maps-pin anchors, median aggregation).
    # Maps pins are survey-grade ground truth; median is robust to single-anchor glitches.
    out["gps_drift_lat_m"] = out["lap"].map(drift_lat_anchor).astype(float)
    out["gps_drift_lon_m"] = out["lap"].map(drift_lon_anchor).astype(float)
    out["gps_drift_n_anchors"] = out["lap"].map(drift_n_anchor).astype(int)
    out["gps_drift_disagreement_m"] = out["lap"].map(drift_anchor_disagree).astype(float)
    # Centerline-based estimate (passive; sanity-check diagnostic only).
    out["gps_drift_lat_m_centerline"] = out["lap"].map(drift_lat_cl).astype(float)
    out["gps_drift_lon_m_centerline"] = out["lap"].map(drift_lon_cl).astype(float)
    out["gps_drift_n_samples_centerline"] = out["lap"].map(drift_n_cl).astype(int)
    out["gps_drift_region_disagreement_m"] = out["lap"].map(drift_region_disagree).astype(float)
    out["gps_drift_outliers_filtered"] = out["lap"].map(drift_n_filtered).astype(int)
    lat_corr = out["lat"].to_numpy() - out["gps_drift_lat_m"].to_numpy() / ref.m_per_deg_lat
    lon_corr = out["long"].to_numpy() - out["gps_drift_lon_m"].to_numpy() / ref.m_per_deg_lon

    # Pass 2: kd-tree lookup using drift-corrected GPS
    track_dist, off = ref.lookup(lat_corr, lon_corr)
    out["track_dist_m"] = track_dist
    out["track_dist_offset_m"] = off

    # Corner labeling by track_dist_m
    segments: list[tuple[float, float, str]] = []
    prev_id = track.corners[-1].id  # for the front-straight wrap
    cursor = 0.0
    for c in track.corners:
        if c.start_m > cursor:
            segments.append((cursor, c.start_m, f"S_post_{prev_id}"))
        segments.append((c.start_m, c.end_m, c.id))
        prev_id = c.id
        cursor = c.end_m
    if cursor < track.lap_length_m:
        segments.append((cursor, track.lap_length_m, f"S_post_{prev_id}"))

    starts = np.array([s[0] for s in segments])
    labels = np.array([s[2] for s in segments], dtype=object)
    seg_idx = np.clip(np.searchsorted(starts, track_dist, side="right") - 1, 0, len(segments) - 1)
    out["corner"] = labels[seg_idx]
    return out


def _interp_at_dist(samples: pd.DataFrame, dist: float, col: str, dist_col: str = "track_dist_m") -> float | None:
    d = samples[dist_col].to_numpy()
    if len(d) == 0 or dist < d[0] or dist > d[-1]:
        return None
    return float(np.interp(dist, d, samples[col].to_numpy()))


def _first_sustained_above(window: pd.DataFrame, col: str, threshold: float, duration_s: float) -> float | None:
    """First track_dist_m where `col > threshold` is sustained for at least `duration_s`."""
    above = (window[col] > threshold).to_numpy()
    t = window["t"].to_numpy()
    d = window["track_dist_m"].to_numpy()
    n = len(window)
    i = 0
    while i < n:
        if above[i]:
            j = i + 1
            while j < n and above[j]:
                j += 1
            if t[j - 1] - t[i] >= duration_s:
                return float(d[i])
            i = j + 1
        else:
            i += 1
    return None


def build_corner_transit(lap_samples: pd.DataFrame, corner: Corner) -> dict | None:
    """Compute one transit row for a (lap × corner). Returns None if corner missing from lap.

    All distance fields are in track_dist_m (canonical, cross-lap-comparable).
    """
    in_corner = lap_samples[(lap_samples["track_dist_m"] >= corner.start_m) & (lap_samples["track_dist_m"] <= corner.end_m)]
    if len(in_corner) < 3:
        return None

    lookback = lap_samples[
        (lap_samples["track_dist_m"] >= corner.start_m - LOOKBACK_M)
        & (lap_samples["track_dist_m"] < corner.start_m)
    ]
    after = lap_samples[lap_samples["track_dist_m"] > corner.end_m].head(1)
    window = pd.concat([lookback, in_corner, after])

    entry_speed = _interp_at_dist(window, corner.start_m, "speed_mph")
    exit_speed = _interp_at_dist(window, corner.end_m, "speed_mph")
    entry_dist = float(in_corner["track_dist_m"].iloc[0])
    exit_dist = float(in_corner["track_dist_m"].iloc[-1])
    time_in = float(in_corner["t"].iloc[-1] - in_corner["t"].iloc[0])

    min_idx = in_corner["speed_mph"].idxmin()
    max_idx = in_corner["speed_mph"].idxmax()
    min_speed = float(in_corner.loc[min_idx, "speed_mph"])
    min_speed_dist = float(in_corner.loc[min_idx, "track_dist_m"])
    max_speed = float(in_corner.loc[max_idx, "speed_mph"])

    latg_peak_idx = in_corner["lat_g"].abs().idxmax()
    latg_peak_dist = float(in_corner.loc[latg_peak_idx, "track_dist_m"])
    latg_peak_value = float(in_corner.loc[latg_peak_idx, "lat_g"])

    apex_speed = _interp_at_dist(in_corner, corner.apex_m, "speed_mph")
    secondary_apex_speed = (
        _interp_at_dist(in_corner, corner.secondary_apex_m, "speed_mph")
        if corner.secondary_apex_m is not None else None
    )

    # Track-dist mapping quality within this corner: how far did the kd-tree have
    # to reach to map each sample? High values = samples are off the reference path,
    # so this transit's spatial analysis is unreliable.
    tdo_med = float(in_corner["track_dist_offset_m"].median())
    tdo_max = float(in_corner["track_dist_offset_m"].max())

    brake_in_window = window[window["brake"] == 1]
    brake_on = float(brake_in_window["track_dist_m"].iloc[0]) if len(brake_in_window) else None
    brake_off = float(brake_in_window["track_dist_m"].iloc[-1]) if len(brake_in_window) else None

    lift_in_window = window[window["throttle_norm"] < THROTTLE_LIFT_THRESHOLD]
    lift = float(lift_in_window["track_dist_m"].iloc[0]) if len(lift_in_window) else None

    post_apex = in_corner[in_corner["track_dist_m"] >= min_speed_dist]
    throttle_return = _first_sustained_above(post_apex, "throttle_norm", THROTTLE_RETURN_THRESHOLD, THROTTLE_RETURN_SUSTAIN_S)

    post_track_apex = in_corner[in_corner["track_dist_m"] >= corner.apex_m]
    wot_hits = post_track_apex[post_track_apex["throttle_norm"] >= WOT_THRESHOLD]
    wot = float(wot_hits["track_dist_m"].iloc[0]) if len(wot_hits) else None

    def offset(x):
        return None if x is None else round(x - corner.start_m, 2)

    return {
        "corner_id": corner.id,
        "entry_speed_mph": round(entry_speed, 2) if entry_speed is not None else None,
        "exit_speed_mph": round(exit_speed, 2) if exit_speed is not None else None,
        "entry_dist_m": round(entry_dist, 2),
        "exit_dist_m": round(exit_dist, 2),
        "time_in_corner_s": round(time_in, 3),
        "distance_in_corner_m": round(exit_dist - entry_dist, 2),
        "min_speed_mph": round(min_speed, 2),
        "min_speed_dist_m": round(min_speed_dist, 2),
        "max_speed_mph": round(max_speed, 2),
        "max_lat_g": round(float(in_corner["lat_g"].abs().max()), 3),
        "max_accel_g": round(float(in_corner["long_g"].max()), 3),
        "max_decel_g": round(float(in_corner["long_g"].min()), 3),
        "peak_brake": int(in_corner["brake"].max()),
        "peak_throttle_norm": round(float(in_corner["throttle_norm"].max()), 3),
        "apex_speed_mph": round(apex_speed, 2) if apex_speed is not None else None,
        "apex_dist_offset_m": round(min_speed_dist - corner.apex_m, 2),
        "latg_peak_dist_m": round(latg_peak_dist, 2),
        "latg_peak_offset_m": round(latg_peak_dist - corner.apex_m, 2),
        "latg_peak_g": round(latg_peak_value, 3),
        "secondary_apex_speed_mph": round(secondary_apex_speed, 2) if secondary_apex_speed is not None else None,
        "throttle_lift_dist_m": offset(lift),
        "brake_on_dist_m": offset(brake_on),
        "brake_off_dist_m": offset(brake_off),
        "throttle_return_dist_m": offset(throttle_return),
        "wot_dist_m": offset(wot),
        "mean_throttle_norm": round(float(in_corner["throttle_norm"].mean()), 3),
        "pct_wot": round(float((in_corner["throttle_norm"] >= WOT_THRESHOLD).mean()), 3),
        "mean_lat_g": round(float(in_corner["lat_g"].abs().mean()), 3),
        "pct_braking": round(float((in_corner["brake"] == 1).mean()), 3),
        "track_dist_offset_med_m": round(tdo_med, 2),
        "track_dist_offset_max_m": round(tdo_max, 2),
        "sample_idx_start": int(in_corner.index[0]),
        "sample_idx_end": int(in_corner.index[-1]),
    }


def compute_lap_drift_centerline(
    lap_samples: pd.DataFrame,
    ref: ReferenceIndex,
    mad_threshold: float = 3.0,
    n_regions: int = 4,
) -> tuple[float, float, int, float, int]:
    """Per-lap GPS drift via aggregation against the synthetic centerline.

    For each sample in the lap, find the nearest centerline point and compute
    the (lat, lon) offset. Apply a MAD-based outlier filter, then take the
    median across remaining samples.

    Steps:
      1. KD-tree project each sample → matched centerline (lat, lon, track_dist_m).
      2. Per-sample offset = sample_latlon - matched_latlon (in meters).
      3. MAD filter: drop samples whose offset magnitude exceeds mad_threshold * MAD.
      4. Median of remaining samples is the drift.
      5. Compute per-region (lap split into n_regions by track_dist_m) median offsets;
         spread of region medians around the lap drift is the "region disagreement"
         diagnostic — replaces the old anchor-disagreement signal.

    Returns (drift_lat_m, drift_lon_m, n_used, region_disagreement_m, n_filtered_outliers).
    """
    sample_lat = lap_samples["lat"].to_numpy()
    sample_lon = lap_samples["long"].to_numpy()
    if len(sample_lat) == 0:
        return 0.0, 0.0, 0, 0.0, 0

    track_dist, ref_lat, ref_lon, _ = ref.lookup_full(sample_lat, sample_lon)
    off_lat_m = (sample_lat - ref_lat) * ref.m_per_deg_lat
    off_lon_m = (sample_lon - ref_lon) * ref.m_per_deg_lon

    # MAD-based outlier filter
    med_lat = float(np.median(off_lat_m))
    med_lon = float(np.median(off_lon_m))
    abs_dev = np.sqrt((off_lat_m - med_lat) ** 2 + (off_lon_m - med_lon) ** 2)
    mad = float(np.median(abs_dev))
    threshold = mad * mad_threshold if mad > 0 else float("inf")
    keep = abs_dev <= threshold
    n_filtered = int((~keep).sum())

    drift_lat_m = float(np.median(off_lat_m[keep])) if keep.any() else 0.0
    drift_lon_m = float(np.median(off_lon_m[keep])) if keep.any() else 0.0
    n_used = int(keep.sum())

    # Per-region disagreement: split by track_dist_m, compute per-region median offset,
    # measure spread of region medians vs the lap-wide drift.
    region_disagree = 0.0
    if n_used >= 100:
        td_kept = track_dist[keep]
        ol_kept = off_lat_m[keep]
        on_kept = off_lon_m[keep]
        edges = np.linspace(td_kept.min(), td_kept.max(), n_regions + 1)
        region_lat_offs, region_lon_offs = [], []
        for i in range(n_regions):
            mask = (td_kept >= edges[i]) & (td_kept < edges[i + 1])
            if mask.sum() < 10:
                continue
            region_lat_offs.append(float(np.median(ol_kept[mask])))
            region_lon_offs.append(float(np.median(on_kept[mask])))
        if len(region_lat_offs) >= 2:
            rl = np.array(region_lat_offs)
            rn = np.array(region_lon_offs)
            region_disagree = float(np.sqrt(
                np.mean((rl - drift_lat_m) ** 2) + np.mean((rn - drift_lon_m) ** 2)
            ))

    return drift_lat_m, drift_lon_m, n_used, region_disagree, n_filtered


def compute_lap_drift(
    lap_samples: pd.DataFrame,
    anchors: list[CalibrationAnchor],
    m_per_deg_lat: float,
    m_per_deg_lon: float,
) -> tuple[float | None, float | None, int, float | None]:
    """For one lap, find closest sample to each anchor's ref position within the anchor's
    corner range, take the median of the offsets, return (drift_lat_m, drift_lon_m,
    n_anchors_used, disagreement_m).

    Aggregation is median (not mean) — robust to one or two anchors landing on glitched
    GPS samples (e.g. inner-loop glitch hits a specific corner on a specific lap).
    `disagreement_m` is the median absolute deviation of per-anchor offsets from the
    lap's median drift, scaled to RMS units. Low (~3-6m) = consistent drift, trustworthy.
    High (~10m+) = anchors disagree, drift estimate unreliable.

    `m_per_deg_lat` and `m_per_deg_lon` are the lat/lon-to-meters conversions for the
    track's center; pass `ref.m_per_deg_lat` / `ref.m_per_deg_lon` from a ReferenceIndex.
    """
    # Spatial search: find the closest sample in the lap to each anchor's GPS reference,
    # within a generous radius. Independent of any dist coordinate.
    SEARCH_RADIUS_M = 100.0
    lat_offs, lon_offs = [], []
    for a in anchors:
        dlat_m = (lap_samples["lat"] - a.ref_lat) * m_per_deg_lat
        dlon_m = (lap_samples["long"] - a.ref_long) * m_per_deg_lon
        d2 = dlat_m.values ** 2 + dlon_m.values ** 2
        if d2.min() > SEARCH_RADIUS_M ** 2:
            continue  # anchor's region not found within the lap
        i = int(np.argmin(d2))
        lat_offs.append(float(dlat_m.iloc[i]))
        lon_offs.append(float(dlon_m.iloc[i]))
    n = len(lat_offs)
    if n == 0:
        return None, None, 0, None
    lat_arr = np.array(lat_offs)
    lon_arr = np.array(lon_offs)
    median_lat = float(np.median(lat_arr))
    median_lon = float(np.median(lon_arr))
    if n == 1:
        return median_lat, median_lon, 1, 0.0
    # Disagreement: per-anchor distance from the lap's median drift, RMS-aggregated.
    devs_lat = lat_arr - median_lat
    devs_lon = lon_arr - median_lon
    disagreement = float(np.sqrt(np.mean(devs_lat ** 2 + devs_lon ** 2)))
    return median_lat, median_lon, n, disagreement


def build_session_corners(
    samples: pd.DataFrame,
    laps: pd.DataFrame,
    track: Track,
    session_id: str,
    date: str,
) -> pd.DataFrame:
    """One row per (clean lap × corner)."""
    rows: list[dict] = []
    clean_laps = laps[laps["is_clean"].astype(bool)]
    for _, lap in clean_laps.iterrows():
        lap_num = int(lap["lap"])
        lap_samples = samples[samples["lap"] == lap_num]
        if len(lap_samples) < 50:
            continue
        # Drift columns were already computed in label_samples and written to samples.parquet.
        # Pull the per-lap values from any sample in this lap.
        first = lap_samples.iloc[0]
        drift_lat = float(first["gps_drift_lat_m"])
        drift_lon = float(first["gps_drift_lon_m"])
        n_anchors = int(first["gps_drift_n_anchors"])
        disagreement = float(first["gps_drift_disagreement_m"])
        for corner in track.corners:
            transit = build_corner_transit(lap_samples, corner)
            if transit is None:
                continue
            transit["session_id"] = session_id
            transit["date"] = date
            transit["lap"] = lap_num
            transit["gps_drift_lat_m"] = round(drift_lat, 3)
            transit["gps_drift_lon_m"] = round(drift_lon, 3)
            transit["gps_drift_n_anchors"] = n_anchors
            transit["gps_drift_disagreement_m"] = round(disagreement, 3)
            rows.append(transit)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    head = ["session_id", "date", "lap", "corner_id"]
    return df[head + [c for c in df.columns if c not in head]]


def label_session(session_dir: Path, track: Track, ref: ReferenceIndex | None = None) -> tuple[int, int]:
    """Process one session: rewrite samples.parquet with `track_dist_m` + `corner`,
    write corners.parquet.

    Returns (n_samples_labeled, n_transits).
    """
    samples = pd.read_parquet(session_dir / "samples.parquet")
    laps = pd.read_csv(session_dir / "laps.csv")
    session_id = session_dir.name
    date = f"{session_id[:4]}-{session_id[4:6]}-{session_id[6:8]}"

    if ref is None:
        ref = build_reference_index(track)

    labeled = label_samples(samples, track, ref=ref)
    tmp = session_dir / "samples.parquet.tmp"
    labeled.to_parquet(tmp, index=False)
    tmp.replace(session_dir / "samples.parquet")

    transits = build_session_corners(labeled, laps, track, session_id, date)
    transits.to_parquet(session_dir / "corners.parquet", index=False)

    return len(labeled), len(transits)
