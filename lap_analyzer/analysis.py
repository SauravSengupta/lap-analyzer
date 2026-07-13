"""Component 5 — analysis library.

Pure functions on the corpus DataFrame. Kept minimal; extended as the
visualizer needs new operations.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import corpus_dir, sessions_dir
from .fused_axis import GLITCH_OFFSET_M
from .gates import TrackFrame


def load_corpus(track: str) -> pd.DataFrame:
    """Read the cross-session corners corpus for a track."""
    return pd.read_parquet(corpus_dir() / f"{track}_corners.parquet")


def load_centerline(track: str) -> pd.DataFrame:
    """Read the synthetic centerline (track_dist_m -> lat/long ruler) for a track."""
    return pd.read_parquet(corpus_dir() / f"{track}_centerline.parquet")


def reliable_transits(corpus: pd.DataFrame) -> pd.DataFrame:
    """Rows passing the STANDARD spatial-reliability filter. Use for line / apex / map views."""
    return corpus[corpus["transit_reliable"]].copy()


def top_decile_laps(corpus: pd.DataFrame) -> set[tuple[str, int]]:
    """(session_id, lap) pairs for the fastest 10% of clean laps (lap_pace_decile == 0)."""
    fast = corpus[corpus["lap_pace_decile"] == 0][["session_id", "lap"]].drop_duplicates()
    return {(r.session_id, int(r.lap)) for r in fast.itertuples(index=False)}


def lap_index(corpus: pd.DataFrame, track: str | None = None) -> pd.DataFrame:
    """One row per (session_id, lap) with corpus lap-level fields, optionally joined
    with lap_time_s + is_clean from each session's laps.csv when `track` is given.
    """
    cols = ["session_id", "date", "lap", "lap_pace_decile", "lap_reliable"]
    avail = [c for c in cols if c in corpus.columns]
    idx = (corpus[avail].drop_duplicates(subset=["session_id", "lap"])
           .sort_values(["date", "session_id", "lap"], na_position="last")
           .reset_index(drop=True))
    if track is not None:
        laps = load_all_laps(track)[["session_id", "lap", "lap_time_s", "is_clean"]]
        idx = idx.merge(laps, on=["session_id", "lap"], how="left")
    return idx


def load_all_laps(track: str) -> pd.DataFrame:
    """Concat every session's laps.csv for a track. Lap-time and is_clean live here."""
    root = sessions_dir(track)
    return pd.concat(
        [pd.read_csv(p) for p in sorted(root.rglob("laps.csv"))],
        ignore_index=True,
    )


def lap_summary(corpus: pd.DataFrame, session_id: str, lap: int) -> dict:
    """Pocket summary of one lap for headers/tooltips."""
    rows = corpus[(corpus["session_id"] == session_id) & (corpus["lap"] == lap)]
    if rows.empty:
        return {}
    r0 = rows.iloc[0]
    return {
        "session_id": session_id,
        "date": r0.get("date"),
        "lap": int(lap),
        "pace_decile": int(r0["lap_pace_decile"]) if pd.notna(r0["lap_pace_decile"]) else None,
        "lap_reliable": bool(r0["lap_reliable"]),
        "n_transits": len(rows),
        "n_reliable_transits": int(rows["transit_reliable"].sum()),
        "max_speed_mph": float(rows["max_speed_mph"].max()),
    }


def load_samples(track: str, session_id: str, lap: int | None = None) -> pd.DataFrame:
    """Read a session's samples.parquet, optionally filtered to one lap.

    Channel traces (speed/throttle/brake/lat-G) live here, not in the corpus.
    """
    df = pd.read_parquet(sessions_dir(track) / session_id / "samples.parquet")
    if lap is not None:
        df = df[df["lap"] == lap].copy()
    return df


def section_bounds(track_def: dict, pre_m: float = 50.0, post_cap_m: float = 250.0) -> dict[str, tuple[float, float]]:
    """Per-corner section window for ranking: braking zone -> corner -> exit straight.

    end = min(next_corner.start_m, end_m + post_cap_m); start = max(0, start_m - pre_m).
    Defined so 'fastest through this corner' can't be won by a lap with poor exit speed.
    """
    corners = sorted(track_def["corners"], key=lambda c: c["start_m"])
    lap_len = float(track_def.get("lap_length_internal_m", corners[-1]["end_m"] + 200))
    out: dict[str, tuple[float, float]] = {}
    for i, c in enumerate(corners):
        next_start = corners[i + 1]["start_m"] if (i + 1) < len(corners) else lap_len
        out[c["id"]] = (
            max(0.0, c["start_m"] - pre_m),
            float(min(next_start, c["end_m"] + post_cap_m)),
        )
    return out


def _first_crossing_t(xs: np.ndarray, ts: np.ndarray, target: float, after_t: float | None = None) -> float | None:
    """Linear-interp time at which xs first crosses target (going up). xs, ts must be sorted by time."""
    crossings = (xs[:-1] < target) & (xs[1:] >= target)
    idxs = np.where(crossings)[0]
    if len(idxs) == 0:
        return None
    if after_t is not None:
        idxs = idxs[ts[idxs] >= after_t]
        if len(idxs) == 0:
            return None
    i = int(idxs[0])
    dx = xs[i + 1] - xs[i]
    if dx == 0:
        return float(ts[i])
    return float(ts[i] + (target - xs[i]) / dx * (ts[i + 1] - ts[i]))


# A section time is only trustworthy where GPS sampled finely enough to time the
# gate crossings. crossing_gap_s measures that: the elapsed time between the good
# GPS fixes bracketing a gate — the window in which the car physically crossed but
# we have no trustworthy fix. Wide gap = coarse GPS (Mode 3) or a teleport-punctured
# bracket (Mode 2). A transit is reliable when its max gate gap is below
# CONFIDENCE_GAP_S. See docs/GPS_TRUST.md.
CONFIDENCE_GAP_S: float = 0.4


def crossing_gap_s(
    t, track_dist_m, dist_lap_m, dist: float, glitch_m: float = GLITCH_OFFSET_M,
) -> float:
    """Elapsed seconds between the good GPS fixes bracketing centerline `dist`.

    A good fix is a *fresh* sample (track_dist_m changed from the previous one,
    not a frozen repeat) that is not a *teleport* (|track_dist_m - dist_lap_m| <
    glitch_m). Returns inf if `dist` is not bracketed by good fixes below and
    above. Large gap = the gate crossing time cannot be trusted.
    """
    t = np.asarray(t, dtype=float)
    td = np.asarray(track_dist_m, dtype=float)
    dl = np.asarray(dist_lap_m, dtype=float)
    if len(td) < 2:
        return float("inf")
    fresh = np.ones(len(td), dtype=bool)
    fresh[1:] = np.abs(np.diff(td)) > 0.01
    good = fresh & (np.abs(td - dl) < glitch_m)
    gi = np.where(good)[0]
    if len(gi) < 2:
        return float("inf")
    gtd = td[gi]
    below = np.where(gtd <= dist)[0]
    above = np.where(gtd > dist)[0]
    if len(below) == 0 or len(above) == 0:
        return float("inf")
    i_lo = gi[below[-1]]
    i_hi = gi[above[0]]
    return abs(float(t[i_hi]) - float(t[i_lo]))


# The trajectory layer needs these per-sample channels; missing ones default so
# OBD-dropout / drift-less sessions still estimate (design R5/R6). session_id, lap
# come from the row context.
_TRAJ_SAMPLE_COLS = ["lap", "t", "track_dist_m", "dist_lap_m", "lat", "long", "lat_g",
                     "speed_mph", "speed_mph_gps", "gps_drift_lat_m", "gps_drift_lon_m"]
_SECTION_TIME_COLS = ["session_id", "lap", "corner_id", "section_time_s", "sigma_t_s",
                      "status", "driven_m", "rank_eligible", "timing_gap_s",
                      "timing_reliable", "checks_json"]
_RANGE_TIME_COLS = [c for c in _SECTION_TIME_COLS if c != "corner_id"]


def _read_traj_samples(sp) -> "pd.DataFrame | None":
    """A session's samples with every channel the trajectory estimator needs.
    None when the parquet lacks the core columns or has no GPS at all."""
    try:
        import pyarrow.parquet as pq
        have = set(pq.read_schema(sp).names)
        if not {"lap", "t", "track_dist_m", "dist_lap_m", "lat", "long"} <= have:
            return None
        s = pd.read_parquet(sp, columns=[c for c in _TRAJ_SAMPLE_COLS if c in have])
    except Exception:
        return None
    if s["track_dist_m"].isna().all():
        return None
    for c in _TRAJ_SAMPLE_COLS:
        if c not in s.columns:
            if c.startswith("gps_drift") or c == "lat_g":
                s[c] = 0.0
            elif c == "speed_mph_gps":
                s[c] = s.get("speed_mph", np.nan)
            elif c == "speed_mph":
                s[c] = np.nan
    return s.sort_values(["lap", "t"])


def _load_or_build_corridor(track: str):
    """The per-track corridor, built on the fly (unsaved) when none is persisted
    (e.g. a fresh DATA_ROOT / the CI sample bundle)."""
    from .trajectory import build_corridor, load_corridor
    try:
        return load_corridor(track)
    except (FileNotFoundError, ValueError, OSError):
        return build_corridor(track, save=False)


def _corpus_trajectories(track: str):
    """Yield (session_id, lap, lap_df, corridor, Trajectory|None) for every lap in
    the corpus — corridor + centerline/frame built once, one trajectory per lap
    (reused across all corners). Trajectory is None only for a degenerate lap."""
    from .trajectory import estimate_trajectory
    centerline = load_centerline(track)
    frame = TrackFrame.from_centerline(centerline)
    corridor = _load_or_build_corridor(track)
    root = sessions_dir(track)
    for sid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sp = sid_dir / "samples.parquet"
        if not sp.exists():
            continue
        s = _read_traj_samples(sp)
        if s is None:
            continue
        for lap_n, g in s.groupby("lap"):
            traj = None
            if len(g) >= 2:
                try:
                    traj = estimate_trajectory(g, corridor, frame)
                except Exception:
                    traj = None
            yield sid_dir.name, int(lap_n), g, corridor, traj


def _section_row(traj, g, corridor, a: float, b: float):
    """(section_time_s, sigma_t_s, status, driven_m, rank_eligible, timing_gap_s,
    checks_json) for one (lap, section) — ALWAYS a row (design R10)."""
    from .trajectory import section_timing
    tt = g["t"].to_numpy()
    td = g["track_dist_m"].to_numpy()
    dl = g["dist_lap_m"].to_numpy()
    gap = max(crossing_gap_s(tt, td, dl, a), crossing_gap_s(tt, td, dl, b))
    if traj is None:
        return (np.nan, np.nan, "no_coverage", np.nan, False, gap, "[]")
    st = section_timing(traj, a, b, corridor)
    return (st.time_s, st.sigma_s, st.status, st.driven_m, bool(st.rank_eligible),
            gap, json.dumps(st.checks))


def section_times(
    track: str,
    track_def: dict,
    pre_m: float = 50.0,
    post_cap_m: float = 250.0,
) -> pd.DataFrame:
    """For every (session_id, lap, corner_id), the section time on the trajectory layer.

    A thin corpus loop over `estimate_trajectory` (once per lap) + `section_timing`
    (per corner). Value AND confidence come from the same evidence pass (design R1);
    a row is ALWAYS emitted (R10) with `status ∈ {ok, no_coverage, rescale_invalid}`
    and NaN value where uncomputable. Long-form columns:
    session_id, lap, corner_id, section_time_s, sigma_t_s, status, driven_m,
    rank_eligible, timing_gap_s (compat GPS-fix gap), timing_reliable (compat alias
    of rank_eligible), checks_json (structured audit records, R12).
    """
    bounds = section_bounds(track_def, pre_m=pre_m, post_cap_m=post_cap_m)
    rows: list[tuple] = []
    for sid, lap_n, g, corridor, traj in _corpus_trajectories(track):
        for cid, (a, b) in bounds.items():
            val, sig, status, driven, rank, gap, checks = _section_row(traj, g, corridor, a, b)
            rows.append((sid, lap_n, cid, val, sig, status, driven, rank, gap, rank, checks))
    return pd.DataFrame(rows, columns=_SECTION_TIME_COLS)


# --- gear derivation --------------------------------------------------------

def find_gear_bands(
    ratios: np.ndarray,
    ratio_lo: float = 30.0,
    ratio_hi: float = 260.0,
    n_bins: int = 230,
    min_peak_frac: float = 0.005,
    min_separation: float = 8.0,
) -> np.ndarray:
    """Cluster pooled rpm/speed ratios into gear band centers.

    Histograms the ratios, takes strict local maxima above `min_peak_frac` of
    the tallest bin, merges maxima closer than `min_separation` ratio units
    (keeping the first), and refines each surviving center as the mean ratio of
    samples within +/- min_separation/2. Returns sorted band centers (ascending
    ratio; smallest ratio = tallest gear).
    """
    r = ratios[(ratios >= ratio_lo) & (ratios <= ratio_hi)]
    if r.size == 0:
        return np.array([])
    hist, edges = np.histogram(r, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    thresh = min_peak_frac * hist.max()
    peak = np.zeros(n_bins, dtype=bool)
    peak[1:-1] = (
        (hist[1:-1] >= hist[:-2])
        & (hist[1:-1] > hist[2:])
        & (hist[1:-1] >= thresh)
    )
    raw = centers[peak]
    merged: list[float] = []
    for c in raw:
        if merged and (c - merged[-1]) < min_separation:
            continue
        merged.append(float(c))
    refined: list[float] = []
    for c in merged:
        near = r[np.abs(r - c) < min_separation / 2]
        refined.append(float(near.mean()) if near.size else c)
    return np.array(sorted(refined))


def gear_bands(track: str) -> np.ndarray:
    """Corpus-wide gear band centers for a track (IO wrapper over find_gear_bands)."""
    root = sessions_dir(track)
    pooled: list[np.ndarray] = []
    for sp in sorted(root.rglob("samples.parquet")):
        try:
            df = pd.read_parquet(sp, columns=["rpm", "speed_mph"])
        except Exception:
            continue
        # Exclude idle / slow-speed creep where the rpm/speed ratio is meaningless.
        m = (df["rpm"] > 1200) & (df["speed_mph"] > 8)
        if m.any():
            pooled.append((df["rpm"][m] / df["speed_mph"][m]).to_numpy())
    if not pooled:
        return np.array([])
    return find_gear_bands(np.concatenate(pooled))


def _confirm_runs(raw: np.ndarray, n_dwell: int) -> np.ndarray:
    """Set non-NaN values in any run shorter than n_dwell to NaN.

    A run is a maximal contiguous stretch of equal values, or of NaN. NaN runs
    are never modified; only non-NaN runs shorter than n_dwell are suppressed.
    Used to drop brief wrong-gear snaps while the clutch slips during a shift.
    """
    out = raw.astype(float).copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and (
            (out[j] == out[i]) or (np.isnan(out[j]) and np.isnan(out[i]))
        ):
            j += 1
        if not np.isnan(out[i]) and (j - i) < n_dwell:
            out[i:j] = np.nan
        i = j
    return out


def derive_gear(
    samples: pd.DataFrame, bands: np.ndarray, min_dwell_s: float = 0.3
) -> pd.Series:
    """Per-sample gear index derived from the rpm/speed ratio.

    Snaps each qualifying sample (rpm > 1200, speed_mph > 8) to the nearest
    band, drops sub-dwell runs (clutch-slip transients) and forward-fills the
    held gear across them, then re-NaNs samples with no OBD (rpm == 0).
    Gear index 0 = shortest gear (highest ratio); higher index = taller gear.
    Returned Series is aligned to `samples.index`.
    """
    s = samples.sort_values("t")
    rpm = s["rpm"].to_numpy(dtype=float)
    spd = s["speed_mph"].to_numpy(dtype=float)
    valid = (rpm > 1200) & (spd > 8)
    raw = np.full(len(s), np.nan)
    if bands.size and valid.any():
        rv = rpm[valid] / spd[valid]
        nearest = np.argmin(np.abs(rv[:, None] - bands[None, :]), axis=1)
        raw[valid] = (bands.size - 1) - nearest
    t = s["t"].to_numpy()
    # Fallback only for a single-sample frame; 0.046s ~= TrackAddict's 21.7 Hz.
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.046
    n_dwell = max(1, int(round(min_dwell_s / dt)))
    confirmed = _confirm_runs(raw, n_dwell)
    gear = pd.Series(confirmed, index=s.index).ffill()
    gear[rpm == 0] = np.nan
    return gear.reindex(samples.index)


def _obd_anchored_time(t, dist_lap_m, dist_a: float, dist_b: float) -> float | None:
    """Elapsed time between the OBD-distance crossings of dist_a and dist_b — the
    robust fallback when GPS is too coarse to time the gate crossings. Immune to
    GPS rate; assumes the (small) GPS-vs-OBD offset is ~0, which is the right prior
    when the true offset is unrecoverable. None if a distance is outside OBD range."""
    dl = np.asarray(dist_lap_m, dtype=float)
    tt = np.asarray(t, dtype=float)
    if dist_a < dl.min() or dist_b > dl.max():
        return None
    return float(np.interp(dist_b, dl, tt) - np.interp(dist_a, dl, tt))


def span_time(
    lap_samples: pd.DataFrame,
    dist_a: float,
    dist_b: float,
    centerline: pd.DataFrame,
    frame: "TrackFrame | None" = None,
    corridor=None,
):
    """Gate-to-gate section time for one lap — a thin shell over the trajectory layer.

    Estimates the lap's monotone `s_hat` (once) and times the section between its
    `dist_a`/`dist_b` crossings via `section_timing`, so value and confidence come
    from the SAME evidence pass (design R1). Returns a `SectionTiming` (never None,
    design R10): `status='no_coverage'` with NaN value when a bound is outside the
    lap's `s_hat` range. `corridor=None` builds a `default_corridor` from
    `centerline`; real callers pass the loaded per-track corridor.
    """
    from .trajectory import default_corridor, estimate_trajectory, section_timing
    if frame is None:
        frame = TrackFrame.from_centerline(centerline)
    if corridor is None:
        corridor = default_corridor(centerline, frame)
    traj = estimate_trajectory(lap_samples, corridor, frame)
    return section_timing(traj, dist_a, dist_b, corridor)


def section_range_bounds(
    track_def: dict,
    from_id: str,
    to_id: str,
    pre_m: float = 50.0,
    post_cap_m: float = 250.0,
) -> tuple[float, float]:
    """Section window spanning corners from_id..to_id inclusive.

    a = from_corner.start_m - pre_m; b = min(start of the corner after to_id,
    to_corner.end_m + post_cap_m). For from_id == to_id this returns exactly the
    same (a, b) as section_bounds(track_def)[from_id]. Raises ValueError if
    to_id precedes from_id in track order.
    """
    corners = sorted(track_def["corners"], key=lambda c: c["start_m"])
    ids = [c["id"] for c in corners]
    i_from = ids.index(from_id)
    i_to = ids.index(to_id)
    if i_to < i_from:
        raise ValueError(f"to_id {to_id!r} precedes from_id {from_id!r}")
    lap_len = float(track_def.get("lap_length_internal_m", corners[-1]["end_m"] + 200))
    next_start = corners[i_to + 1]["start_m"] if (i_to + 1) < len(corners) else lap_len
    a = max(0.0, corners[i_from]["start_m"] - pre_m)
    b = float(min(next_start, corners[i_to]["end_m"] + post_cap_m))
    return (a, b)


def range_section_times(
    track: str,
    track_def: dict,
    from_id: str,
    to_id: str,
    pre_m: float = 50.0,
    post_cap_m: float = 250.0,
) -> pd.DataFrame:
    """For every (session_id, lap), the section time across from_id..to_id.

    Same trajectory-layer path as `section_times` over the single range window
    (`section_range_bounds`): one `estimate_trajectory` per lap + one
    `section_timing`; a row is ALWAYS emitted (R10). For from_id == to_id this
    matches `section_times()` for that corner. Long-form columns match
    `section_times` minus `corner_id`: session_id, lap, section_time_s, sigma_t_s,
    status, driven_m, rank_eligible, timing_gap_s, timing_reliable, checks_json.
    """
    a, b = section_range_bounds(track_def, from_id, to_id, pre_m, post_cap_m)
    rows: list[tuple] = []
    for sid, lap_n, g, corridor, traj in _corpus_trajectories(track):
        val, sig, status, driven, rank, gap, checks = _section_row(traj, g, corridor, a, b)
        rows.append((sid, lap_n, val, sig, status, driven, rank, gap, rank, checks))
    return pd.DataFrame(rows, columns=_RANGE_TIME_COLS)


def classify_t8_section(
    samples: pd.DataFrame,
    track_def: dict,
    bands: np.ndarray,
    min_dwell_s: float = 0.3,
) -> dict:
    """Label one lap's T8 entry by downshift behaviour.

    Scans the detection window [T8.start_m - 50, T10.end_m] in track_dist_m.
    The window ends at T10 — before the T11 braking zone — so the routine
    downshift for T11 braking is never counted. Returns a dict with:
      label           - 'downshift' | 'no-downshift' | 'excluded'
      excluded_reason - 'traffic' | 'no-obd' | 'ambiguous' | None
      downshift_dist_m- track_dist_m of the single downshift, or None
      t8_apex_gear    - last confirmed gear at/before the T8 apex, or None

    downshift = exactly one downshift, inside the T8 entry zone
    [T8.start_m - 50, T8.apex_m]. no-downshift = zero downshifts in the window.
    Everything else (downshift inside the window but outside the zone = traffic,
    multiple downshifts, no OBD, too few samples) is excluded.
    """
    corners = {c["id"]: c for c in track_def["corners"]}
    t8 = corners["T8"]
    win_lo = t8["start_m"] - 50.0
    win_hi = corners["T10"]["end_m"]
    zone_hi = t8["apex_m"]

    w = samples[
        (samples["track_dist_m"] >= win_lo)
        & (samples["track_dist_m"] <= win_hi)
    ].sort_values("t")
    result = {
        "label": "excluded",
        "excluded_reason": "ambiguous",
        "downshift_dist_m": None,
        "t8_apex_gear": None,
    }
    if len(w) < 10:
        return result
    if (w["rpm"] == 0).all():
        result["excluded_reason"] = "no-obd"
        return result

    gear = derive_gear(w, bands, min_dwell_s=min_dwell_s).to_numpy()
    x = w["track_dist_m"].to_numpy()

    apex_gear = gear[x <= t8["apex_m"]]
    apex_gear = apex_gear[~np.isnan(apex_gear)]
    result["t8_apex_gear"] = float(apex_gear[-1]) if apex_gear.size else None

    valid = ~np.isnan(gear)
    gv = gear[valid]
    xv = x[valid]
    if gv.size < 2:
        return result

    downshift_locs = xv[1:][np.diff(gv) < 0]
    if downshift_locs.size == 0:
        result.update(label="no-downshift", excluded_reason=None)
        return result
    if downshift_locs.size > 1:
        return result

    loc = float(downshift_locs[0])
    result["downshift_dist_m"] = loc
    if loc <= zone_hi:
        result.update(label="downshift", excluded_reason=None)
    else:
        result["excluded_reason"] = "traffic"
    return result
