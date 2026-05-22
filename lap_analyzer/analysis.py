"""Component 5 — analysis library.

Pure functions on the corpus DataFrame. Kept minimal; extended as the
visualizer needs new operations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import corpus_dir, sessions_dir


def load_corpus(track: str) -> pd.DataFrame:
    """Read the cross-session corners corpus for a track."""
    return pd.read_parquet(corpus_dir() / f"{track}_corners.parquet")


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


def section_times(
    track: str,
    track_def: dict,
    pre_m: float = 50.0,
    post_cap_m: float = 250.0,
    obd_distance_tolerance_m: float = 15.0,
) -> pd.DataFrame:
    """For every (session_id, lap, corner_id), compute time across the corner's section
    as the (interpolated) interval between the first track_dist_m crossings of the
    section bounds. Returns long-form: session_id, lap, corner_id, section_time_s.

    Rows are emitted only when:
    - the lap crosses both bounds (no extrapolation), AND
    - the OBD-integrated distance traversed during the section is within
      `obd_distance_tolerance_m` of the nominal section width. This rejects
      GPS-glitched laps that "win" by recording a shorter physical path than
      the section nominally spans.
    """
    bounds = section_bounds(track_def, pre_m=pre_m, post_cap_m=post_cap_m)
    rows: list[tuple] = []
    root = sessions_dir(track)
    for sid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sp = sid_dir / "samples.parquet"
        if not sp.exists():
            continue
        sid = sid_dir.name
        try:
            s = pd.read_parquet(sp, columns=["lap", "t", "track_dist_m", "dist_lap_m"])
        except Exception:
            continue
        s = s.sort_values(["lap", "t"])
        for lap_n, g in s.groupby("lap"):
            xs = g["track_dist_m"].to_numpy()
            ts = g["t"].to_numpy()
            ds = g["dist_lap_m"].to_numpy()
            if len(xs) < 2:
                continue
            for cid, (a, b) in bounds.items():
                ta = _first_crossing_t(xs, ts, a)
                if ta is None:
                    continue
                tb = _first_crossing_t(xs, ts, b, after_t=ta)
                if tb is None:
                    continue
                # OBD-distance sanity check: reject if the car's physically traveled
                # distance differs from the nominal section width by more than tolerance.
                obd_dist = float(np.interp(tb, ts, ds) - np.interp(ta, ts, ds))
                if abs(obd_dist - (b - a)) > obd_distance_tolerance_m:
                    continue
                rows.append((sid, int(lap_n), cid, tb - ta))
    return pd.DataFrame(rows, columns=["session_id", "lap", "corner_id", "section_time_s"])


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


def _span_crossing(
    lap_samples: pd.DataFrame, dist_a: float, dist_b: float
) -> tuple[float, float] | tuple[None, None]:
    """Raw span measurement for one lap between two track_dist_m positions.

    Returns (elapsed_s, obd_discrepancy_m), or (None, None) if either bound
    isn't crossed. obd_discrepancy_m = OBD-integrated distance travelled minus
    (dist_b - dist_a): a GPS glitch that mis-scales track_dist_m, or a tighter/
    wider racing line, shows up as a non-zero value. No tolerance is applied —
    callers decide.
    """
    s = lap_samples.sort_values("t")
    xs = s["track_dist_m"].to_numpy()
    ts = s["t"].to_numpy()
    ds = s["dist_lap_m"].to_numpy()
    if len(xs) < 2:
        return None, None
    ta = _first_crossing_t(xs, ts, dist_a)
    if ta is None:
        return None, None
    tb = _first_crossing_t(xs, ts, dist_b, after_t=ta)
    if tb is None:
        return None, None
    obd_dist = float(np.interp(tb, ts, ds) - np.interp(ta, ts, ds))
    return float(tb - ta), obd_dist - (dist_b - dist_a)


def span_time(
    lap_samples: pd.DataFrame,
    dist_a: float,
    dist_b: float,
    obd_tol_m: float | None = None,
) -> float | None:
    """Elapsed seconds for one lap between two track_dist_m positions.

    Interpolates t at the first crossing of dist_a and the first later crossing
    of dist_b. Returns None if either bound isn't crossed, or if the OBD-
    integrated distance over the interval differs from (dist_b - dist_a) by more
    than the tolerance (rejects GPS-glitched laps).

    When obd_tol_m is None the tolerance scales with span length —
    max(15.0, 0.04 * (dist_b - dist_a)). A long multi-corner span legitimately
    accumulates a few percent of path-length variation from a tighter or wider
    racing line; a fixed 15m would reject those clean laps. GPS glitches (tens
    to hundreds of metres off) are still caught.
    """
    elapsed, discrepancy = _span_crossing(lap_samples, dist_a, dist_b)
    if elapsed is None:
        return None
    tol = obd_tol_m if obd_tol_m is not None else max(15.0, 0.04 * (dist_b - dist_a))
    if abs(discrepancy) > tol:
        return None
    return elapsed


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
    obd_distance_tolerance_m: float | None = None,
) -> pd.DataFrame:
    """For every (session_id, lap), elapsed time across the from_id..to_id
    section window. Returns long-form columns: session_id, lap, section_time_s,
    obd_discrepancy_m.

    A lap is emitted only when it cleanly crosses both bounds and its OBD-
    integrated distance matches the nominal span within the tolerance (rejects
    GPS-glitched laps). obd_distance_tolerance_m defaults to None → the
    tolerance scales with span length (max(15.0, 0.04 * span)), so a long range
    isn't rejected for normal racing-line path-length variation. For
    from_id == to_id this matches section_times().

    obd_discrepancy_m is the signed OBD-vs-track_dist divergence per lap — a
    consumer that needs a glitch-free reference (e.g. find_best_lap) can gate on
    it with a tighter bound than the loose emission tolerance.
    """
    a, b = section_range_bounds(track_def, from_id, to_id, pre_m, post_cap_m)
    tol = (obd_distance_tolerance_m if obd_distance_tolerance_m is not None
           else max(15.0, 0.04 * (b - a)))
    rows: list[tuple] = []
    root = sessions_dir(track)
    for sid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sp = sid_dir / "samples.parquet"
        if not sp.exists():
            continue
        sid = sid_dir.name
        try:
            s = pd.read_parquet(sp, columns=["lap", "t", "track_dist_m", "dist_lap_m"])
        except Exception:
            continue
        s = s.sort_values(["lap", "t"])
        for lap_n, g in s.groupby("lap"):
            elapsed, discrepancy = _span_crossing(g, a, b)
            if elapsed is None or abs(discrepancy) > tol:
                continue
            rows.append((sid, int(lap_n), elapsed, discrepancy))
    return pd.DataFrame(
        rows, columns=["session_id", "lap", "section_time_s", "obd_discrepancy_m"])


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
