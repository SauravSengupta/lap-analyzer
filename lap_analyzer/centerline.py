"""Build a synthetic track centerline from drift-corrected clean laps.

The centerline is the canonical track_dist_m ruler for a track. See README §3.

Pipeline:
  1. Load all clean-lap samples in the corpus.
  2. Apply per-lap GPS drift correction (subtract gps_drift_lat_m / gps_drift_lon_m
     stored on each sample by the labeler).
  3. Filter out bad samples:
       - samples on laps with `gps_drift_disagreement_m > 12m`
       - samples on transits where `track_dist_offset_max_m > 40m`
       - sessions flagged `gps_unreliable` in data/notes/<track>.json
       - sessions with `exclude` in notes
  4. Bin remaining samples to a 1 m grid by track_dist_m. Per bin, take the median
     of (lat, long) across all contributing laps.
  5. Apply a light Gaussian smoothing (sigma=2m).
  6. Save parquet with columns (track_dist_m, lat, long, n, spread_m).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter1d

M_PER_DEG_LAT = 111_132.0

DRIFT_DISAGREEMENT_LIMIT_M = 12.0
TRANSIT_OFFSET_LIMIT_M = 40.0
BIN_M = 1.0
WINDOW_RADIUS_M = 2.0
SMOOTH_SIGMA_M = 2.0
# Non-corner (straight/transition) bins whose contributing samples disagree
# laterally by more than this are rejected (set NaN) and filled by periodic
# interpolation. Straights should have tight spread; high spread there means the
# lap-wrap artifact, where kd-tree projection assigns samples from physically
# different positions to the same track_dist_m. Applied ONLY outside corner boxes.
SPREAD_REJECT_M = 6.0
# Non-corner bins holding more than this multiple of the median sample count are
# also rejected: a pile-up signals track_dist_m folding (the wrap zone crams many
# laps' front-straight samples into a few bins, producing the visible zigzag).
N_FOLD_MULT = 1.5
# Corners keep the sharp smoothing (SMOOTH_SIGMA_M); open straights get this
# heavier sigma to erase residual lap-wrap wiggle that survives rejection (two
# internally-consistent lap clusters at the same track_dist_m). A real straight's
# gentle bow is low-frequency and survives heavy smoothing; the wiggle does not.
STRAIGHT_SMOOTH_SIGMA_M = 10.0
# Blend ramp from sharp (at corner edge) to heavy (this far into a straight).
STRAIGHT_BLEND_TRANSITION_M = 25.0


def build_centerline(
    sessions_dir: Path,
    notes_path: Path,
    lap_length_m: float,
    track_lat_deg: float,
    out_path: Path,
    protected_ranges: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Aggregate clean-lap GPS into a 1m centerline polyline.

    Reads every (session, lap) under sessions_dir, drops sessions flagged
    exclude/gps_unreliable in notes_path, drift-corrects each lap using its
    pre-computed gps_drift_{lat,lon}_m columns, and writes the binned median
    (lat, long) polyline to out_path. Returns the DataFrame for the caller.

    protected_ranges: corner [start_m, end_m] spans (in track_dist_m). Bins inside
    these are never rejected for high spread/fold — real corner line-variation is
    legitimate. Bins OUTSIDE them (straights, esp. the start/finish wrap zone) are
    rejected when their samples disagree, then periodic-interpolated. This removes
    the front-straight "zigzag" caused by cross-lap track_dist_m wrap misalignment
    without flattening any actual corner.
    """
    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.radians(track_lat_deg))
    protected_ranges = protected_ranges or []

    out_path.parent.mkdir(parents=True, exist_ok=True)

    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    excluded_sessions = {sid for sid, m in notes.items() if "exclude" in m}
    unreliable_sessions = {sid for sid, m in notes.items() if m.get("flag") == "gps_unreliable"}
    print(f"Excluded sessions: {len(excluded_sessions)} ({sorted(excluded_sessions)})")
    print(f"Unreliable sessions: {len(unreliable_sessions)} ({sorted(unreliable_sessions)})")

    # Build lookup: (session_id, lap, corner) -> track_dist_offset_max_m
    glitch_keys: set[tuple[str, int, str]] = set()
    n_glitched_transits = 0
    n_total_transits = 0
    for sid in [d.name for d in sessions_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]:
        cpath = sessions_dir / sid / "corners.parquet"
        if not cpath.exists():
            continue
        cdf = pd.read_parquet(cpath)
        n_total_transits += len(cdf)
        bad = cdf[cdf.track_dist_offset_max_m > TRANSIT_OFFSET_LIMIT_M]
        n_glitched_transits += len(bad)
        for _, r in bad.iterrows():
            glitch_keys.add((r.session_id, int(r.lap), r.corner_id))
    pct = 100 * n_glitched_transits / n_total_transits if n_total_transits else float("nan")
    print(f"\nGlitched transits: {n_glitched_transits} / {n_total_transits} ({pct:.1f}%)")
    print(f"Distinct (session, lap, corner) tuples flagged glitched: {len(glitch_keys)}")

    # Aggregate samples
    print("\nLoading and filtering samples from all sessions...")
    all_lat = []
    all_lon = []
    all_td = []
    n_laps_used = 0
    n_laps_skipped_disagree = 0
    n_samples_used = 0
    n_samples_dropped_glitch = 0

    for sid in sorted([d.name for d in sessions_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]):
        if sid in excluded_sessions or sid in unreliable_sessions:
            continue
        spath = sessions_dir / sid / "samples.parquet"
        lpath = sessions_dir / sid / "laps.csv"
        if not spath.exists() or not lpath.exists():
            continue
        samples = pd.read_parquet(spath)
        laps = pd.read_csv(lpath)
        clean = laps[laps.is_clean.astype(bool)]
        for _, lap_row in clean.iterrows():
            ln = int(lap_row.lap)
            lap_s = samples[samples["lap"] == ln]
            if len(lap_s) == 0:
                continue
            # Whole-lap filter: anchor disagreement
            disagree = float(lap_s["gps_drift_disagreement_m"].iloc[0])
            if disagree > DRIFT_DISAGREEMENT_LIMIT_M:
                n_laps_skipped_disagree += 1
                continue
            # Drift-correct GPS
            lat_corr = lap_s["lat"].values - lap_s["gps_drift_lat_m"].values / M_PER_DEG_LAT
            lon_corr = lap_s["long"].values - lap_s["gps_drift_lon_m"].values / m_per_deg_lon
            td = lap_s["track_dist_m"].values
            corner = lap_s["corner"].values
            # Per-sample filter: drop samples in glitched corner-transits
            keep = np.ones(len(lap_s), dtype=bool)
            for cid in pd.unique(corner):
                if (sid, ln, cid) in glitch_keys:
                    keep &= corner != cid
            n_samples_dropped_glitch += int((~keep).sum())
            all_lat.append(lat_corr[keep])
            all_lon.append(lon_corr[keep])
            all_td.append(td[keep])
            n_samples_used += int(keep.sum())
            n_laps_used += 1

    print(f"\nLaps used: {n_laps_used}")
    print(f"Laps skipped (anchor disagreement > {DRIFT_DISAGREEMENT_LIMIT_M}m): {n_laps_skipped_disagree}")
    print(f"Samples used: {n_samples_used:,}")
    print(f"Samples dropped (in glitched transit): {n_samples_dropped_glitch:,}")

    lat_all = np.concatenate(all_lat)
    lon_all = np.concatenate(all_lon)
    td_all = np.concatenate(all_td)
    print(f"\nTotal samples for binning: {len(lat_all):,}")

    # Aggregation: at each 1m grid point, gather samples within ±window_radius and take median.
    n_grid = int(np.ceil(lap_length_m / BIN_M))
    grid_td = np.arange(n_grid) * BIN_M + BIN_M / 2  # bin centers: 0.5, 1.5, ..., lap_length-0.5
    print(f"\nAggregating to {n_grid} grid points at {BIN_M}m spacing using ±{WINDOW_RADIUS_M}m sliding window...")

    # Sort samples by track_dist_m for window-based slicing
    order = np.argsort(td_all)
    td_sorted = td_all[order]
    lat_sorted = lat_all[order]
    lon_sorted = lon_all[order]

    # For each grid point, find samples in window via searchsorted (O(log n))
    lo_idx = np.searchsorted(td_sorted, grid_td - WINDOW_RADIUS_M, side="left")
    hi_idx = np.searchsorted(td_sorted, grid_td + WINDOW_RADIUS_M, side="right")

    lat_med = np.empty(n_grid)
    lon_med = np.empty(n_grid)
    lat_std = np.empty(n_grid)
    lon_std = np.empty(n_grid)
    n_per = np.empty(n_grid, dtype=int)
    for i in range(n_grid):
        lo, hi = lo_idx[i], hi_idx[i]
        if hi - lo < 5:  # too sparse, will interpolate
            lat_med[i] = np.nan
            lon_med[i] = np.nan
            lat_std[i] = np.nan
            lon_std[i] = np.nan
            n_per[i] = hi - lo
            continue
        lat_med[i] = np.median(lat_sorted[lo:hi])
        lon_med[i] = np.median(lon_sorted[lo:hi])
        lat_std[i] = np.std(lat_sorted[lo:hi])
        lon_std[i] = np.std(lon_sorted[lo:hi])
        n_per[i] = hi - lo

    empty = int(np.isnan(lat_med).sum())

    # Per-bin lateral spread of contributing samples.
    spread_per = np.sqrt((lat_std * M_PER_DEG_LAT) ** 2 + (lon_std * m_per_deg_lon) ** 2)

    # "Protected" bins lie inside a corner box — never reject these (corner line
    # variation is real). Everything else is a straight/transition; there, high
    # sample disagreement or pile-up indicates the lap-wrap track_dist_m artifact.
    protected = np.zeros(n_grid, dtype=bool)
    for s, e in protected_ranges:
        protected |= (grid_td >= s) & (grid_td <= e)

    median_n = float(np.median(n_per[n_per > 0])) if (n_per > 0).any() else 0.0
    high_spread = np.nan_to_num(spread_per, nan=0.0) > SPREAD_REJECT_M
    folded = n_per > N_FOLD_MULT * median_n  # many laps' samples crammed into one bin
    reject = (~protected) & (high_spread | folded)
    valid = (~np.isnan(lat_med)) & (~reject)
    print(f"Sparse grid points (<5 samples in window): {empty}")
    print(f"Rejected (non-corner, spread>{SPREAD_REJECT_M:.0f}m or n>{N_FOLD_MULT:.1f}x median): {int(reject.sum())}")
    print(f"Valid bins used for interpolation: {int(valid.sum())} / {n_grid}")

    valid_td = grid_td[valid]
    lat_filled = np.interp(grid_td, valid_td, lat_med[valid], period=lap_length_m)
    lon_filled = np.interp(grid_td, valid_td, lon_med[valid], period=lap_length_m)

    agg = pd.DataFrame({
        "track_dist_m": grid_td,
        "lat_med": lat_filled,
        "lon_med": lon_filled,
        "lat_std": lat_std,
        "lon_std": lon_std,
        "n": n_per,
    })

    # Smooth: corner-aware variable sigma. Corners get the sharp sigma; open
    # straights blend toward a heavier sigma to erase residual wrap wiggle.
    # Blend weight w: 0 at/inside a corner, ramping to 1 once TRANSITION_M into a
    # straight. Distance-to-nearest-corner computed on a 3x-tiled array for wrap.
    free = (~protected).astype(np.uint8)
    dist_to_corner = distance_transform_edt(np.tile(free, 3))[n_grid:2 * n_grid] * BIN_M
    w = np.clip(dist_to_corner / STRAIGHT_BLEND_TRANSITION_M, 0.0, 1.0)

    lat_sharp = gaussian_filter1d(agg["lat_med"].values, sigma=SMOOTH_SIGMA_M, mode="wrap")
    lon_sharp = gaussian_filter1d(agg["lon_med"].values, sigma=SMOOTH_SIGMA_M, mode="wrap")
    lat_heavy = gaussian_filter1d(agg["lat_med"].values, sigma=STRAIGHT_SMOOTH_SIGMA_M, mode="wrap")
    lon_heavy = gaussian_filter1d(agg["lon_med"].values, sigma=STRAIGHT_SMOOTH_SIGMA_M, mode="wrap")
    agg["lat"] = (1 - w) * lat_sharp + w * lat_heavy
    agg["lon"] = (1 - w) * lon_sharp + w * lon_heavy

    # Per-grid-point spread
    agg["lat_spread_m"] = agg["lat_std"].fillna(0) * M_PER_DEG_LAT
    agg["lon_spread_m"] = agg["lon_std"].fillna(0) * m_per_deg_lon
    agg["spread_m"] = np.sqrt(agg["lat_spread_m"] ** 2 + agg["lon_spread_m"] ** 2)

    # Write
    out = agg[["track_dist_m", "lat", "lon", "n", "spread_m"]].copy()
    out = out.rename(columns={"lon": "long"})
    out.to_parquet(out_path, index=False)
    print(f"\nWrote centerline to {out_path}")
    print(f"  {len(out)} points at {BIN_M}m grid spacing")
    print(f"  Median samples per bin: {out.n.median():.0f}  (min {out.n.min()}, max {out.n.max()})")
    print(f"  Median between-lap spread: {out.spread_m.median():.2f}m  p90 {out.spread_m.quantile(0.9):.2f}m  max {out.spread_m.max():.2f}m")

    return out


def install_centerline(
    track: str,
    centerline_path: Path,
    synth_dir: Path,
) -> None:
    """Wrap a centerline parquet into a synthetic session directory the labeler can use.

    Reads centerline_path, writes synth_dir/{samples.parquet, laps.csv, meta.json} shaped
    like a real normalized session with NaN sentinels for unused channels. After this runs,
    the track JSON's reference_lap block should point to ("_synthetic_centerline", 1).

    Prints the JSON patch the caller should apply to tracks/<track>.json.
    """
    synth_dir.mkdir(parents=True, exist_ok=True)

    cl = pd.read_parquet(centerline_path)
    print(f"Loaded centerline: {len(cl)} points, track_dist_m {cl.track_dist_m.min():.1f}-{cl.track_dist_m.max():.1f}")

    # Build a samples.parquet that mimics the labeler's expected schema.
    # The labeler only reads lat, long, dist_lap_m for the reference. Other
    # channels are filled with sentinel values so any accidental access surfaces clearly.
    n = len(cl)
    samples = pd.DataFrame({
        "session_id": ["_synthetic_centerline"] * n,
        "t": cl.track_dist_m.values * 0.05,  # synthetic time, monotonic; no real meaning
        "lap": [1] * n,
        "dist_m": cl.track_dist_m.values,
        "dist_lap_m": cl.track_dist_m.values,
        "speed_mph": np.full(n, np.nan),
        "speed_mph_gps": np.full(n, np.nan),
        "throttle_norm": np.full(n, np.nan),
        "brake": np.zeros(n, dtype=int),
        "rpm": np.full(n, np.nan),
        "lat_g": np.full(n, np.nan),
        "long_g": np.full(n, np.nan),
        "coolant_f": np.full(n, np.nan),
        "iat_f": np.full(n, np.nan),
        "lat": cl.lat.values,
        "long": cl["long"].values,
        "altitude_m": np.full(n, np.nan),
        # 0.0 = "perfect accuracy" sentinel: synthetic centerline IS the truth,
        # so any downstream `gps_accuracy_m > threshold` quality filter should pass.
        "gps_accuracy_m": np.full(n, 0.0),
    })
    samples_path = synth_dir / "samples.parquet"
    samples.to_parquet(samples_path, index=False)
    print(f"Wrote {samples_path}  ({len(samples)} rows)")

    # Build a laps.csv with one entry for lap 1 — marked clean so the labeler accepts it.
    laps = pd.DataFrame([{
        "lap": 1,
        "lap_time_s": float(cl.track_dist_m.max() - cl.track_dist_m.min()) * 0.05,  # arbitrary, matches synthetic t
        "is_clean": True,
        "session_id": "_synthetic_centerline",
    }])
    laps_path = synth_dir / "laps.csv"
    laps.to_csv(laps_path, index=False)
    print(f"Wrote {laps_path}")

    # Also write a stub meta.json so the corpus-builder doesn't choke if it ever sees this dir
    meta = {
        "session_id": "_synthetic_centerline",
        "track_id": track,
        "synthetic": True,
        "source": f"data/corpus/{track}_centerline.parquet",
        "purpose": "Canonical track centerline. Reference for track_dist_m alignment. Not a real recording.",
        "n_grid_points": int(n),
        "grid_spacing_m": 1.0,
    }
    (synth_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {synth_dir / 'meta.json'}")
    print(f"\nNow update tracks/{track}.json reference_lap to:")
    note = f"Synthetic centerline at {1.0:.0f}m resolution built from corpus median. See data/corpus/{track}_centerline.parquet."
    print(f'  {{"session_id": "_synthetic_centerline", "lap": 1,')
    print(f'   "notes": "{note}"}}')
