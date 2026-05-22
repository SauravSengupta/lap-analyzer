"""Legacy notebook — superseded by `python -m lap_analyzer.cli.build_centerline --track ridge`.

The original body is preserved below in a comment block for historical reference.
The Ridge-specific 'regions' diagnostic at the bottom is preserved here only;
the new CLI prints global spread stats but not per-region breakdown.
"""
# Refactored to CLI on 2026-05-19. Original body follows (commented out):
#
# """Build a synthetic track centerline by aggregating across clean laps.
#
# Pipeline:
#   1. Load all clean-lap samples in the corpus.
#   2. Apply per-lap GPS drift correction (subtract gps_drift_lat_m / gps_drift_lon_m
#      stored on each sample by the labeler).
#   3. Filter out bad samples:
#        - samples on laps with `gps_drift_disagreement_m > 12m` (drift estimate
#          untrustworthy for the whole lap)
#        - samples on transits where `track_dist_offset_max_m > 40m` (the per-corner
#          glitch signature). We pull this from corners.parquet and apply per-corner.
#        - sessions flagged `gps_unreliable` in data/notes/ridge.json
#        - sessions with `exclude` in notes
#   4. Bin remaining samples to a 1 m grid by track_dist_m. Per bin, take the median
#      of (lat, long) across all contributing laps.
#   5. Apply a light Gaussian smoothing (sigma=2m).
#   6. Save as data/corpus/ridge_centerline.parquet with columns
#      (track_dist_m, lat, long, n_samples, lat_spread_m, long_spread_m).
#
# The output is the canonical ground-truth track centerline, parameterized by track
# distance. Subsequent labeler runs will project samples onto this centerline.
# """
# import json
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# from scipy.ndimage import gaussian_filter1d
#
# M_PER_DEG_LAT = 111_132.0
# M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))
#
# ROOT = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
# NOTES_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge.json")
# OUT_PATH = Path(r"D:\Projects\lap-analyzer\data\corpus\ridge_centerline.parquet")
# OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
#
# DRIFT_DISAGREEMENT_LIMIT_M = 12.0
# TRANSIT_OFFSET_LIMIT_M = 40.0
# BIN_M = 1.0
# SMOOTH_SIGMA_M = 2.0
# LAP_LENGTH_M = 3962.0  # from ridge.json lap_length_internal_m
#
# notes = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
# excluded_sessions = {sid for sid, m in notes.items() if "exclude" in m}
# unreliable_sessions = {sid for sid, m in notes.items() if m.get("flag") == "gps_unreliable"}
# print(f"Excluded sessions: {len(excluded_sessions)} ({sorted(excluded_sessions)})")
# print(f"Unreliable sessions: {len(unreliable_sessions)} ({sorted(unreliable_sessions)})")
#
# # Build lookup: (session_id, lap, corner) -> track_dist_offset_max_m
# glitch_keys: set[tuple[str, int, str]] = set()
# n_glitched_transits = 0
# n_total_transits = 0
# for sid in [d.name for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")]:
#     cpath = ROOT / sid / "corners.parquet"
#     if not cpath.exists():
#         continue
#     cdf = pd.read_parquet(cpath)
#     n_total_transits += len(cdf)
#     bad = cdf[cdf.track_dist_offset_max_m > TRANSIT_OFFSET_LIMIT_M]
#     n_glitched_transits += len(bad)
#     for _, r in bad.iterrows():
#         glitch_keys.add((r.session_id, int(r.lap), r.corner_id))
# print(f"\nGlitched transits: {n_glitched_transits} / {n_total_transits} ({100*n_glitched_transits/n_total_transits:.1f}%)")
# print(f"Distinct (session, lap, corner) tuples flagged glitched: {len(glitch_keys)}")
#
# # Aggregate samples
# print("\nLoading and filtering samples from all sessions...")
# all_lat = []
# all_lon = []
# all_td = []
# n_laps_used = 0
# n_laps_skipped_disagree = 0
# n_laps_skipped_excluded = 0
# n_samples_used = 0
# n_samples_dropped_glitch = 0
#
# for sid in sorted([d.name for d in ROOT.iterdir() if d.is_dir() and not d.name.startswith("_")]):
#     if sid in excluded_sessions or sid in unreliable_sessions:
#         continue
#     spath = ROOT / sid / "samples.parquet"
#     lpath = ROOT / sid / "laps.csv"
#     if not spath.exists() or not lpath.exists():
#         continue
#     samples = pd.read_parquet(spath)
#     laps = pd.read_csv(lpath)
#     clean = laps[laps.is_clean.astype(bool)]
#     for _, lap_row in clean.iterrows():
#         ln = int(lap_row.lap)
#         lap_s = samples[samples["lap"] == ln]
#         if len(lap_s) == 0:
#             continue
#         # Whole-lap filter: anchor disagreement
#         disagree = float(lap_s["gps_drift_disagreement_m"].iloc[0])
#         if disagree > DRIFT_DISAGREEMENT_LIMIT_M:
#             n_laps_skipped_disagree += 1
#             continue
#         # Drift-correct GPS
#         lat_corr = lap_s["lat"].values - lap_s["gps_drift_lat_m"].values / M_PER_DEG_LAT
#         lon_corr = lap_s["long"].values - lap_s["gps_drift_lon_m"].values / M_PER_DEG_LON
#         td = lap_s["track_dist_m"].values
#         corner = lap_s["corner"].values
#         # Per-sample filter: drop samples in glitched corner-transits
#         keep = np.ones(len(lap_s), dtype=bool)
#         for cid in pd.unique(corner):
#             if (sid, ln, cid) in glitch_keys:
#                 keep &= corner != cid
#         n_samples_dropped_glitch += int((~keep).sum())
#         all_lat.append(lat_corr[keep])
#         all_lon.append(lon_corr[keep])
#         all_td.append(td[keep])
#         n_samples_used += int(keep.sum())
#         n_laps_used += 1
#
# print(f"\nLaps used: {n_laps_used}")
# print(f"Laps skipped (anchor disagreement > {DRIFT_DISAGREEMENT_LIMIT_M}m): {n_laps_skipped_disagree}")
# print(f"Samples used: {n_samples_used:,}")
# print(f"Samples dropped (in glitched transit): {n_samples_dropped_glitch:,}")
#
# lat_all = np.concatenate(all_lat)
# lon_all = np.concatenate(all_lon)
# td_all = np.concatenate(all_td)
# print(f"\nTotal samples for binning: {len(lat_all):,}")
#
# # Aggregation: at each 1m grid point, gather samples within ±window_radius and take median.
# # Sliding-window approach because the labeler's kd-tree produces discrete track_dist_m values
# # (one per seed-lap sample), so strict 1m bins leave ~75% of grid points empty.
# WINDOW_RADIUS_M = 2.0
# n_grid = int(np.ceil(LAP_LENGTH_M / BIN_M))
# grid_td = np.arange(n_grid) * BIN_M + BIN_M / 2  # bin centers: 0.5, 1.5, ..., 3961.5
# print(f"\nAggregating to {n_grid} grid points at {BIN_M}m spacing using ±{WINDOW_RADIUS_M}m sliding window...")
#
# # Sort samples by track_dist_m for window-based slicing
# order = np.argsort(td_all)
# td_sorted = td_all[order]
# lat_sorted = lat_all[order]
# lon_sorted = lon_all[order]
#
# # For each grid point, find samples in window via searchsorted (O(log n))
# lo_idx = np.searchsorted(td_sorted, grid_td - WINDOW_RADIUS_M, side="left")
# hi_idx = np.searchsorted(td_sorted, grid_td + WINDOW_RADIUS_M, side="right")
#
# lat_med = np.empty(n_grid)
# lon_med = np.empty(n_grid)
# lat_std = np.empty(n_grid)
# lon_std = np.empty(n_grid)
# n_per = np.empty(n_grid, dtype=int)
# for i in range(n_grid):
#     lo, hi = lo_idx[i], hi_idx[i]
#     if hi - lo < 5:  # too sparse, will interpolate
#         lat_med[i] = np.nan
#         lon_med[i] = np.nan
#         lat_std[i] = np.nan
#         lon_std[i] = np.nan
#         n_per[i] = hi - lo
#         continue
#     lat_med[i] = np.median(lat_sorted[lo:hi])
#     lon_med[i] = np.median(lon_sorted[lo:hi])
#     lat_std[i] = np.std(lat_sorted[lo:hi])
#     lon_std[i] = np.std(lon_sorted[lo:hi])
#     n_per[i] = hi - lo
#
# empty = int(np.isnan(lat_med).sum())
# print(f"Sparse grid points (<5 samples in window, will interpolate): {empty}")
#
# agg = pd.DataFrame({
#     "track_dist_m": grid_td,
#     "lat_med": lat_med,
#     "lon_med": lon_med,
#     "lat_std": lat_std,
#     "lon_std": lon_std,
#     "n": n_per,
# })
# agg["lat_med"] = agg["lat_med"].interpolate(limit_direction="both")
# agg["lon_med"] = agg["lon_med"].interpolate(limit_direction="both")
#
# # Smooth: gaussian on lat/long
# agg["lat"] = gaussian_filter1d(agg["lat_med"].values, sigma=SMOOTH_SIGMA_M, mode="wrap")
# agg["lon"] = gaussian_filter1d(agg["lon_med"].values, sigma=SMOOTH_SIGMA_M, mode="wrap")
#
# # Per-grid-point spread
# agg["lat_spread_m"] = agg["lat_std"].fillna(0) * M_PER_DEG_LAT
# agg["lon_spread_m"] = agg["lon_std"].fillna(0) * M_PER_DEG_LON
# agg["spread_m"] = np.sqrt(agg["lat_spread_m"] ** 2 + agg["lon_spread_m"] ** 2)
#
# # Write
# out = agg[["track_dist_m", "lat", "lon", "n", "spread_m"]].copy()
# out = out.rename(columns={"lon": "long"})
# out.to_parquet(OUT_PATH, index=False)
# print(f"\nWrote centerline to {OUT_PATH}")
# print(f"  {len(out)} points at {BIN_M}m grid spacing")
# print(f"  Median samples per bin: {out.n.median():.0f}  (min {out.n.min()}, max {out.n.max()})")
# print(f"  Median between-lap spread: {out.spread_m.median():.2f}m  p90 {out.spread_m.quantile(0.9):.2f}m  max {out.spread_m.max():.2f}m")
#
# # Diagnostic: spread by region
# print("\nBetween-lap spread by region (catches any remaining glitch zones):")
# print(f"  {'region':<22s} {'samples':>9s}  {'median spread':>14s}  {'p90 spread':>11s}  {'max':>8s}")
# regions = [
#     ("Front straight", 0, 380),
#     ("T1-T2",  380, 670),
#     ("T3-T5",  670, 1100),
#     ("Carousel T6", 1100, 1648),
#     ("Inner loop T7-T9", 1648, 2237),
#     ("T10-T11", 2237, 2700),
#     ("T12 + back straight", 2700, 3203),
#     ("Chicane T13-T15", 3203, 3490),
#     ("T16 + finish", 3490, 3962),
# ]
# for name, lo, hi in regions:
#     sub = out[(out.track_dist_m >= lo) & (out.track_dist_m < hi)]
#     n_total = sub.n.sum()
#     print(f"  {name:<22s} {int(n_total):>9d}  {sub.spread_m.median():>12.2f}m  {sub.spread_m.quantile(0.9):>9.2f}m  {sub.spread_m.max():>6.2f}m")
