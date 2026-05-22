"""Legacy notebook — superseded by `python -m lap_analyzer.cli.install_centerline --track ridge`.

The original body is preserved below in a comment block for historical reference.
"""
# Refactored to CLI on 2026-05-19. Original body follows (commented out):
#
# """Install the synthetic centerline as the canonical reference for the labeler.
#
# Wraps the centerline parquet into the directory layout the labeler expects
# (samples.parquet + laps.csv) so no labeler code changes are needed. The
# synthetic "session" lives at data/sessions/ridge/_synthetic_centerline/.
#
# After this runs, ridge.json's reference_lap should point to ("_synthetic_centerline", 1).
# """
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
#
# CENTERLINE_PATH = Path(r"D:\Projects\lap-analyzer\data\corpus\ridge_centerline.parquet")
# SYNTH_DIR = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge\_synthetic_centerline")
# SYNTH_DIR.mkdir(parents=True, exist_ok=True)
#
# cl = pd.read_parquet(CENTERLINE_PATH)
# print(f"Loaded centerline: {len(cl)} points, track_dist_m {cl.track_dist_m.min():.1f}-{cl.track_dist_m.max():.1f}")
#
# # Build a samples.parquet that mimics the labeler's expected schema.
# # The labeler only reads lat, long, dist_lap_m for the reference. Other
# # channels are filled with sentinel values so any accidental access surfaces clearly.
# n = len(cl)
# samples = pd.DataFrame({
#     "session_id": ["_synthetic_centerline"] * n,
#     "t": cl.track_dist_m.values * 0.05,  # synthetic time, monotonic; no real meaning
#     "lap": [1] * n,
#     "dist_m": cl.track_dist_m.values,
#     "dist_lap_m": cl.track_dist_m.values,
#     "speed_mph": np.full(n, np.nan),
#     "speed_mph_gps": np.full(n, np.nan),
#     "throttle_norm": np.full(n, np.nan),
#     "brake": np.zeros(n, dtype=int),
#     "rpm": np.full(n, np.nan),
#     "lat_g": np.full(n, np.nan),
#     "long_g": np.full(n, np.nan),
#     "coolant_f": np.full(n, np.nan),
#     "iat_f": np.full(n, np.nan),
#     "lat": cl.lat.values,
#     "long": cl["long"].values,
#     "altitude_m": np.full(n, np.nan),
#     "gps_accuracy_m": np.full(n, 0.0),
# })
# samples_path = SYNTH_DIR / "samples.parquet"
# samples.to_parquet(samples_path, index=False)
# print(f"Wrote {samples_path}  ({len(samples)} rows)")
#
# # Build a laps.csv with one entry for lap 1 — marked clean so the labeler accepts it.
# laps = pd.DataFrame([{
#     "lap": 1,
#     "lap_time_s": float(cl.track_dist_m.max() - cl.track_dist_m.min()) * 0.05,  # arbitrary, matches synthetic t
#     "is_clean": True,
#     "session_id": "_synthetic_centerline",
# }])
# laps_path = SYNTH_DIR / "laps.csv"
# laps.to_csv(laps_path, index=False)
# print(f"Wrote {laps_path}")
#
# # Also write a stub meta.json so the corpus-builder doesn't choke if it ever sees this dir
# import json
# meta = {
#     "session_id": "_synthetic_centerline",
#     "track_id": "ridge",
#     "synthetic": True,
#     "source": "data/corpus/ridge_centerline.parquet",
#     "purpose": "Canonical track centerline. Reference for track_dist_m alignment. Not a real recording.",
#     "n_grid_points": int(n),
#     "grid_spacing_m": 1.0,
# }
# (SYNTH_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
# print(f"Wrote {SYNTH_DIR / 'meta.json'}")
# print("\nNow update ridge.json reference_lap to:")
# print('  {"session_id": "_synthetic_centerline", "lap": 1, "notes": "Synthetic centerline..."}')
