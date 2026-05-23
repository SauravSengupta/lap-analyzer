# Pipeline

A runbook for the data pipeline: what each CLI does, the input it expects, and
the output schema it produces. For *why* the pipeline is shaped this way, see
[ARCHITECTURE.md](ARCHITECTURE.md). To bootstrap a brand-new track, see
[NEW-TRACK.md](NEW-TRACK.md).

Everything runs as a module from the repo root with `PYTHONPATH` set so the
`lap_analyzer` package is importable:

```powershell
# PowerShell
$env:PYTHONPATH = "."
python -m lap_analyzer.cli.normalize --track ridge --all
```

```bash
# bash
PYTHONPATH=. python -m lap_analyzer.cli.normalize --track ridge --all
```

Data lives under `data/` (overridable with `DATA_ROOT`). Track definitions live
under `tracks/` and are read regardless of `DATA_ROOT`.

---

## Pipeline overview

For a track that already has a centerline installed, the everyday pipeline is
four steps, run in order:

| # | CLI module | Reads | Writes |
|---|---|---|---|
| 1 | `lap_analyzer.cli.normalize` | TrackAddict CSVs in `data/raw/<track>/` | `data/sessions/<track>/<sid>/{samples.parquet, laps.csv, meta.json}` |
| 2 | `lap_analyzer.cli.label_corners` | each session's `samples.parquet` + `tracks/<track>.json` | adds `track_dist_m`/`corner` to `samples.parquet`; writes `corners.parquet` |
| 3 | `lap_analyzer.cli.flag_quality` | every session's `corners.parquet` (corpus-wide) | rewrites each `corners.parquet` with quality flag columns |
| 4 | `lap_analyzer.cli.build_corpus` | every session's `corners.parquet` | `data/corpus/<track>_corners.parquet` |

Three more CLIs are **bootstrap-time only** — run once when setting up a track
or when rebuilding the geometry ruler (see [NEW-TRACK.md](NEW-TRACK.md)):

| CLI module | Purpose | Writes |
|---|---|---|
| `lap_analyzer.cli.extract_corner_candidates` | scan lat-G peaks across all laps to seed corner ranges | `data/corpus/<track>_candidates.csv` + per-session `candidates.csv` |
| `lap_analyzer.cli.build_centerline` | aggregate clean laps into the synthetic centerline polyline | `data/corpus/<track>_centerline.parquet` |
| `lap_analyzer.cli.install_centerline` | install that centerline as the labeler's reference session | `data/sessions/<track>/_synthetic_centerline/` |

The centerline is the canonical `track_dist_m` ruler. It only needs rebuilding
when the corpus changes meaningfully (many new laps, or a corner-box edit). For
day-to-day "I logged a new session," steps 1–4 are all you run.

---

## CLI reference

### `normalize` — TrackAddict CSV → parquet + lap summary + meta

```
python -m lap_analyzer.cli.normalize [csv] --track TRACK [--all] [--out DIR] [--force]
```

| Flag | Meaning |
|---|---|
| `csv` | Path to a single TrackAddict CSV. Omit when using `--all`. |
| `--track` | Track slug (e.g. `ridge`). **Required.** |
| `--all` | Process every CSV in `data/raw/<track>/`. |
| `--out` | Override the output directory (default `data/sessions/<track>/`). |
| `--force` | Re-normalize even if outputs already exist. |

Per-CSV status line is one of: `ok` (normalized), `skip` (already present, no
`--force`), `excl` (listed in session notes with `exclude`), `noobd` (CSV has no
OBD channels — see below), or `FAIL` (any other error; the run exits non-zero).

```powershell
# Normalize one session
python -m lap_analyzer.cli.normalize "data/raw/ridge/Log-20260517-100304 ....csv" --track ridge
# Re-normalize everything (e.g. after editing lap_length_internal_m)
python -m lap_analyzer.cli.normalize --track ridge --all --force
```

### `label_corners` — corner labels + per-corner-transit table

```
python -m lap_analyzer.cli.label_corners --track TRACK [--session SID]
```

| Flag | Meaning |
|---|---|
| `--track` | Track id; reads `tracks/<id>.json`. **Required.** |
| `--session` | Process only this session id. Default: every session for the track. |

Requires the track's `reference_lap` (the installed centerline) to exist.
Rewrites each `samples.parquet` in place to add `track_dist_m`,
`track_dist_offset_m`, the per-lap drift columns, and a `corner` label, then
writes `corners.parquet` (one row per clean lap × corner). Sessions whose name
starts with `_` (the `_synthetic_centerline` fixture) are skipped.

```powershell
python -m lap_analyzer.cli.label_corners --track ridge
python -m lap_analyzer.cli.label_corners --track ridge --session 20260517-100304
```

### `flag_quality` — corpus-wide reliability flags

```
python -m lap_analyzer.cli.flag_quality --track TRACK
```

Reads every session's `corners.parquet` for the track, computes corpus-wide
columns (pace decile, per-corner z-scores, neighborhood offset, the
`transit_reliable` / `lap_reliable` flags), and writes the augmented table back
to each session's `corners.parquet`. Run it after `label_corners` and before
`build_corpus`. Reference sessions (notes `"reference": true`) are excluded from
the corpus-wide stats but still get flagged.

### `build_corpus` — concatenate sessions into one queryable table

```
python -m lap_analyzer.cli.build_corpus --track TRACK
```

Concatenates every session's `corners.parquet` (excluding reference sessions)
into `data/corpus/<track>_corners.parquet`. This single file is what the
analysis library and visualizer load. It does **not** produce a separate
lap-summary or summary-JSON file — lap-level data is read on demand from each
session's `laps.csv` (see `lap_analyzer.analysis.load_all_laps`).

### `extract_corner_candidates` — lat-G peak scan (bootstrap)

```
python -m lap_analyzer.cli.extract_corner_candidates --track TRACK [--out PATH]
```

Detects lat-G peaks across all flying laps and writes a candidate table used to
seed corner ranges for a new track. Output: `data/corpus/<track>_candidates.csv`
plus a per-session `candidates.csv`. Optional for an established track.

### `build_centerline` — synthetic centerline polyline (bootstrap)

```
python -m lap_analyzer.cli.build_centerline --track TRACK [--out PATH]
```

Aggregates clean drift-corrected laps into a 1 m-resolution median polyline and
writes `data/corpus/<track>_centerline.parquet`. Corner `[start_m, end_m]` spans
are read from the track JSON and passed as "protected ranges" so corner
line-variation is preserved while straights (especially the start/finish wrap
zone) get cleaned. See [ARCHITECTURE.md](ARCHITECTURE.md) §3.

### `install_centerline` — install the centerline as the labeler's reference

```
python -m lap_analyzer.cli.install_centerline --track TRACK
```

Reads `data/corpus/<track>_centerline.parquet` and writes a session-shaped
`data/sessions/<track>/_synthetic_centerline/` (samples.parquet + meta.json +
laps.csv) that `label_corners` uses as the canonical ruler. The track JSON's
`reference_lap` block points at this `_synthetic_centerline` session.

---

## TrackAddict CSV column contract

A TrackAddict export begins with a block of `#`-prefixed comment lines carrying
session metadata, then a header row, then ~21 Hz samples. `normalize` parses the
comment block separately (for vehicle string and start/finish coords) and reads
the data with `comment="#"`.

### Comment header

Lines the parser looks for (others are ignored):

- `# Vehicle: ...` → `meta.vehicle`
- `# End Point: <lat>, <long> @ ...` → `meta.trackaddict_start_finish`
- `# Split Point N: <lat>, <long> @ ...` → `meta.trackaddict_split_points[]`

These are informational. The lap-cut line that actually segments the session
into laps comes from `start_finish` in `tracks/<track>.json`, **not** from the
CSV header.

### Data columns

`normalize` renames the raw TrackAddict columns to canonical names:

| Raw TrackAddict column | Canonical | Notes |
|---|---|---|
| `Time` | `t` | session-relative seconds |
| `UTC Time` | `utc` | wall-clock; used for `date_utc` |
| `Lap` | `lap` | TrackAddict's lap counter |
| `Sector` | `sector` | read but not kept in output |
| `Latitude` | `lat` | WGS84 |
| `Longitude` | `long` | WGS84 |
| `Altitude (m)` | `altitude_m` | |
| `Speed (MPH)` | `speed_mph_gps` | GPS speed (secondary) |
| `Heading` | `heading` | read but not kept |
| `Accuracy (m)` | `gps_accuracy_m` | GPS quality |
| `Accel X` | `long_g` | **longitudinal** on this phone's orientation |
| `Accel Y` | `lat_g` | **lateral**; negated so + = right turn |
| `Accel Z` | `vert_g` | read but not kept |
| `Brake (calculated)` | `brake` | binary 0/1 (TrackAddict-derived) |
| `Engine Speed (RPM) *OBD` | `rpm` | OBD |
| `Vehicle Speed (mph) *OBD` | `speed_mph` | OBD — **canonical speed channel** |
| `Throttle Position (%) *OBD` | `throttle_raw` | normalized per-session → `throttle_norm` |
| `Engine Coolant Temp (F) *OBD` | `coolant_f` | OBD |
| `Intake Air Temp (F) *OBD` | `iat_f` | OBD |
| `Intake Manifold Pressure (PSI) *OBD` | `manifold_psi` | OBD; read but not kept |

> **Axis note.** On the phone/orientation used for this car, TrackAddict's
> `Accel X` is the *longitudinal* channel and `Accel Y` is the *lateral* one —
> the opposite of the column names' usual meaning. `normalize` maps them
> accordingly and negates `lat_g` so the canonical convention is **positive =
> right turn** (and `long_g` positive = acceleration, negative = braking). If you
> add a track logged on different hardware, verify this mapping first.

**Required vs optional:**

- **OBD channels are required.** The six `*OBD` columns above
  (`rpm, speed_mph, throttle_raw, coolant_f, iat_f, manifold_psi`) must all be
  present. If any is missing, `normalize` raises `MissingOBDError` and the
  session is reported as `noobd` and skipped — no output is written. (Roughly
  3–4% of sessions log without OBD; this is expected and normalized away.)
- **GPS + accel + timing columns are required** for output (`Time`, `Lap`,
  `Latitude`, `Longitude`, `Speed (MPH)`, `Accel X`, `Accel Y`,
  `Brake (calculated)`, `Altitude (m)`, `Accuracy (m)`, `UTC Time`). A CSV
  missing these fails normalization (`FAIL`).
- `Sector`, `Heading`, `Accel Z`, and `Intake Manifold Pressure` are read but
  dropped from the output parquet.

---

## Output schema

### `samples.parquet`

One row per sample. After `normalize`, the columns are:

| Column | Type | Meaning |
|---|---|---|
| `session_id` | str | `YYYYMMDD-HHMMSS`, derived from the CSV filename |
| `t` | float | session-relative seconds |
| `lap` | int32 | TrackAddict lap counter (0 = warmup) |
| `dist_m` | float | distance from session start; trapezoidal integration of OBD speed. **Not rescaled** — this is the raw GPS/OBD-integrated distance. |
| `dist_lap_m` | float | distance from the current lap's start, **rescaled** so each lap's total equals `lap_length_internal_m` (see note below) |
| `speed_mph` | float | OBD vehicle speed (canonical) |
| `speed_mph_gps` | float | GPS speed (cross-check) |
| `throttle_norm` | float | `throttle_raw / per-session max`, 0–1 |
| `brake` | int8 | 0/1 |
| `rpm` | int32 | |
| `lat_g` | float | + = right turn |
| `long_g` | float | + = accel, − = braking |
| `coolant_f`, `iat_f` | float | OBD temps |
| `lat`, `long` | float | WGS84 |
| `altitude_m` | float | |
| `gps_accuracy_m` | float | |

After `label_corners`, each `samples.parquet` gains:

| Column | Meaning |
|---|---|
| `track_dist_m` | canonical cross-lap distance: this sample's GPS position projected onto the nearest centerline point |
| `track_dist_offset_m` | how far the kd-tree had to reach to map the sample (high = off the reference path) |
| `corner` | corner id (`T1`…) or straight label (`S_post_T16`) |
| `gps_drift_lat_m`, `gps_drift_lon_m` | per-lap anchor-based drift correction applied (meters) |
| `gps_drift_n_anchors` | how many calibration anchors were found for this lap |
| `gps_drift_disagreement_m` | spread of per-anchor offsets (low = trustworthy drift estimate) |
| `gps_drift_*_centerline`, `gps_drift_region_disagreement_m`, `gps_drift_outliers_filtered` | passive centerline-based drift diagnostics (not used for correction) |

> **`dist_lap_m` vs `dist_m` — important.** `dist_lap_m` is rescaled per lap to
> the configured `lap_length_internal_m`, so for any complete lap its maximum is
> *by construction* equal to `lap_length_internal_m`. It is therefore **not** an
> independent measurement of how far the car drove — it's a normalized coordinate
> that makes "`dist_lap_m = X`" mean the same physical point across laps. If you
> want the real GPS/OBD-integrated lap distance (e.g. to set
> `lap_length_internal_m` for a new track), use the raw `dist_m` deltas instead:
> `samples.groupby("lap")["dist_m"].agg(lambda s: s.max() - s.min())`.

### `laps.csv`

One row per lap:

`session_id, lap, lap_time_s, lap_dist_m, max_speed_mph, avg_speed_mph,
max_rpm, max_lat_g, max_accel_g, max_decel_g, pct_wot, avg_throttle,
pct_braking, coolant_min_f, coolant_max_f, iat_min_f, iat_max_f, is_clean,
clean_reason`

- `lap_time_s` is measured start-crossing to next start-crossing (the final
  in-lap uses end-of-data minus its start).
- `lap_dist_m` is `max(dist_lap_m)` — i.e. `lap_length_internal_m` for any
  complete lap (see the note above; it is not an independent distance).
- `pct_wot` = fraction of samples at `throttle_norm ≥ 0.95`.
- `is_clean` flags **shape only** — `False` for the warmup (first) and cooldown
  (last) lap, with `clean_reason` set accordingly. It is *not* a quality filter;
  per-corner quality lives in `corners.parquet` (traffic on one corner shouldn't
  poison the rest of the lap).

### `meta.json`

A `SessionMeta` record: `session_id, track, vehicle, date_utc, n_laps,
n_clean_laps, sample_rate_hz, duration_s, best_lap, best_lap_time_s,
throttle_max_observed, speed_max_obd_mph, rpm_max, coolant_min_f, coolant_max_f,
iat_first_f, iat_max_f, trackaddict_start_finish, trackaddict_split_points,
raw_csv_path`.

### `corners.parquet` — the analytical workhorse

One row per (clean lap × corner). All distance fields are in `track_dist_m`.
Identity and core metrics (written by `label_corners`):

| Group | Columns |
|---|---|
| Identity | `session_id, date, lap, corner_id` |
| Boundary | `entry_speed_mph, exit_speed_mph, entry_dist_m, exit_dist_m, time_in_corner_s, distance_in_corner_m` |
| Min/max | `min_speed_mph, min_speed_dist_m, max_speed_mph, max_lat_g, max_accel_g, max_decel_g, peak_brake, peak_throttle_norm` |
| Apex | `apex_speed_mph, apex_dist_offset_m` (= `min_speed_dist_m − apex_m`), `latg_peak_dist_m, latg_peak_offset_m` (= `latg_peak_dist_m − apex_m`), `latg_peak_g, secondary_apex_speed_mph` |
| Input timing (offset from corner start) | `throttle_lift_dist_m, brake_on_dist_m, brake_off_dist_m, throttle_return_dist_m, wot_dist_m` |
| Aggregate inputs | `mean_throttle_norm, pct_wot, mean_lat_g, pct_braking` |
| Mapping quality | `track_dist_offset_med_m, track_dist_offset_max_m` |
| Per-lap drift | `gps_drift_lat_m, gps_drift_lon_m, gps_drift_n_anchors, gps_drift_disagreement_m` |
| Sample refs | `sample_idx_start, sample_idx_end` (into the session's `samples.parquet`) |

`flag_quality` then adds:

| Column | Meaning |
|---|---|
| `lap_pace_decile` | 0–9, where 0 = fastest 10% of clean laps for the track |
| `latg_peak_offset_z` | per-corner robust z-score of `latg_peak_offset_m` (line-position outlier) |
| `entry_speed_z` | per-corner robust z-score of `entry_speed_mph` |
| `gps_drift_mag_m` | `sqrt(drift_lat² + drift_lon²)` |
| `neighborhood_offset_max_m` | max `track_dist_offset_max_m` over this corner + its two neighbors (catches glitch spillover) |
| `transit_reliable` | bool — passes STANDARD: drift disagreement ≤ 20 m AND transit offset ≤ 40 m AND neighborhood offset ≤ 40 m |
| `lap_reliable` | bool — every transit on this (session, lap) is `transit_reliable` |

See [ARCHITECTURE.md](ARCHITECTURE.md) "Quality flags" for how to use these:
**filter spatial/line/apex analysis by `transit_reliable`; trust kinematic
channels (speed/throttle/brake/lat-G) even on unreliable laps** — OBD data is
unaffected by GPS issues.

### `data/corpus/<track>_corners.parquet`

The concatenation of every session's `corners.parquet` (same schema), built by
`build_corpus`. This is the cross-session query surface.

---

## Common operations

**Re-normalize a single session** (e.g. after fixing its raw CSV):

```powershell
python -m lap_analyzer.cli.normalize "data/raw/ridge/Log-20260517-100304 ....csv" --track ridge --force
```

**Re-label and rebuild after adding new sessions:**

```powershell
$env:PYTHONPATH = "."
python -m lap_analyzer.cli.normalize --track ridge --all
python -m lap_analyzer.cli.label_corners --track ridge
python -m lap_analyzer.cli.flag_quality --track ridge
python -m lap_analyzer.cli.build_corpus --track ridge
```

**Full rebuild including the centerline** (after a corner-box edit or many new
laps — see [NEW-TRACK.md](NEW-TRACK.md) for the full bootstrap):

```powershell
python -m lap_analyzer.cli.normalize --track ridge --all
python -m lap_analyzer.cli.label_corners --track ridge      # first pass (uses old centerline)
python -m lap_analyzer.cli.build_centerline --track ridge
python -m lap_analyzer.cli.install_centerline --track ridge
python -m lap_analyzer.cli.label_corners --track ridge      # second pass (uses new centerline)
python -m lap_analyzer.cli.flag_quality --track ridge
python -m lap_analyzer.cli.build_corpus --track ridge
```

**Annotate sessions** via `data/notes/<track>.json` — a flat map of
`session_id → annotation`:

```json
{
  "20251017-092131": {"exclude": "wet"},
  "20250518-100457": {"flag": "gps_unreliable", "reason": "drift median ~15m, max 33m"},
  "20260516-114331": {"reference": true, "reason": "instructor lap — keep on disk, exclude from corpus"}
}
```

- `exclude` — dropped at normalize time; any prior output is deleted so the
  corpus stays clean. Use for wet, off-pace, single-lap, or mechanical-issue
  sessions.
- `flag: "gps_unreliable"` — kept in the data but excluded from the centerline
  build and typically filtered out of spatial analysis. Kinematic channels stay
  usable.
- `reference: true` — fully normalized and labeled, but excluded from the corpus
  and every corpus-wide stat (pace deciles, per-corner medians) so it never
  shifts the user's own reference. Stays on disk for side-by-side comparison.
