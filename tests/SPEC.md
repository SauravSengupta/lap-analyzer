# lap_analyzer behavioral spec (clean-room test source)

This is the ONLY source of truth about behavior for the test suite. It is
derived from `docs/ARCHITECTURE.md` (design intent) and `docs/PIPELINE.md`
(schema and rounding). **Test authors: do not read `lap_analyzer/` source.**

## Conventions

- **Units / rounding:** distances and speeds are rounded to 2 dp; g-forces and
  throttle to 3 dp. Use `pytest.approx(..., abs=0.01)` for 2-dp fields and
  `abs=0.001` for 3-dp fields unless an entry says otherwise.
- **Prefer property/contract/invariant tests** over golden exact-value asserts.
  Golden asserts only where THIS spec states an exact value, and sparingly.
- **Cite the spec** in a one-line comment on every test
  (e.g. `# SPEC: normalize.normalize_dataframe — lat_g sign`).
- **KNOWN-LIMITATION** items must be `@pytest.mark.xfail(reason=..., strict=False)`
  or skipped — never asserted as fixed.
- **Do not reverse-engineer expected numbers from sample data.** You may load
  files under `data/samples/` to drive integration tests, but assert against the
  documented schema/invariants here, not against values you happen to observe.
- **Fixtures** (in `tests/conftest.py`): `sample_data_root` (sets `DATA_ROOT` to
  the committed bundle), `make_trackaddict_csv(columns, *, session_id=...,
  vehicle=..., end_point=..., filename=...)`, `make_lap_samples(n=200, **overrides)`.
- The sample bundle has tracks `ridge` and `pir`, 3 real sessions each plus a
  `_synthetic_centerline` session, a corpus, centerlines, and notes.

---

## config

Import: `from lap_analyzer.config import data_root, raw_dir, sessions_dir, corpus_dir, tracks_dir`.

- **Interface:** all are zero-arg except `raw_dir(track)` and `sessions_dir(track)`.
  All return `pathlib.Path`.
- **Contract:** `data_root()` resolves `DATA_ROOT` (env var, default `"data"`)
  relative to the repo root. `raw_dir(t)` = `data_root()/raw/<t>`; `sessions_dir(t)`
  = `data_root()/sessions/<t>`; `corpus_dir()` = `data_root()/corpus`.
- **Invariants:**
  - `data_root()` reads the env var **on every call** — changing `DATA_ROOT`
    between calls changes the result (no caching).
  - An **absolute** `DATA_ROOT` is honored as-is (wins over the repo-root join).
  - `tracks_dir()` is **independent of `DATA_ROOT`** — always repo-root `/tracks`.
  - All returned paths are absolute.

---

## schemas.SessionMeta

Import: `from lap_analyzer.schemas import SessionMeta`. (pydantic v2 BaseModel.)

- **Required fields:** `session_id` (str), `track` (str), `vehicle` (str),
  `date_utc` (datetime), `n_laps` (int), `n_clean_laps` (int), `sample_rate_hz`
  (float), `duration_s` (float), `raw_csv_path` (str).
- **`has_obd`** (bool) defaults to `True` — `False` for GPS-only sessions. The
  default keeps pre-existing `meta.json` files (written before this field) valid.
- **Optional fields (default `None`):** `best_lap`, `best_lap_time_s`,
  `throttle_max_observed`, `speed_max_obd_mph`, `rpm_max`, `coolant_min_f`,
  `coolant_max_f`, `iat_first_f`, `iat_max_f`, `trackaddict_start_finish` (dict).
  The OBD-derived ones (`throttle_max_observed`, `speed_max_obd_mph`, `rpm_max`)
  are `None` for GPS-only sessions.
- **`trackaddict_split_points`** defaults to `[]` (list of dict).
- **Invariants:** constructing with only the required fields succeeds and leaves
  optionals at their defaults; `model_dump_json()` → `model_validate_json()`
  round-trips; omitting a required field raises a pydantic `ValidationError`.

---

## normalize.session_id_from_filename

Import: `from lap_analyzer.normalize import session_id_from_filename`.
Signature: `(path) -> str`.

- **Contract:** extracts `YYYYMMDD-HHMMSS` from a filename matching `Log-<id>`
  (regex search anywhere in the basename).
- **Examples:** `"Log-20260517-100304 Ridge.csv"` → `"20260517-100304"`.
- **Edge case:** a name with no `Log-<8digits>-<6digits>` pattern raises
  `ValueError`.

## normalize._parse_coords

Import: `from lap_analyzer.normalize import _parse_coords`.
Signature: `(s: str) -> dict | None`.

- **Contract:** parses `"<lat>, <long>"`, ignoring anything after an `@` and a
  trailing comma. Returns `{"lat": float, "long": float}`.
- **Edge cases:** fewer than 2 comma-parts → `None`; a part that isn't a float
  → `None`. `"45.5, -122.6 @ 2026..."` → `{"lat": 45.5, "long": -122.6}`.

## normalize.parse_metadata_header

Import: `from lap_analyzer.normalize import parse_metadata_header`.
Signature: `(path) -> dict`.

- **Contract:** reads only the leading `#`-prefixed comment block (stops at the
  first non-`#` line). Recognizes `# Vehicle: X` → `meta["vehicle"]=X`,
  `# End Point: <lat>, <long> @ ...` → `meta["start_finish"]={"lat","long"}`,
  `# Split Point N: <lat>, <long> @ ...` → appends `{"index": N, "lat", "long"}`
  to `meta["split_points"]`.
- **Invariants:** the returned dict always carries `raw_csv_path` (str path) and
  a `split_points` list (possibly empty); unrecognized comment lines are ignored.

## normalize.normalize_dataframe

Import: `from lap_analyzer.normalize import normalize_dataframe`.
Signature: `(raw: pd.DataFrame, session_id: str, canonical_lap_length_m: float | None = None) -> tuple[pd.DataFrame, dict]`.
The `raw` frame is assumed already column-renamed to canonical names (as
`read_csv` does): `t, lap, lat, long, speed_mph, speed_mph_gps, lat_g, long_g,
brake, rpm, throttle_raw, coolant_f, iat_f`, etc. **The input must contain all 6
canonical OBD channels — `rpm, speed_mph, throttle_raw, coolant_f, iat_f,
manifold_psi` — because they are forward/back-filled together. `manifold_psi` is
required on input even though it is dropped from the output.**

- **Contract (PIPELINE.md):** returns `(out_df, {"throttle_max_observed": <float>})`
  where `out_df` has exactly the documented `samples.parquet` columns
  (`session_id, t, lap, dist_m, dist_lap_m, speed_mph, speed_mph_gps,
  throttle_norm, brake, rpm, lat_g, long_g, coolant_f, iat_f, lat, long,
  altitude_m, gps_accuracy_m`).
- **Invariants (ARCHITECTURE / axis note):**
  - **`lat_g` is negated:** canonical `lat_g == -raw["lat_g"]`. A raw left-turn
    (raw > 0) becomes negative; **positive `lat_g` = right turn.**
  - **`long_g` is negated:** canonical `long_g == -raw["long_g"]`. Raw braking
    (raw > 0) becomes negative; **positive `long_g` = acceleration.**
  - **OBD fill:** the 6 OBD channels are forward- then back-filled — given ≥1
    non-null value per channel, the output has no NaN in those columns.
  - **`throttle_norm` = `throttle_raw / max(throttle_raw)`**, in `[0, 1]`, max
    → 1.0. If `max(throttle_raw) == 0`, `throttle_norm` is `0.0` everywhere (no
    division by zero).
  - **`dist_m`** is trapezoidal integration of `speed·0.44704` over `t`, starting
    at 0; **monotonic non-decreasing** for non-negative speeds. The integrated
    speed is `speed_mph` when it has any non-null value, else `speed_mph_gps`
    (GPS-only / OBD-dropout sessions).
  - **`throttle_norm`** is NaN everywhere when `throttle_raw` is all-NaN
    (GPS-only sessions) — distinct from the all-zero-throttle case, which is 0.0.
  - **`dist_lap_m`** starts at 0 at each lap's first sample. With
    `canonical_lap_length_m` set, each lap's `max(dist_lap_m)` is rescaled to
    exactly that value (per-lap). With it `None`, no rescale.
  - dtypes: `lap` int32, `brake` int8, `rpm` int32.
- **Edge case:** a single-row frame yields `dist_m == 0` (no integration step).
- **Tolerances:** speeds/dist 2 dp tolerance is fine; `lat_g`/`long_g`
  exact-negation can be asserted exactly.

## normalize.compute_lap_times

Import: `from lap_analyzer.normalize import compute_lap_times`.
Signature: `(df: pd.DataFrame) -> pd.DataFrame` with columns `lap, lap_time_s`.

- **Contract:** for laps in ascending order, each lap's time is `next_lap_start_t
  − this_lap_start_t`. The **final lap is treated as an incomplete in-lap**: its
  time is `last_sample_t − this_lap_start_t`. Rounded to 3 dp.
- **Invariant:** one row per distinct `lap`; a synthetic frame with known per-lap
  start times yields the documented differences (e.g. laps starting at t=0,100,
  205 with last sample at 305 → times 100.0, 105.0, 100.0).

## normalize.lap_summary

Import: `from lap_analyzer.normalize import lap_summary`.
Signature: `(df: pd.DataFrame, lap_times: pd.DataFrame) -> pd.DataFrame`.

- **Contract:** one row per lap with the documented `laps.csv` columns.
- **Invariants (cleanness is shape-only):**
  - **`is_clean` is `False` only for the first and last lap.** `clean_reason` is
    `"warmup"` for the first lap, `"cooldown"` for the last, `""` otherwise.
    `is_clean == (clean_reason == "")`. (It is NOT a quality filter.)
  - `pct_wot` = fraction of samples with `throttle_norm ≥ 0.95`.
  - `pct_braking` = fraction with `brake == 1`.
  - `max_decel_g` = `max(−long_g)` (most-negative long_g surfaces as a positive
    decel magnitude); `max_accel_g` = `max(long_g)`.
  - `lap_dist_m` = `max(dist_lap_m)` for the lap.
- See KNOWN-LIMITATION note under "cleanness" — do not treat `is_clean` as a
  data-quality signal.

## normalize.reference_session_ids

Import: `from lap_analyzer.normalize import reference_session_ids`.
Signature: `(track: str) -> set[str]`.

- **Contract:** reads `data/notes/<track>.json` and returns the set of session
  ids whose annotation has a truthy `"reference"` key. Missing notes file →
  empty set. (Use `sample_data_root`; assert it returns a `set`, and that a
  session NOT marked reference is absent.)

## normalize.normalize_session (OBD-dropout: GPS-only ingest)

Import: `from lap_analyzer.normalize import normalize_session`.
Signature: `normalize_session(csv_path, track, out_dir) -> SessionMeta` — `out_dir`
is a required positional (the directory the `<session_id>/` output folder is
created under).

- **KNOWN behavior — OBD dropout (~12% of real sessions log without OBD):** if the
  raw CSV is missing any of the 6 OBD columns, the session is ingested as
  **GPS-only** rather than rejected. `normalize_session` writes its output
  (`samples.parquet`, `laps.csv`, `meta.json`) and the returned `SessionMeta` has
  `has_obd == False`. Build a CSV with `make_trackaddict_csv(columns=...)` that
  OMITS the OBD columns and assert: the session dir is written, `meta.has_obd` is
  `False`, and lap times are present in `laps.csv`.
- **GPS-only sample columns:** the 6 OBD channels (`rpm, speed_mph, throttle_raw,
  coolant_f, iat_f` — and the input-only `manifold_psi`) are all-NaN in the
  output samples. `dist_m` is integrated from `speed_mph_gps` instead of the
  absent OBD speed, so it is still monotonic non-decreasing. `brake` survives
  (it is the accelerometer-derived `Brake (calculated)` column, present in
  GPS-only CSVs).
- **GPS-only meta:** OBD-derived `SessionMeta` fields are `None` when
  `has_obd == False`: `speed_max_obd_mph`, `rpm_max`, `throttle_max_observed`,
  `coolant_min_f`, `coolant_max_f`, `iat_first_f`, `iat_max_f`.
- **Normal sessions:** a CSV containing all 6 OBD columns yields
  `meta.has_obd == True` with those fields populated (regression — unchanged).

Note: `MissingOBDError` is no longer raised by `normalize_session` and has been
removed from the public API.

---

## labeler.load_track

Import: `from lap_analyzer.labeler import load_track`.
Signature: `(track_id: str) -> Track` (dataclass).

- **Contract:** parses `tracks/<id>.json`. `Track` has `track_id, name,
  direction, lap_length_m` (from `lap_length_internal_m`), `corners` (list of
  `Corner`), `calibration_anchors`, `reference_session_id`, `reference_lap`.
- **Invariants:**
  - `corners` are **sorted ascending by `start_m`**.
  - Each `CalibrationAnchor` is populated with its corner's `start_m`/`end_m`;
    anchors whose `corner_id` has no matching corner are dropped.
  - `reference_session_id`/`reference_lap` come from the JSON `reference_lap`
    block (`""`/`0` if absent).
- Use the real `tracks/ridge.json` / `tracks/pir.json` (read via `tracks_dir()`,
  not `data/samples`).

## labeler.ReferenceIndex.project / .lookup

Construct a `ReferenceIndex` via `build_reference_index(track)` (needs
`sample_data_root` so the reference session is found), or test `.project`
arithmetic directly on a hand-built instance if feasible.

- **`.project(lat, lon)`:** returns an `(N, 2)` array of local meters about the
  index center: `x = (lon − center_lon)·m_per_deg_lon`, `y = (lat − center_lat)·
  m_per_deg_lat`. The center maps to `(0, 0)`.
- **`.lookup(lat, lon)`:** returns `(track_dist_m, dist_to_ref_m)` for the nearest
  reference sample. A point coincident with a reference sample has
  `dist_to_ref_m ≈ 0` and returns that sample's `dist_lap_m`.

## labeler.compute_lap_drift

Import: `from lap_analyzer.labeler import compute_lap_drift`.
Signature: `(lap_samples, anchors, m_per_deg_lat, m_per_deg_lon) -> (drift_lat_m | None, drift_lon_m | None, n_used: int, disagreement_m | None)`.

- **Contract (ARCHITECTURE §2 — the key intent):** for each anchor, find the
  closest sample within a **100 m** radius of the anchor's `(ref_lat, ref_long)`;
  the per-anchor offset is `(sample − anchor)` in meters. The lap's drift is the
  **median** of those offsets.
- **Invariants:**
  - **Median aggregation is robust to a single bad anchor:** with several good
    anchors agreeing and one wildly-off anchor, the returned drift stays at the
    good cluster's median (the outlier does not move it). This is the property to
    test — construct anchors + a synthetic lap so most offsets agree and one is
    far off, and assert the median (not the mean) is returned.
  - `n_used == 1` → `disagreement == 0.0`.
  - No anchor within radius → `(None, None, 0, None)`.
  - With ≥2 anchors, `disagreement_m = sqrt(mean(per-anchor deviations² from the
    median))` ≥ 0.
- Pass `m_per_deg_lat = 111_132.0` and `m_per_deg_lon =
  111_132.0·cos(radians(center_lat))` (the values `build_reference_index`
  computes) so meters are consistent.

## labeler.label_samples — corner segmentation

Import: `from lap_analyzer.labeler import label_samples` (needs a `Track` and a
`ReferenceIndex`; or test the segmentation contract via a small session under
`sample_data_root`). Adds `track_dist_m, track_dist_offset_m`, drift columns, and
`corner`.

- **Contract (segmentation):** the lap is partitioned by `track_dist_m` into the
  corner boxes `[start_m, end_m]` (labeled by corner id) and the gaps between
  them (straights). A straight is labeled **`S_post_<id-of-preceding-corner>`**.
- **Invariant (front-straight wrap):** the segment before the first corner is
  labeled `S_post_<last-corner-id>` — the start/finish straight is attributed to
  the final corner, wrapping around. Every sample gets exactly one `corner` label.

## labeler.build_corner_transit

Import: `from lap_analyzer.labeler import build_corner_transit`.
Signature: `(lap_samples: pd.DataFrame, corner: Corner) -> dict | None`. Build a
`Corner` via the dataclass (`from lap_analyzer.labeler import Corner`) and a lap
via `make_lap_samples`, overriding `track_dist_m`, `speed_mph`, `lat_g`, etc.
**`Corner` is a plain dataclass with all 8 fields required (no defaults):
`id, name, start_m, end_m, apex_m, secondary_apex_m, type, notes`** — pass them
all (`name`/`notes` may be `None`, `secondary_apex_m` `None` or a float).

- **Edge case:** fewer than 3 samples inside `[start_m, end_m]` → returns `None`.
- **Invariants (apex metrics — ARCHITECTURE §4):**
  - `apex_dist_offset_m == round(min_speed_dist_m − corner.apex_m, 2)`.
  - `latg_peak_offset_m == round(latg_peak_dist_m − corner.apex_m, 2)`.
  - `latg_peak_dist_m` is the `track_dist_m` of the **max |lat_g|** sample inside
    the corner; `min_speed_dist_m` is the `track_dist_m` of the min-speed sample.
  - input-timing offsets (`brake_on_dist_m`, `throttle_lift_dist_m`,
    `wot_dist_m`, …) are **offsets from `corner.start_m`** (value − start_m), or
    `None` when the event never occurs in the window.
  - `max_lat_g` is reported as the max of **|lat_g|** (sign-agnostic magnitude).
  - `peak_brake` is an int (max of the 0/1 brake channel).
- Construct a lap with a known V-shaped speed profile and a known lat_g peak to
  assert the offsets resolve to the expected signed values.
- **GPS-only (OBD-dropout) sessions:**
  - The transit dict carries `obd_present` (bool) and `speed_source`
    (`"obd"`/`"gps"`). `obd_present` is `True` when the lap has any non-null
    `speed_mph` (OBD), else `False`.
  - **Speed metrics fall back to GPS:** `entry/exit/min/max/apex/secondary` speeds
    are computed from `speed_mph` when present, else from `speed_mph_gps`. So a
    GPS-only lap still gets a real `min_speed_mph` (apex speed) from GPS.
  - **Throttle/WOT metrics are NaN** when `throttle_norm` is all-NaN (no OBD):
    `peak_throttle_norm`, `mean_throttle_norm`, `pct_wot`, `throttle_lift_dist_m`,
    `throttle_return_dist_m`, `wot_dist_m`.
  - **Brake metrics survive** (the brake channel is accelerometer-derived):
    `peak_brake`, `pct_braking`, `brake_on_dist_m`, `brake_off_dist_m`.

## labeler.build_session_corners

Import: `from lap_analyzer.labeler import build_session_corners`. Signature:
`build_session_corners(samples, laps, track, session_id, date) -> pd.DataFrame`.
**It operates on already-labeled `samples` (a DataFrame carrying `track_dist_m`,
`corner`, and `gps_drift_*` columns) plus the `laps` DataFrame — it does NOT read
a session directory or take a `ReferenceIndex`.** Drive it from a
`sample_data_root` session, whose `samples.parquet` is already labeled:
`build_session_corners(pd.read_parquet(.../samples.parquet),
pd.read_csv(.../laps.csv), track, "<sid>", "<YYYY-MM-DD>")`.

- **Contract:** emits one row per (clean lap × corner). Only laps with
  `is_clean == True` are processed; laps with < 50 samples are skipped; the
  per-lap drift columns are pulled from the lap's first sample. Assert
  clean-lap-only and the presence of `session_id, date, lap, corner_id` + drift
  columns.

---

## quality.compute_quality

Import: `from lap_analyzer.quality import compute_quality`. Signature:
`(track: str) -> pd.DataFrame`. **Integration tier** — run with `sample_data_root`.

- **Contract / invariants (ARCHITECTURE "Quality flags"):**
  - `lap_pace_decile` ∈ `0..9`, computed by `qcut` on clean laps' `lap_time_s`;
    **0 = fastest 10%.** (Assert the range and that 0 corresponds to the smallest
    lap times, not exact bin membership.)
  - per-corner robust z-scores `latg_peak_offset_z`, `entry_speed_z` use
    `(x − median) / (MAD · 1.4826)`; a corner with MAD ≤ 0.1 yields NaN z (no
    divide-by-near-zero).
  - **`entry_speed_z` baseline excludes GPS-only sessions.** The per-corner median
    and MAD for `entry_speed_z` are computed from OBD sessions only (`obd_present`
    True). GPS-only rows are *scored against* that OBD baseline (they still get an
    `entry_speed_z`) but do not *define* it — so a GPS-only session's GPS-sourced
    entry speeds (which read ~1–2 mph low) cannot shift the OBD references. The
    lat-G-based `latg_peak_offset_z` baseline uses all sessions (lat-G is present
    regardless of OBD). `lap_pace_decile` includes GPS-only laps (lap times valid).
  - `gps_drift_mag_m == hypot(gps_drift_lat_m, gps_drift_lon_m)`.
  - `neighborhood_offset_max_m` = max of this corner's `track_dist_offset_max_m`
    and those of its two corner-sequence neighbors (wraps at start/finish).
  - **`transit_reliable == (gps_drift_disagreement_m ≤ 20) AND
    (track_dist_offset_max_m ≤ 40) AND (neighborhood_offset_max_m ≤ 40)`** — assert
    this boolean relationship holds row-wise on the returned frame.
  - **`lap_reliable`** is `True` iff **every** transit on that `(session_id, lap)`
    is `transit_reliable` (group-min). Assert: no row has `lap_reliable=True`
    while any sibling row on the same lap has `transit_reliable=False`.
  - Reference sessions are excluded from the corpus-wide stats.
- Thresholds live in `quality` as `DRIFT_DISAGREEMENT_LIMIT_M=20`,
  `TRANSIT_OFFSET_LIMIT_M=40`, `NEIGHBORHOOD_OFFSET_LIMIT_M=40` (importable).

---

## corpus.build_corpus

Import: `from lap_analyzer.corpus import build_corpus`. Signature:
`(track: str) -> dict`. **Integration tier** — but note it WRITES to
`corpus_dir()`; under `sample_data_root` that would overwrite the committed
sample corpus, so point `DATA_ROOT` at a **copy** in `tmp_path` (copy the
relevant `sessions/<track>` tree) before calling, OR assert only the
no-input edge case.

- **Edge case:** raises `FileNotFoundError` when no `corners.parquet` exists under
  the track's sessions dir (e.g. a fresh `tmp_path` `DATA_ROOT`).
- **Invariants:** returns a dict with keys `out_path, n_sessions, n_transits,
  n_laps`; reference sessions are excluded (so `n_sessions` counts non-reference
  session dirs that have a `corners.parquet`); `n_laps` = distinct
  `(session_id, lap)` pairs.

---

## analysis.section_bounds

Import: `from lap_analyzer.analysis import section_bounds`. Signature:
`(track_def: dict, pre_m=50.0, post_cap_m=250.0) -> dict[str, tuple[float,float]]`.
`track_def` is the parsed `tracks/<t>.json` dict (load via `tracks_dir()`), or a
small hand-built dict `{"corners": [{"id","start_m","end_m"}...],
"lap_length_internal_m": ...}`.

- **Contract:** for each corner, `start = max(0, corner.start_m − pre_m)`,
  `end = min(next_corner.start_m, corner.end_m + post_cap_m)`. For the last
  corner, the "next start" is `lap_length_internal_m` (fallback
  `last.end_m + 200`).
- **Invariant:** the window always contains the corner box; `start ≥ 0`; `end`
  never spills past the next corner's start.

## analysis._first_crossing_t

Import: `from lap_analyzer.analysis import _first_crossing_t`. Signature:
`(xs: np.ndarray, ts: np.ndarray, target: float, after_t: float | None = None) -> float | None`.

- **Contract:** time of the first **upward** crossing of `target`
  (`xs[i] < target ≤ xs[i+1]`), linearly interpolated between samples.
- **Invariants:**
  - no upward crossing → `None`.
  - `after_t` restricts to crossings at/after that time.
  - exact crossing on a sample / zero-width step (`xs[i+1] == xs[i]`) → returns
    `ts[i]` (no division by zero).
  - For `xs=[0,10]`, `ts=[0,1]`, `target=5` → `0.5`.

## analysis.find_gear_bands

Import: `from lap_analyzer.analysis import find_gear_bands`. Signature:
`(ratios, ratio_lo=30, ratio_hi=260, n_bins=230, min_peak_frac=0.005, min_separation=8.0) -> np.ndarray`.

- **Contract:** clusters rpm/speed ratios into gear-band centers. Histograms the
  in-range ratios, keeps strict local maxima at least `min_peak_frac × tallest
  bin`, merges maxima closer than `min_separation` (keeping the first), refines
  each to the mean of nearby ratios. Returns **sorted ascending** band centers.
- **Invariants:**
  - empty input (or none in `[lo, hi]`) → empty array.
  - a synthetic ratio distribution with 3 well-separated tight clusters returns
    ~3 ascending centers, each near its cluster mean (assert count and
    approximate locations, not exact floats).
- **Gear semantics (read with `derive_gear`):** smallest ratio = **tallest**
  gear; largest ratio = **shortest** (lowest) gear.

## analysis._confirm_runs

Import: `from lap_analyzer.analysis import _confirm_runs`. Signature:
`(raw: np.ndarray, n_dwell: int) -> np.ndarray`.

- **Contract:** any maximal run of equal **non-NaN** values shorter than
  `n_dwell` is set to NaN; NaN runs are never modified; non-NaN runs ≥ `n_dwell`
  are preserved.
- **Invariant:** e.g. `[1,1,1,2,1,1,1]` with `n_dwell=2` → the single `2`
  becomes NaN, the `1`-runs survive.

## analysis.derive_gear

Import: `from lap_analyzer.analysis import derive_gear`. Signature:
`(samples: pd.DataFrame, bands: np.ndarray, min_dwell_s=0.3) -> pd.Series`.
Needs `t, rpm, speed_mph` columns (use `make_lap_samples`).

- **Contract:** snaps each qualifying sample (`rpm > 1200` AND `speed_mph > 8`)
  to the nearest band, suppresses sub-dwell runs (clutch-slip transients) and
  forward-fills the held gear, then re-NaNs samples with `rpm == 0`.
- **Invariants (the key intent — gear index orientation):**
  - **Gear index 0 = the shortest / lowest gear (the highest rpm/speed ratio).
    Higher index = taller gear.** (Implementation: `(bands.size − 1) − argmin|ratio
    − bands|` over ratio-ascending bands. Real gear ≈ index + 1.) Construct a lap
    whose ratio sits clearly in the highest-ratio band and assert gear index 0;
    one clearly in the lowest-ratio band → index `len(bands) − 1`.
  - samples with `rpm == 0` → NaN gear.
  - samples that don't qualify (`rpm ≤ 1200` or `speed ≤ 8`) get no fresh
    assignment (NaN unless filled from a held gear).
  - the returned Series is aligned to `samples.index`.

## analysis.span_time

Import: `from lap_analyzer.analysis import span_time`. Signature:
`(lap_samples, dist_a, dist_b, centerline, frame=None, half_width_m=40.0,
seed_window_m=120.0) -> tuple[float, float] | None`. Needs
`lat, long, t, track_dist_m, dist_lap_m`. See `docs/GPS_TRUST.md`.

- **Contract:** returns `(section_time_s, timing_gap_s)`. A gate is a line segment
  laid across the track (perpendicular to the centerline) at each of
  `dist_a`/`dist_b`. `timing_gap_s` is the GPS-timing-confidence gap (max over the
  two gates) from `crossing_gap_s` — the elapsed time between the good GPS fixes
  bracketing a gate.
- **Invariants:**
  - `None` if either gate isn't crossed (the path stayed beyond the gate's
    `±half_width_m`, or `t_b <= t_a`), or the OBD range can't supply the fallback.
  - **Gate tangent is a smoothed local fit, not a 2-sample tangent.** The gate
    perpendicular at each `dist_a`/`dist_b` comes from a σ=10 m Gaussian-weighted
    linear regression of centerline position over a ±20 m window, so it is immune
    to the centerline's transverse noise. On a straight track carrying a 5 m /
    18 m-wavelength transverse oscillation the gate stays within 3° of true — a
    2-sample tangent there rotates tens of degrees and converts a lateral line
    offset into spurious crossing time.
  - Immune to lateral GPS/line offset: a wider line crossing the same gates
    returns the same time — genuine line-length variation is preserved.
  - **Reliable** (`timing_gap_s < CONFIDENCE_GAP_S`, default 0.4 s):
    `section_time_s` is the gate-crossing time (`t_b − t_a`).
  - **Not reliable** (coarse GPS / a teleport-punctured bracket): `section_time_s`
    is the OBD-anchored fallback — time between the `dist_lap_m` crossings of
    `dist_a` and `dist_b` (`_obd_anchored_time`). Surfaced/flagged, never silently
    dropped. (Replaced the earlier spatial `_gates_glitch_free` offset guard, which
    missed coarse-GPS mistiming and over-dropped clean fast lines.)
  - a clean synthetic lap driving down the centerline returns the true elapsed
    time between the two gate positions with a sub-threshold gap.

`section_times` / `range_section_times` share `crossing_gap_s` and emit
`timing_gap_s` + `timing_reliable` columns alongside `section_time_s`.

## analysis.section_range_bounds

Import: `from lap_analyzer.analysis import section_range_bounds`. Signature:
`(track_def, from_id, to_id, pre_m=50.0, post_cap_m=250.0) -> tuple[float,float]`.

- **Contract:** window spanning corners `from_id..to_id` inclusive: `a = max(0,
  from.start_m − pre_m)`, `b = min(start of corner after to_id, to.end_m +
  post_cap_m)`.
- **Invariants:**
  - `from_id == to_id` returns exactly `section_bounds(track_def)[from_id]`.
  - raises `ValueError` when `to_id` precedes `from_id` in track order.

## analysis.classify_t8_section

Import: `from lap_analyzer.analysis import classify_t8_section`. Signature:
`(samples, track_def, bands, min_dwell_s=0.3) -> dict`. Uses corners T8/T10 and
T8.apex_m; build a `track_def` dict with those corners + a `make_lap_samples`
window with `track_dist_m`, `rpm`, `speed_mph`.

- **Contract:** returns `{label, excluded_reason, downshift_dist_m,
  t8_apex_gear}`. Scans `[T8.start − 50, T10.end]`.
- **Invariants:**
  - `< 10` samples in the window → `label="excluded"`, `excluded_reason="ambiguous"`.
  - all `rpm == 0` in the window → `excluded_reason="no-obd"`.
  - zero downshifts → `label="no-downshift"`, `excluded_reason=None`.
  - exactly one downshift at `loc ≤ T8.apex_m` → `label="downshift"`.
  - exactly one downshift past `T8.apex_m` (still inside the window) →
    `excluded_reason="traffic"`.
  - more than one downshift → excluded.
- **KNOWN-LIMITATION** (memory: T8–T11 traffic contamination): the cohort filter
  doesn't catch traffic lifts in general; do not assert the classifier perfectly
  separates traffic from clean downshifts on real data — test only the documented
  rule above on synthetic windows.

## analysis.top_decile_laps / lap_summary / lap_index

Import from `lap_analyzer.analysis`. **Integration tier** with `sample_data_root`
(`load_corpus(track)` first).

- `top_decile_laps(corpus) -> set[(session_id, int lap)]`: exactly the
  `(session, lap)` pairs whose rows have `lap_pace_decile == 0`.
- `lap_summary(corpus, session_id, lap) -> dict`: empty `{}` when that lap isn't
  in the corpus; otherwise includes `n_transits`, `n_reliable_transits`,
  `lap_reliable`, `max_speed_mph`.
- `lap_index(corpus, track=None) -> pd.DataFrame`: one row per `(session_id, lap)`;
  when `track` is given, joins `lap_time_s`/`is_clean` from each session's laps.csv.

---

## corners.extract_lap_candidates

Import: `from lap_analyzer.corners import extract_lap_candidates`. Signature:
`(lap_df: pd.DataFrame, sample_rate_hz: float) -> list[dict]`. Needs
`lat_g, speed_mph, dist_lap_m, t, long_g, throttle_norm, lat, long`.

- **Contract:** smooths lat_g, finds |lat_g| peaks (`find_peaks`, height 0.4 g,
  min spacing 3 s), and emits one candidate dict per peak with peak/entry/exit
  distances and per-candidate offsets.
- **Invariants:**
  - `< 50` samples → `[]`.
  - `direction` is `"right"` when the **smoothed** peak lat_g is `> 0`, else
    `"left"` (consistent with the positive=right convention).
  - offsets (`brake_on_offset_m`, etc.) are relative to the candidate's
    `entry_dist_m`, or `None` when the event doesn't occur.
  - **GPS-only sessions** (OBD dropout, all-NaN `speed_mph`) fall back to
    `speed_mph_gps` for the min-speed apex pick (mirrors
    `labeler.build_corner_transit`), so a candidate is still emitted with a
    finite `min_speed_mph` instead of crashing on an all-NA `idxmin`. Each
    candidate records that provenance: `obd_present` (bool) and `speed_source`
    (`"obd"`/`"gps"`), like the transit dict.
  - **brake/throttle-lift onset is detected over a lookback window extending
    `LOOKBACK_M` (150 m) before `entry_dist_m`** (mirrors
    `labeler.build_corner_transit`), so braking that begins before turn-in is
    captured — `brake_on_offset_m` is **negative** when braking starts before the
    lat-G entry. (Throttle-return is still measured post-apex.)
- Construct a lap with one clear right-hand lat_g hump (e.g. a Gaussian bump > 0.4
  g) over ≥ 50 samples and assert a single `direction == "right"` candidate.

## corners.extract_session_candidates

Import: `from lap_analyzer.corners import extract_session_candidates`.
Signature: `(session_dir: Path) -> pd.DataFrame`. **Integration tier** — point at
a `sample_data_root` session dir.

- **Invariants:** skips warmup/cooldown laps (uses `clean_reason`); when no
  candidates are found returns an empty DataFrame whose columns equal
  `CANDIDATE_COLUMNS` (importable from the module).

---

## trajectory.Corridor / build_corridor

Import: `from lap_analyzer.trajectory import Corridor, build_corridor, load_corridor`.
The corridor is the per-track "asphalt ribbon" the trajectory layer reads: a
per-10m-bin clean-lap lateral envelope + the σ=10m-smoothed centerline field +
corpus signed curvature. Design:
`docs/superpowers/specs/2026-07-11-unified-gps-trust-trajectory-design.md`.

- **`Corridor`** (frozen dataclass): equal-length arrays `s_bin` (10m bin
  centers), `e_lo`/`e_hi` (signed lateral envelope, metres), `tx`/`ty` (unit
  smoothed tangent), `gx`/`gy` (smoothed position, TrackFrame metres),
  `kappa_signed` (per bin, `+` = right, GPS-free from `lat_g·g/v²`), and `meta`.
  - `bin_index(track_dist_m)` → bin indices clipped to range.
  - `lateral_offset(x, y, track_dist_m)` → signed offset of frame-XY points from
    the smoothed centerline, **`+` = left of travel**. Inputs must be
    drift-corrected (design R5).
- **`build_corridor(track)`** — from clean flying laps (≥100 samples, sane
  `dist_lap_m` rescale, no >40m GPS excursion):
  - **envelope**: per-bin p2/p98 of per-lap-per-bin median lateral offset, +2m
    pad, hard cap ±20m, built in **two EM-trim passes** (pass 2 drops the votes
    pass-1's envelope rejects — purges Mode-4 contamination without dragging the
    ribbon).
  - **`kappa_signed`**: corpus clean-lap median `lat_g·g/v²` per bin, **NaN-safe**
    (OBD-dropout laps have NaN `speed_mph`; excluded, never propagated).
  - **Invariants**: `e_lo ≤ e_hi`; `|e_lo|,|e_hi| ≤ 20`; `kappa_signed` all
    finite; `s_bin` increases by 10m.
- **EM-trim convergence** (judge requirement, asserted): on synthetic clean votes
  plus a Mode-4 contamination cluster (≈ −30m in some bins), the 2-pass envelope
  recovers the clean ribbon in the contaminated bins where the 1-pass envelope is
  dragged toward the cap.
- **`load_corridor` / save**: round-trips arrays + meta; raises when the persisted
  `calib_version` differs from the code's (forces a rebuild on schema change).
- Note: the corridor's lateral envelope reproduces the *validated prototype*
  corridor (95 strict-clean laps + 2-pass EM-trim), which is intentionally tighter
  than the design brief's broader per-corner [−13,+16]/[−14,+15] figures; the
  envelope's behavioural validation is the canonical-case discrimination (PR 2),
  not those numbers.

## trajectory.estimate_trajectory

Import: `from lap_analyzer.trajectory import estimate_trajectory, Trajectory`.
Signature: `(lap_samples, corridor, frame) -> Trajectory`. Estimates one lap's
along-track position on the canonical ruler with honest per-sample σ — a robust,
corridor-weighted, slope-bounded smooth of the offset evidence
`δ = track_dist_m − dist_lap_m`, blended toward the δ=0 (OBD-backbone) prior.

- **`Trajectory`**: `t`, `s_hat`, `sigma_m`, `delta_hat`, `v` (per time-sorted
  sample), `evidence` (bool mask of accepted GPS fixes), `status`
  (`ok`/`rescale_invalid`/`gps_backbone`), `dl` (per-sample odometer),
  `knots`/`knot_sigma`, and `checks` (structured audit records). Helpers:
  `time_at(s)` → `(t, σ_t)`, or **`None`** when `s` is outside
  `[s_hat.min, s_hat.max]` (no crossing → the caller emits `no_coverage`, never a
  clamped/fabricated time); `sigma_at(s)` inverts s→dl through `s_hat` before
  reading the knot σ (δ̂ is not constant); `v_at(t)`.
- **Invariants:**
  - `s_hat` is **monotone non-decreasing** (`= maximum.accumulate(dist_lap + δ̂)`),
    so a scalar ruler position has a unique crossing — ghost crossings die
    structurally.
  - `len(s_hat) == len(t) == len(sigma_m) == len(delta_hat)`.
  - a lap whose `dist_lap_m` rescale is invalid (`|median δ| > 150m`, e.g. a
    session's first/last lap) → `status='rescale_invalid'`, δ̂≡0 (prior only),
    nothing silently dropped.
  - no GPS evidence (all fixes masked, or an OBD-dropout GPS-only lap) → δ̂≡0 with
    the mid-lap prior σ; the zero-evidence limit reproduces the legacy
    OBD-anchored fallback (design R6). GPS-only laps carry `status='gps_backbone'`.
  - drift-corrected coords (design R5): subtracts `gps_drift_{lat,lon}_m` when
    present before computing lateral offsets.
- **Extra Trajectory field `e_lat`** (per time-sorted sample): the drift-corrected
  signed lateral offset from the smoothed centerline (metres, + = left of travel),
  or NaN where the lap has no usable `lat`/`long`. Consumed by the consistency net.

## trajectory.section_timing

Import: `from lap_analyzer.trajectory import section_timing, SectionTiming`.
Signature: `(traj: Trajectory, dist_a: float, dist_b: float, corridor: Corridor)
-> SectionTiming`. Times the section between ruler positions `dist_a` and `dist_b`
on one lap's estimate, and reports honest per-section σ with three physical σ-nets.

- **`SectionTiming`** fields: `time_s`, `sigma_s` (1σ in **seconds**), `t_a`, `t_b`,
  `driven_m` (odometer distance between the posterior crossings), `status`
  (`ok`/`no_coverage`/`rescale_invalid`), `tier` (`A`/`B`/`C`, derived from
  `sigma_s`), `rank_eligible` (bool), and `checks` (structured audit records, R12).
- **Always emits a row (design R10).** It never returns `None` and never raises on a
  missing crossing:
  - `dist_a`/`dist_b` outside the lap's `s_hat` range → `status='no_coverage'`,
    `time_s`/`sigma_s`/`driven_m` NaN, `tier='C'`, `rank_eligible=False`.
  - the lap's trajectory `status=='rescale_invalid'` → `status='rescale_invalid'`,
    a value is still computed (prior-only, OBD-anchored), `rank_eligible=False`.
  - otherwise `status='ok'`.
- **Value** comes from the SAME estimate as the confidence (design R1): `t_a`, `t_b`
  are the times where the monotone `s_hat` crosses `dist_a`, `dist_b`;
  `time_s = t_b - t_a`. A scalar ruler position crosses a monotone `s_hat` exactly
  once, so ghost crossings are structurally impossible.
- **R6 exact-equality:** when the lap has no accepted GPS evidence (δ̂≡0), the emitted
  `time_s` equals the legacy `analysis._obd_anchored_time(t, dist_lap_m, a, b)` to
  within 1e-6 s — the zero-evidence limit reproduces the OBD-anchored fallback.
- **σ is correlation-aware** (judge-mandated; independence is wrong-signed for
  Mode 4): `Var(T) = [σ_A² + σ_B² − 2ρσ_Aσ_B] / (v_A·v_B)` with `σ_A`/`σ_B` the
  posterior σ at the two crossings and `ρ = RHO_SECTION` a corpus-fitted constant.
  A common-mode (Mode-1) offset error partly cancels; the three nets below then
  **inflate** σ (never reject — design R2):
  - **driven-band net:** the per-section band `= (∫|κ| ds over [a,b])·6 + 7` metres
    (κ from the corridor, GPS-free); `excess = max(0, |driven_m − (b−a)| − band)`
    inflates σ by `excess / v̄` in quadrature. Evaluated at the **posterior**
    crossings, so a genuine tight line (in-band there) is untouched while on-ribbon
    odometer-inconsistent drift is demoted.
  - **line-length consistency net:** `resid = driven_m − [(b−a) + Σ κ_signed·e_left·Δs]`
    over the section (does the lap's own lateral line explain its odometer
    shortening?); `excess = max(0, |resid| − 2·CONSISTENCY_STD_M)` inflates σ by
    `excess / v̄`. `CONSISTENCY_STD_M = 6.6`. Always emitted as a check record.
  - **speed-consistency tripwire:** `|time_s − obd_anchored_time|` vs
    `(|δ_a|+|δ_b|)/v̄ + 0.2s` — a **diagnostic** check record only, never inflates.
- **Derived tiers (never stored booleans):** `A` if `sigma_s ≤ 0.10`, `B` if
  `≤ 0.30`, else `C`. A `gps_backbone` trajectory caps at `B` (reserved §9.3).
  `rank_eligible == (tier == 'A' and status == 'ok')`.
- **Structured checks (R12):** every row carries records `{name, value, threshold,
  pass}` for at least the driven-band, consistency, and speed-consistency nets, so a
  post-mortem can read *why* a row is/ isn't rankable straight from the object.

### Per-mode σ-calibration (design R7)

`scripts/gps_trust_calibration.py` synthesises randomised laps for each GPS failure
mode (`clean`, `teleport`, `drift`, `line_offset`, `gps_hold`, `lockup`, `gps_only`)
with a KNOWN true section time, and asserts the emitted σ honestly covers the error:
**|section-time error| < 2·σ̂ in ≥95% of draws, asserted separately for each mode** (a
global scale can hide Mode-4 under-coverage behind Mode-3 over-coverage). The
authoritative hard gate is ≥500 draws/mode on the real corridor (`run_calibration`);
`test_per_mode_calibration_within_2sigma` runs it on a self-contained SYNTHETIC
corridor so CI enforces the property without the (gitignored) corpus. The
`line_offset` mode doubles as the R4 discrimination check: a genuine in-corridor,
odometer-consistent tight line is recovered accurately (the estimator follows the
real δ), not shrunk toward the OBD backbone.

---

## fused_axis.compute_fused_dist

Import: `from lap_analyzer.fused_axis import compute_fused_dist`. Signature:
`(samples: pd.DataFrame, smooth_window=25) -> np.ndarray`. Needs `t, dist_lap_m,
track_dist_m`.

- **Contract:** `fused = dist_lap_m + lowpass(track_dist_m − dist_lap_m)` — OBD
  distance supplies the smooth monotone base; the low-frequency part of the GPS−OBD
  offset bends it onto the centerline.
- **Invariants:**
  - `< 2` rows → a copy of `dist_lap_m`.
  - **output is monotonic non-decreasing** by construction (cumulative max),
    even when `track_dist_m` jitters or steps backward.
  - output is aligned to the input row order (rows may be unsorted by `t`).
  - a sample whose `|track_dist_m − dist_lap_m| ≥ 50 m` (a teleport) does not drag
    the low-pass — its offset is interpolated over from good neighbors.
  - on clean data where `track_dist_m ≈ dist_lap_m`, fused ≈ `dist_lap_m`.

## fused_axis.glitch_runs

Import: `from lap_analyzer.fused_axis import glitch_runs, GLITCH_OFFSET_M`.
Signature: `(track_dist_m, dist_lap_m, fused, threshold_m=GLITCH_OFFSET_M,
merge_gap_m=0.0) -> list[tuple[float, float]]`. Parallel arrays in one
time-sorted order; `fused` is `compute_fused_dist` for those samples.

- **Contract:** one `(fused_lo, fused_hi)` span per contiguous run of samples with
  `|track_dist_m - dist_lap_m| >= threshold_m`, in fused coordinates.
- **Invariants:**
  - `[]` when no sample exceeds the threshold.
  - separate glitched runs -> separate spans; a single contiguous run -> one span.
  - `GLITCH_OFFSET_M == 50.0` (shared with compute_fused_dist's teleport mask).
  - runs whose fused gap is <= `merge_gap_m` coalesce into one span;
    `merge_gap_m=0` (default) never merges.

---

## centerline.build_centerline

Import: `from lap_analyzer.centerline import build_centerline`. Signature:
`build_centerline(sessions_dir, notes_path, lap_length_m, track_lat_deg, out_path,
protected_ranges=None) -> pd.DataFrame` (also writes the parquet to `out_path`).
`protected_ranges` is a list of `(start_m, end_m)` corner spans. **Integration
tier** — heavy; verify against a **copied** Ridge tree (it reads every session's
`samples.parquet`/`corners.parquet`/`laps.csv` and writes `out_path`, so copy
`sessions/ridge` + `notes` + `corpus` into a tmp `DATA_ROOT` first). Asserting
exact coordinates is out of scope; assert structure and the protected-range
intent.

- **Output schema:** a DataFrame / parquet with columns `track_dist_m, lat, long,
  n, spread_m`, one row per 1 m grid point (`≈ lap_length_m` rows).
- **Invariants (ARCHITECTURE §3 lap-wrap refinement):**
  - bins **inside** a `protected_range` (corner box) are never rejected for
    spread/fold — real corner line-variation is preserved.
  - bins **outside** protected ranges with lateral spread > 6 m or sample pile-up
    > 1.5× median bin count are rejected and filled by periodic interpolation.
  - `track_dist_m` is increasing and within `[0, lap_length_m]`.
- This is the canonical ruler; treat tests as schema/invariant checks plus a
  determinism check (same inputs → identical output).



## centerline.install_centerline

Import: `from lap_analyzer.centerline import install_centerline`. Signature:
`(track: str, centerline_path: Path, synth_dir: Path) -> None`. Use a `tmp_path`
`synth_dir` and a real `data/samples/corpus/<track>_centerline.parquet` as input.

- **Contract:** writes `samples.parquet`, `laps.csv`, `meta.json` into `synth_dir`
  shaped like a normalized session.
- **Invariants:**
  - `samples.parquet` has the normalized-session columns; `lat`/`long` come from
    the centerline; unused channels (`speed_mph`, `throttle_norm`, …) are NaN
    sentinels; `gps_accuracy_m` is `0.0` (the "perfect accuracy" sentinel).
  - `laps.csv` has a single lap `1` marked `is_clean == True`.
  - `meta.json` records `synthetic: true` and the source path.

---

## normalize CLI — status lines & exit codes

Import: `from lap_analyzer.cli.normalize import main`. Call `main(argv: list[str])`
and capture stdout (`capsys`). Point `DATA_ROOT` at a `tmp_path` and create CSVs
with `make_trackaddict_csv`; create raw CSVs under
`<DATA_ROOT>/raw/<track>/` for the `--all` path, or pass a CSV path directly.

- **Contract (PIPELINE.md):** per-CSV status line is one of `ok`, `skip`
  (already normalized, no `--force`), `excl` (listed with `exclude` in notes),
  `gpsonly` (missing OBD — ingested GPS-only, output IS written), `FAIL`
  (other error).
- **CSV format note:** the raw `UTC Time` column is a **numeric Unix epoch
  (seconds)**, not an ISO string — `normalize` does `float(raw["utc"])` and
  `datetime.fromtimestamp(...)`. A synthetic "valid" CSV must supply epoch
  floats, e.g. `pd.Timestamp("2026-01-01T20:00:00Z").timestamp() + t`.
- **Invariants:**
  - normalizing a valid CSV prints a line starting `ok ` and **returns 0**.
  - a CSV missing OBD columns prints `gpsonly ` and the run still **returns 0**
    (the session IS ingested GPS-only; not a failure, not a skip).
  - a malformed/failing CSV prints `FAIL ` and the run **returns 2**.
  - re-running without `--force` on already-normalized output prints `skip `.
  - the final summary line reports the processed/skipped/excluded/gps-only/failed
    counts.
