# Track Telemetry Analysis Pipeline — Spec

> **NOTE (2026-05-10, updated 2026-05-22):** This is the **original design spec**, written before implementation, and is preserved only as a historical record of the original intent. The pipeline has since diverged in important ways — most notably, the track-definition and drift-correction architectures were rebuilt around Google Maps satellite-derived apex pins as the single source of geometric truth. **It is not the source of truth** for the track JSON schema, the calibration/anchor approach, the labeler's drift correction, or the meaning of `apex_m`. For current behavior, use the live docs: design rationale is in [ARCHITECTURE.md](ARCHITECTURE.md), the CLI runbook + column contract + output schema are in [PIPELINE.md](PIPELINE.md), and the project overview is in [../README.md](../README.md). Read this spec only for color on the original component decomposition and design intent.

## Purpose

Build a deterministic data pipeline that ingests TrackAddict CSV exports and produces structured outputs designed to be analyzed in conversation with an LLM. The goal is **diagnostic insight** ("what did I do differently in T11 on my fastest lap vs typically?") rather than visualization or coaching prose generation.

This is a personal-use one-off, not a product. Optimize for clarity, ease of iteration, and the ability to grow into a SaaS later if the workflow proves valuable.

## Context for the implementer

The user is a track-day driver running a Porsche 981 Cayman. They log sessions with TrackAddict (an iOS/Android app + OBD reader + external GPS). They want to do corner-by-corner and cross-session analysis that no off-the-shelf tool currently offers — specifically, they want to ask questions like:

- "What do I usually do through T11 at The Ridge?"
- "What was different about my fastest T6 transit vs my typical one?"
- "Which corners am I most inconsistent at?"
- "Am I getting better at T13 over time?"

These questions require a corpus across many sessions, with corner-level granularity. The tools they've evaluated (MyRaceLab, RaceStudio, MoTeC i2) are all session-scoped and stop at sector granularity.

The pipeline outputs are intended to be loaded into a chat conversation with an LLM. The LLM does the interpretation. The pipeline does the deterministic data work.

## Design principles

1. **Deterministic and reproducible.** Same input always produces same output. No random sampling, no non-deterministic algorithms.
2. **Token-efficient outputs.** A 5MB CSV reduces to a small structured artifact that can fit comfortably in an LLM context window.
3. **Store the raw transit, derive metrics on demand.** Per-corner-transit rows should contain enough that any reasonable downstream question can be answered without going back to the raw samples — but the raw samples are also kept available for deeper drills.
4. **OBD speed is the canonical speed channel.** Don't try to correct it for the user's non-OEM tire size. The 6% offset is consistent across all laps and sessions, so all comparisons are valid. GPS speed is a secondary signal used only for cross-checks.
5. **No auto-generated prose.** The pipeline outputs metrics and structured data. Interpretation happens in conversation. (We've seen MyRaceLab generate templated observations like "may suggest there may be an issue with your approach" — actively avoid this pattern.)

## Architecture

Five components:

1. **Session normalizer** — TrackAddict CSV → tidy parquet + per-lap summary
2. **Track definition** — JSON file per track, hand-built once, defines corner ranges in distance-from-start
3. **Corner labeler** — adds corner labels to normalized sessions and produces a corner-transit table
4. **Corpus builder** — concatenates many normalized sessions into one queryable dataset per track
5. **Analysis library** — functions called on the corpus during conversational analysis

Build in this order. Stop after #3 to use it on real data before building #4 and #5.

---

## Component 1: Session normalizer

**Script:** `normalize.py`

**Invocation:**
```
python normalize.py path/to/session.csv --track the-ridge --out sessions/
```

**Input:** A TrackAddict CSV. The first ~12 lines are comments starting with `#` containing session metadata (vehicle, app version, start/finish coordinates, sector splits). Skip these for the data parse but extract the metadata separately.

The data has these columns (approximately, names are exact):
- `Time` — session-relative seconds, ~21 Hz sample rate (median 47ms between samples, but variable)
- `UTC Time` — wall clock
- `Lap` — integer, 0 = warmup, increments at start/finish crossing, last lap is in-lap
- `Sector` — integer, sector within the lap
- `Predicted Lap Time`, `Predicted vs Best Lap` — TrackAddict's own predictions, ignore
- `GPS_Update`, `GPS_Delay`, `GPS_Update`, `Accuracy (m)` — GPS quality channels
- `Latitude`, `Longitude`, `Altitude (m)`, `Altitude (ft)`
- `Speed (MPH)` — GPS-derived
- `Heading` — degrees
- `Accel X` — lateral G (positive = right turn, in this car's logging convention; verify with track direction)
- `Accel Y` — longitudinal G (positive = forward accel, negative = braking)
- `Accel Z` — vertical
- `Brake (calculated)` — binary 0/1 derived from Accel Y threshold
- `Barometric Pressure (PSI)`, `Pressure Altitude (ft)`
- `OBD_Update`
- `Engine Speed (RPM) *OBD`
- `Vehicle Speed (mph) *OBD` — **this is the canonical speed channel**
- `Throttle Position (%) *OBD` — note: this car's pedal sensor maxes at ~90.6%, not 100%. Normalize per-session by dividing by the observed per-session max, store as `throttle_norm` (0–1)
- `Engine Coolant Temp (F) *OBD`
- `Intake Air Temp (F) *OBD`
- `Intake Manifold Pressure (PSI) *OBD`

**Outputs:**

1. **Long-form parquet:** `sessions/the-ridge/<session_id>.parquet`
   
   Schema (one row per sample):
   ```
   session_id      string    derived from input filename
   t               float64   session-relative seconds
   lap             int32
   dist_m          float64   distance from start, integrated from OBD speed
   dist_lap_m      float64   distance from current lap's start crossing
   speed_mph       float64   from OBD
   speed_mph_gps   float64   from GPS, kept for sanity checks
   throttle_norm   float64   0–1, normalized against per-session max
   brake           int8      0 or 1
   rpm             int32
   lat_g           float64   from Accel X
   long_g          float64   from Accel Y
   coolant_f       float32
   iat_f           float32
   lat             float64
   long            float64
   altitude_m      float32
   gps_accuracy_m  float32
   ```

2. **Per-lap summary CSV (or parquet):** `sessions/the-ridge/<session_id>_laps.csv`
   
   One row per lap:
   ```
   session_id, lap, lap_time_s, lap_dist_m,
   max_speed, avg_speed,
   max_rpm,
   max_lat_g, max_accel_g, max_decel_g,
   pct_wot, avg_throttle, pct_braking,
   coolant_min, coolant_max, iat_min, iat_max,
   is_clean, clean_reason
   ```
   
   `pct_wot` is computed as fraction of samples with `throttle_norm >= 0.95`.
   
   `is_clean` heuristic: a lap is clean if it's not the first or last lap, AND no 5-second window within the lap is more than 2 standard deviations slower than the same distance window across the session's other flying laps. (This is approximate — refine if it produces obviously wrong results.) Set `clean_reason` to a short string when False ("warmup", "in-lap", "traffic", etc.).

3. **Session metadata JSON:** `sessions/the-ridge/<session_id>_meta.json`
   ```json
   {
     "session_id": "20260502-143515",
     "track": "the-ridge",
     "vehicle": "2014 Porsche Cayman",
     "date_utc": "2026-05-02T21:35:15Z",
     "best_lap": 2,
     "best_lap_time_s": 125.617,
     "n_laps": 6,
     "n_clean_laps": 4,
     "sample_rate_hz": 21.3,
     "trackaddict_start_finish": {"lat": 47.254699, "long": -123.192475},
     "trackaddict_sectors": [...],
     "throttle_max_observed": 90.59
   }
   ```

**Distance calculation note:** Integrate OBD speed (mph → m/s) using trapezoidal rule over the variable timestep `dt`. Document that this gives a self-consistent distance that's about 6% over published track length due to the wheel-speed-to-distance offset. We accept this; everything downstream uses this same internal distance.

**Lap timing note:** Lap times are computed start-of-lap to start-of-next-lap, NOT end-of-lap as logged. The `Lap` column transitions at start/finish crossing, so use the time delta between transitions.

---

## Component 2: Track definition

**File:** `tracks/the-ridge.json` (and similar for other tracks later)

**Built once per track**, by hand, using a helper script (next section).

**Schema:**
```json
{
  "track_id": "the-ridge",
  "name": "The Ridge Motorsports Park",
  "configuration": "full",
  "official_length_m": 3975,
  "direction": "counter-clockwise",
  "start_finish": {"lat": 47.254699, "long": -123.192475},
  "reference_session_id": "20260502-143515",
  "reference_lap": 2,
  "lap_length_internal_m": 4214.7,
  "corners": [
    {
      "id": "T1",
      "name": null,
      "start_m": 145.0,
      "end_m": 235.0,
      "apex_m": 190.0,
      "type": "right",
      "notes": "Fast right-hander after the front straight"
    },
    {
      "id": "T6",
      "name": null,
      "start_m": 950.0,
      "end_m": 1280.0,
      "apex_m": 1080.0,
      "secondary_apex_m": 1180.0,
      "type": "right",
      "notes": "Long carousel, single segment despite two visual apexes"
    }
  ],
  "sectors": [
    {"id": "S1", "start_m": 0, "end_m": 1300},
    {"id": "S2", "start_m": 1300, "end_m": 2700},
    {"id": "S3", "start_m": 2700, "end_m": 4214.7}
  ]
}
```

Notes:
- `lap_length_internal_m` is in the pipeline's distance-from-start coordinate (which is ~6% longer than reality due to OBD offset). All `start_m` / `end_m` / `apex_m` are also in this coordinate system. This is fine because all data uses the same coordinate system.
- `secondary_apex_m` is optional, only for multi-apex segments like T6.
- `name` is for nicknames if they exist (Cathedral, the Ridge Complex, etc.). Leave null if unknown.

---

## Component 2.5: Corner-finding helper

**Script:** `find_corners.py`

This is a one-time tool to assist in building a track JSON. Not run regularly.

**Invocation:**
```
python find_corners.py sessions/the-ridge/<session_id>.parquet --lap 2 --out track-draft.json
```

**Algorithm:**

1. Load the specified lap from the normalized parquet.
2. Smooth the lat-G channel with a 7-sample centered moving average (~0.33 sec at 21 Hz).
3. Find peaks in `|lat_g_smooth|` with `scipy.signal.find_peaks`, parameters: `height=0.4`, `distance=63` (about 3 seconds at 21 Hz).
4. For each peak (a candidate corner apex):
   - Walk backward from the peak until `|lat_g_smooth|` drops below 0.2 → that's the tentative corner entry distance
   - Walk forward similarly → tentative corner exit distance
   - Find the speed minimum within the [entry, exit] window — that's the apex distance (refines step 3's peak which is in lat-G space)
   - Find the brake-on distance: the latest point before the apex where `brake == 1` after a stretch of `brake == 0`
   - Find the throttle-return distance: the earliest point after the apex where `throttle_norm > 0.5` for at least 0.3 seconds
5. Output a draft track JSON with these candidate corners numbered T1, T2, ... in order of distance-from-start. Type (left/right) is determined from the sign of the lat-G peak.
6. Also output a sanity-check visualization (PNG or HTML):
   - 2D plot of the lap's lat/long (the GPS path), color-coded by speed
   - Markers at each candidate apex (filled circle), entry (open triangle), exit (open square)
   - Label each candidate with its T-number
   - Side-by-side or alongside, a distance-vs-speed line chart with the same markers, so the user can cross-reference

The user iterates on this draft visually, comparing to the official track map, and edits the JSON to:
- Merge candidates that are actually one corner (e.g. multi-apex sweepers)
- Add corners the algorithm missed (low-lat-G kinks)
- Adjust boundaries
- Add nicknames

---

## Component 3: Corner labeler

**Script:** `label_corners.py`

**Invocation:**
```
python label_corners.py sessions/the-ridge/<session_id>.parquet --track tracks/the-ridge.json
```

**Outputs:**

1. Updates the parquet in place to add a `corner` column. For each sample:
   - If `dist_lap_m` falls within any corner's `[start_m, end_m]` range → that corner's id (e.g. "T6")
   - Otherwise → the straight name, e.g. "S_post_T6" (a straight is the gap between corners; name it after the preceding corner)
   - Out-laps and in-laps (laps where `is_clean=False` because warmup/cooldown) still get labeled.

2. Produces a **corner-transit table:** `sessions/the-ridge/<session_id>_corners.parquet`

   One row per (lap × corner) for clean laps only:
   ```
   session_id, lap, corner_id, date,
   
   # Boundary stats
   entry_speed_mph, exit_speed_mph, entry_dist_m, exit_dist_m,
   time_in_corner_s, distance_in_corner_m,
   
   # Min/max within corner
   min_speed_mph, min_speed_dist_m,
   max_speed_mph, max_lat_g, max_accel_g, max_decel_g,
   peak_brake (0 or 1), peak_throttle_norm,
   
   # Apex stats
   apex_speed_mph,           # speed at apex_m as defined in track JSON
   apex_dist_offset_m,       # min_speed_dist_m - apex_m (negative = early apex this lap)
   
   # Secondary apex (null if not applicable)
   secondary_apex_speed_mph,
   
   # Input timing (distances from corner start)
   throttle_lift_dist_m,     # first sample where throttle_norm < 0.8
   brake_on_dist_m,          # first sample with brake==1
   brake_off_dist_m,         # last sample with brake==1
   throttle_return_dist_m,   # first sample where throttle_norm > 0.8 sustained for 0.3s
   wot_dist_m,               # first sample at WOT (>= 0.95) after apex; null if never
   
   # Aggregate inputs
   mean_throttle_norm, pct_wot, mean_lat_g, pct_braking,
   
   # Reference link to long-form
   sample_idx_start, sample_idx_end
   ```
   
   Each of `throttle_lift_dist_m`, `brake_on_dist_m`, etc. should be expressed as **distance from corner start** (so 0 means right at corner entry, negative would mean before corner entry, positive means inside corner). Null if event didn't occur in the corner.

This table is the analytical workhorse. Most cross-session questions can be answered by aggregating across this table without touching the long-form parquet.

---

## Component 4: Corpus builder

**Script:** `build_corpus.py`

**DO NOT BUILD THIS UNTIL ≥3 NORMALIZED SESSIONS EXIST AT A TRACK.** Building it on one session is wasted work.

**Invocation:**
```
python build_corpus.py --track the-ridge --out corpus/
```

**Inputs:** All `*_corners.parquet` and `*_laps.csv` files in `sessions/the-ridge/`, plus the track JSON.

**Outputs:**
- `corpus/the-ridge_corners.parquet` — all corner-transits across all sessions, concatenated. Index by (session_id, lap, corner_id).
- `corpus/the-ridge_laps.parquet` — all lap summaries.
- `corpus/the-ridge_summary.json` — small JSON with counts (n_sessions, n_clean_laps, date range, etc.) for quick orientation.

The long-form sample-level parquets stay per-session; we don't concatenate those into the corpus by default (too large). The corner-transit rows have `sample_idx_start/end` references back into them when needed.

---

## Component 5: Analysis library

**Module:** `analysis.py`

A small set of pure functions called during interactive analysis. Designed to be imported in a Python REPL or Jupyter, or used as building blocks in a chat-with-LLM workflow.

Suggested API:

```python
from analysis import Corpus

corpus = Corpus.load("corpus/the-ridge")

# Single-corner summaries
corpus.corner_summary("T11", agg="median")    # typical
corpus.corner_summary("T11", agg="best")      # best single transit
corpus.corner_summary("T11", agg="distribution")  # full distribution stats

# Compare specific transits
corpus.corner_compare("T11", 
    transit_a=("20260502-143515", lap=2),
    transit_b=("20260502-143515", lap=3))

# Time-loss decomposition for a specific lap
corpus.time_loss_decomposition(
    session="20260502-143515", lap=2,
    reference="theoretical_best")  # or "session_best", or specific lap

# Consistency analysis
corpus.consistency("T11")          # variance metrics for one corner
corpus.consistency_ranking()       # rank all corners by inconsistency

# Trend analysis
corpus.progression("T11", metric="time_in_corner_s", by="date")

# Outlier detection
corpus.find_outliers("T11", metric="apex_speed_mph", n=5)
```

Each function returns a small structured result (dict, DataFrame, or named tuple) that's easy to print or further reduce. None of them generate prose.

---

## Tech choices

- **Python 3.11+**
- **pandas** for data handling
- **pyarrow** for parquet I/O
- **scipy** for `signal.find_peaks` and any smoothing
- **matplotlib** for the corner-finding visualization (component 2.5 only); skip for everything else
- No web framework, no UI library. This is scripts and data files.

Project layout:
```
.
├── normalize.py
├── find_corners.py
├── label_corners.py
├── build_corpus.py
├── analysis.py
├── tracks/
│   └── the-ridge.json
├── sessions/
│   └── the-ridge/
│       ├── <session_id>.parquet
│       ├── <session_id>_laps.csv
│       ├── <session_id>_corners.parquet
│       └── <session_id>_meta.json
├── corpus/
│   └── the-ridge_corners.parquet
└── raw/
    └── the-ridge/
        └── *.csv  (original TrackAddict exports, kept for re-runs)
```

---

## Build order and stopping points

**Phase 1 (build now):**
1. `normalize.py` — get one CSV in, validate the schema and the per-lap summary by spot-check
2. `find_corners.py` + first draft of `tracks/the-ridge.json` — generate the visualization, share with Saurav, iterate until the corner ranges match his understanding of the track
3. `label_corners.py` — produces corner-transit table, validate against the same lap

**STOP. Use these on a real session. Verify the output makes sense before going further.**

**Phase 2 (build only after ≥3 sessions normalized):**
4. `build_corpus.py`
5. `analysis.py`

**Phase 3 (only if Phase 2 proves valuable):**
- Anything else (UI, web app, multi-format ingest, automatic track database from OSM, etc.)

---

## Things that are explicitly NOT in scope for v1

- Wheel-speed correction or true ground-speed derivation
- Multi-format support (no AiM, MoTeC, VBOX yet)
- Track database automation (manual JSON per track is fine for now)
- Visualization beyond the one-time corner-finding sanity check
- Auto-generated coaching prose
- Multi-driver or multi-vehicle analysis
- Real-time streaming or in-car use
- Web interface, mobile app, anything user-facing beyond CLI

---

## Edge cases to handle but not over-engineer

- A lap where the `Lap` counter doesn't transition cleanly (rare but happens) — log a warning, don't crash
- A session with zero clean laps (e.g. all laps had traffic) — pipeline should still produce outputs, just with empty corner-transit table
- A corner where the algorithm can't find e.g. brake-on (you didn't brake) — leave the field null, don't error
- Sample rate variation — most code should be timestep-agnostic. Only thing that depends on Hz is `find_peaks` distance parameter; document the assumption (~21 Hz)

---

## Open questions to resolve as we go

1. The first corner-finding pass on The Ridge will likely produce some spurious or merged candidates. Plan to iterate on the track JSON 2–3 times before locking it.
2. The "clean lap" heuristic is a first guess. May need refinement once we see which laps it incorrectly flags or excludes.
3. Throttle normalization assumes the per-session max is "true 100%." If a particularly conservative session never reaches WOT, the normalization is off. Worth a sanity check: throttle max should be ~90% raw on a Porsche 981; if a session shows max << 90%, flag it.

---

## Reference: today's session for testing

The user has uploaded one session: `Log-20260502-143515 The Ridge - Full Track - 2:05.749`.

Validation expectations from a prior analysis pass:
- 6 laps total; lap 0 = warmup, laps 1-4 = flying, lap 5 = cooldown
- Lap times (start-to-start): L1=131.4s, L2=125.6s (best), L3=132.7s, L4=142.9s (slow due to apparent traffic mid-lap)
- Sample rate: ~21 Hz
- Max GPS speed: 118.9 mph; max OBD speed: 122.4 mph
- Throttle max raw: 90.59% (use this as the per-session normalization denominator)
- Coolant range: 190°F start → 221°F end
- Peak lat-G: ~1.18g; peak decel: ~1.2g
- Should produce ~12-16 corner candidates from `find_corners.py` (the track has 16 numbered corners but some may merge)

Use these numbers as smoke tests during implementation.
