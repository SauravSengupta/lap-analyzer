# T8–T11 Downshift Comparison — Design

**Date:** 2026-05-17
**Track:** Ridge Motorsports Park
**Status:** Approved design, pre-implementation

## Goal

Let the driver compare T8→T11 laps split by whether a downshift was made
entering T8. T9–10 are full-throttle uphill; downshifting for T8 trades shift
losses for more torque up the hill. The view should make the net time
trade-off visible and show where time is won and lost across the span.

## Constraints / context

- Source data: per-sample `rpm` and `speed_mph` (OBD) live in
  `samples.parquet`. There is no `gear` channel anywhere.
- Gears separate cleanly from the `rpm / speed_mph` ratio — a corpus-wide
  histogram shows well-separated peaks.
- ~3–4% of sessions have no OBD data at all (`rpm == 0`).
- TrackAddict GPS glitches affect ~3–7% of per-lap corner transits; existing
  `_drop_gps_glitches` and the OBD-distance section-time check handle these.
- The pipeline (`normalize.py`, corpus build) is reserved for validated
  metrics. Gear-from-ratio is a derivation and stays out of the pipeline for
  now (Approach 1 — derive on the fly). Promotion to a real channel is a
  later, separate change if it proves useful.

## Architecture

Three layers, no pipeline changes:

1. **Analysis library** (`app/src/lap_analyzer/analysis.py`) — pure functions:
   gear derivation, lap classification, gear-band computation.
2. **Visualizer shared module** (`app/visualizer/shared.py`, new) — cached
   loaders and helpers extracted from `app.py` so both pages can use them.
3. **New visualizer page** (`app/visualizer/pages/1_Downshift_T8-T11.py`,
   new) — the comparison UI.

## Component 1 — Gear derivation

`derive_gear(samples: pd.DataFrame, bands: np.ndarray) -> pd.Series`

- Compute `ratio = rpm / speed_mph` for samples with `rpm > 1200` and
  `speed_mph > 8` (idle / standstill ratios are meaningless).
- Snap each qualifying sample to the nearest band center in `bands`.
- Apply a **min-dwell filter**: a gear value only counts once it is held for
  N consecutive samples (~0.3 s, so N derived from sample rate). This removes
  the one-/two-sample transients while the clutch slips mid-shift.
- `rpm == 0` or `speed_mph <= 8` → `gear = NaN`.
- Gear indices are numbered so a higher index = taller gear (lower ratio).
  Numbering is relative to the bands found; published 981 ratios are NOT
  hard-coded. Absolute "3rd/4th" labels are an optional later off-by-one
  adjustment.

`gear_bands(track: str) -> np.ndarray`

- Scans every session's `samples.parquet` once, pools the qualifying
  `rpm / speed_mph` ratios, finds the cluster centers (histogram peak
  detection), returns sorted band centers.
- Computed once; the visualizer wraps it in `@st.cache_data`.
- The Ridge corpus has **four** bands (~55, ~71, ~99.7, ~181). The shortest
  gear (~181) is rare — used only in slow corners — so the peak-detection
  threshold must be low enough to keep it (a 2% threshold drops it; 0.5%
  keeps it). Missing it would break T8/T11/T13 gear traces, which use it.

## Component 2 — Lap classification

`classify_t8_section(samples, track_def, bands) -> dict`

Operates on one lap's samples restricted to the **detection window**
`track_dist_m` in `[T8.start_m - 50, T10.end_m]`. The window deliberately
ends at the end of T10 — *before* the T11 braking zone — so the routine
downshift for T11 braking is never counted. This window is fixed and
independent of the Component 3 timing endpoint.

- Derive gear (Component 1), detect gear-change events after the dwell
  filter: a **downshift** = snapped gear steps down, an **upshift** = steps up.
- T8 entry zone = `track_dist_m` in `[T8.start_m - 50, T8.apex_m]`.
- Classification:
  - **`downshift`** — exactly one downshift, located in the T8 entry zone.
    Upshifts elsewhere (accelerating out toward T9) are expected and ignored.
  - **`no-downshift`** — zero downshifts anywhere in the window.
  - **`excluded`** — a downshift inside the window but outside the T8 entry
    zone (traffic), OR more than one downshift, OR gear undeterminable (OBD
    dropout / too few samples). Sub-reason: `traffic` / `no-obd` / `ambiguous`.
- Returns: `label`, `excluded_reason` (or None), `downshift_dist_m` (or None),
  `t8_apex_gear`.

## Component 3 — Section span & timing

- Span = `track_dist_m` `[T8.start_m, endpoint]` where `endpoint` is
  `T11.start_m` (default) or `T10.end_m` ("end of T10"), per a UI radio.
- Per-lap elapsed time reuses `_first_crossing_t` plus the OBD-distance
  sanity check from `section_times()` — a lap is only timed if it crosses
  both bounds and its OBD-integrated distance matches the nominal span width
  within tolerance. This rejects GPS-glitched laps.
- Per-corner breakdown (T8 / T9 / T10) reuses the existing `section_times()`.

## Component 4 — Visualizer page

New page `app/visualizer/pages/1_Downshift_T8-T11.py`. Streamlit auto-discovers
`pages/`; `app.py` becomes the default page with no logic changes.

**Shared refactor:** extract `_corpus`, `_track_def`, `_samples`,
`_drop_gps_glitches`, `session_hhmm`, `format_lap_time` into
`app/visualizer/shared.py`; `app.py` imports them instead of defining them
inline. No behavior change.

**Controls (sidebar):**
- Clean-laps-only toggle — default on.
- "Top N pace deciles" slider — tighten to the fastest laps.
- Section endpoint radio — `T11 entry` (default) / `end of T10`.

**Group summary header:** lap counts per group (downshift / no-downshift /
excluded), with `excluded` broken out by sub-reason. Per-group pace-decile
breakdown table so the driver can see whether the two groups are pace-matched
before trusting the comparison.

**Main plot:** stacked panels over `track_dist_m` across the span (+200 m
display buffer each side), corner shading reused from `app.py`:
- Speed, Throttle, Gear — each panel draws both groups as a median line +
  p10–p90 band in distinct colors.
- Cumulative time-delta panel: median time of one group minus the other
  across the span, anchored at span start — shows where the downshift wins
  (T9-10 hill) and loses (the shifts).

**Section-time comparison:** table of median T8→endpoint section time per
group + delta, with a per-corner (T8 / T9 / T10) breakdown.

**Single-lap overlay:** a selectbox per group to pick one individual lap;
its raw trace overlays on top of the group bands.

## Error handling

- OBD-dropout sessions → gear NaN → `excluded (no-obd)`, counted separately.
- GPS glitches → `_drop_gps_glitches` before plotting against `track_dist_m`;
  section times use the OBD-distance check.
- Thin groups (< ~5 laps after filtering) → warning, skip the p10–p90 band,
  still draw the median.
- Lap not crossing both span bounds → no section time, drops from the time
  table; can still contribute to channel bands where samples exist.
- Empty result after filters → `st.warning` + `st.stop()`, mirroring `app.py`.

## Verification

No pytest harness exists; follow the project's `app/notebooks/` script
pattern. New script `app/notebooks/verify_gear_classification.py`:

- Derive corpus-wide gear bands; print band centers + a ratio histogram to
  eyeball separation.
- Run `classify_t8_section` over all laps; print group counts and a few
  example laps per group with downshift locations.
- Spot-check specific laps the driver knows they downshifted in vs not.

Manual: run the visualizer, open the new page, confirm bands / time-delta /
tables render and the filters behave.

## Out of scope

- Persisting `gear` as a pipeline channel (Approach 2) — deferred until the
  derivation proves useful.
- Absolute gear numbering against published 981 ratios — a later off-by-one
  adjustment.
- Applying the comparison to corners other than the T8–T11 span.
