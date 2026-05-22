# Visualizer design

For overall project state, architecture rationale, and pipeline components, see [`/README.md`](../../README.md). This doc captures the design of the visualizer (Component 6) so it can be picked up cold when implemented.

## Stack

**Streamlit**, single Python file, runs locally with `streamlit run app.py`. ~150-300 LOC.

The corpus comes from `data/corpus/ridge_corners.parquet` (built by the not-yet-implemented Component 4 — `build_corpus.py`, a near-trivial concat of every session's `corners.parquet`). The visualizer should also have direct access to per-session `samples.parquet` for channel traces. Both are kept on local disk; no database needed.

## Layout

```
┌─ sidebar ─────────┐  ┌─ main view ─────────────────────────────────┐
│ track             │  │  GPS map (selected lap vs envelope of fast) │
│ corner (T1-T16)   │  │  colored by selected metric                 │
│ metric            │  │                                             │
│ session/lap pick  │  ├─────────────────────────────────────────────┤
│ ── reference ───  │  │  channel traces vs track_dist_m (overlaid)  │
│   top-decile env  │  │  speed | throttle | brake | lat-G           │
│   median fast     │  │  with the corner shaded                     │
│   user-picked     │  ├─────────────────────────────────────────────┤
│                   │  │  numeric transit-stats table (selected vs ref)│
└───────────────────┘  └─────────────────────────────────────────────┘
```

## Load-bearing principles

- **Default reference is the envelope (top-10% laps overlaid faintly), NOT a single "best" lap.** Single-lap A/B comparisons amplify GPS noise; the envelope absorbs it.
- **Spatial queries** (GPS map, line/apex analysis) honor the three reliability filters: session `gps_unreliable` flag in `data/notes/<track>.json`, `gps_drift_disagreement_m > 12m` per-lap, `track_dist_offset_max_m > 40m` per-transit.
- **Kinematic queries** (channel traces, time-deltas, transit stats) ignore those filters — OBD data is unaffected by GPS issues. A glitched lap's speed and throttle traces are still trustworthy.
- **No auto-prose.** The pipeline emits metrics and structured data. Interpretation happens in the user's head or in an LLM conversation. (We've seen tools like MyRaceLab generate templated observations like "may suggest there may be an issue with your approach" — actively avoid this pattern.)
- **Make the "best lap for this corner" definition swappable.** Default to "fastest T11 on a top-decile lap," but expose alternatives (highest exit speed, highest min-speed, median fast lap, user-picked specific lap, etc.).

## Three apex definitions are first-class, not aliased

The corner-transit schema carries three apex positions per transit. The visualizer should let the user pick which one is the "apex" for any view, and label clearly:

| Apex definition | Source | What it means |
|---|---|---|
| **Visual apex** (`apex_m` field on the corner) | Maps pin's centerline position | Where the inside kerb is. Fixed per corner; same across all transits. |
| **Speed-minimum apex** (`min_speed_dist_m` field on the transit) | Per-transit data | Where this lap was slowest in the corner. Captures trail-brake duration. |
| **Lat-G peak apex** (`latg_peak_dist_m` field on the transit) | Per-transit data | Where the load peaks. Closest to the "geometric apex" intuition. |

The metric `apex_dist_offset_m = min_speed_dist_m - apex_m` measures trail-brake duration. The metric `latg_peak_offset_m = latg_peak_dist_m - apex_m` measures geometric apex placement relative to the kerb. Both have meaningful sign conventions: negative = before the kerb, positive = after.

## High-value features

1. **Time-delta-vs-distance trace** — cumulative time delta between selected lap and reference plotted along `track_dist_m`. Shows exactly where time was gained or lost. The single most informative product for a track-day driver.

2. **Multi-corner overview** — small bar/sparkline per corner showing "your lap vs reference time delta" so the user picks the corner with the biggest delta to drill into.

3. **Channel traces vs `track_dist_m`** with the selected corner shaded — speed, throttle, brake, lat-G overlaid for the selected lap and the reference. Complements the spatial GPS map.

4. **Three-apex display in the GPS map.** Render all three apex points for a corner: the visual apex (fixed marker — where the kerb is), this lap's speed-min position, and this lap's lat-G peak. Driver can see at a glance whether they're trail-braking late (speed-min far past visual apex) vs braking early (speed-min before visual apex).

## Things to NOT build into v1

- **Auto "find me what's different" / clustering / outlier detection** — push these questions to LLM conversation, per the original spec.
- **Multi-tenant / SaaS plumbing** — personal-use tool first.
- **Auto-prose tooltips** ("your apex is wider, costing 0.2s") — they age badly and mislead.
- **Per-region calibration uncertainty overlays** — used to be relevant when the T4-T10 region was unanchored. With the current 17-Maps-pin anchor set, drift correction is spatially uniform; no specific region needs flagging.
