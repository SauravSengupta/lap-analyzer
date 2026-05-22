# Lap Analyzer — Ridge Telemetry Pipeline

Personal telemetry analysis pipeline for the user's track-day laps at Ridge Motorsports Park. Takes TrackAddict CSV exports through to a clean per-corner transit table suitable for cross-lap analysis (line, apex, throttle, brake, lat-G).

## Architectural overview

```
TrackAddict CSV  ──►  normalize (Component 1)  ──►  samples.parquet + laps.csv + meta.json
                                                          │
                                                          ▼
                       Synthetic centerline  ◄──┐    label_corners (Component 3)
                       (track geometry ruler)   │         │
                                                │         ▼
                          built from many ──────┘    samples.parquet (+ track_dist_m, corner, drift)
                          drift-corrected laps         │
                                                       ▼
                                              corners.parquet (per-transit metrics)
                                                       │
                                                       ▼
                                              flag_quality (Component 3.5)
                                              (corpus-wide quality flags)
```

All geometric truth flows from **17 Google Maps satellite-derived apex pins**. Every other geometric field in the pipeline derives from those pins.

## Core design decisions

Each decision below is the result of trying multiple approaches and converging on the one that best survived contact with the data. Reasoning is preserved here so future-you (or future-LLM) understands why each component looks the way it does rather than rederiving.

### 1. Maps pins are the only hand-curated source of geometric truth

**What we do.** The user pins each corner's visual apex (inside-kerb-touch location) on Google Maps satellite imagery. These 17 pins live in [`data/notes/ridge_apex_pins.json`](data/notes/ridge_apex_pins.json) and are the *only* hand-curated geometric data in the pipeline.

**Why.**
- Survey-grade accuracy (~1-2m). Higher resolution and more sharply defined than any GPS recording can provide.
- Immune to recording artifacts — multipath, satellite geometry, hardware bias. The kerb is where it is regardless of how a phone happens to record it that day.
- Stable across time. No re-validation needed when adding new corpus laps.
- Reproducible. Anyone (or any agent) can look at the satellite image and verify the pin location, and disagreements are resolvable by inspection.

**Approaches we rejected:**

- **Hand-curated cluster apexes (the original approach).** Apex positions were chosen by clustering the corpus's lat-G peaks. Result: apex points reflect what was already in the corpus, not where the kerb actually is. When the corpus has biased GPS recordings (it does — TrackAddict has localized GPS issues), the cluster apex inherits those biases. Maps pins decouple the geometric truth from the corpus.

- **OpenStreetMap polyline as ground truth.** OSM has Ridge mapped, but the trace has 10-40m of region-varying georeference error. Two adjacent corners can be traced from satellite imagery of different vintages with different alignment. OSM is internally inconsistent at sub-30m precision, which is exactly the precision we need.

- **High-resolution county GIS imagery (Mason County WA GIS).** Has higher-resolution imagery than Google but the public portal only displays 3 decimal degrees in its coordinate readout (~78m precision). Useful as a visual reference but not as a coord source.

### 2. Per-lap GPS drift correction uses Maps-pin anchors with median aggregation

**What we do.** For each lap, we find the closest GPS sample to each of 17 anchor coords (within a 100m search radius inside each corner's bounding box). The per-anchor offset is (sample - pin) in meters. The lap's drift estimate is the **median** of those per-anchor offsets. The drift is subtracted from every sample before kd-tree mapping to the centerline.

**Why.**
- 17 anchors give dense spatial coverage — every section of the lap has a nearby ground-truth point.
- Median aggregation is robust to single-anchor failures (one anchor catches a glitched sample, or driver line varies wildly at one corner — median ignores it).
- Each anchor is grounded in physical truth (the Maps pin), so the drift correction is unbiased.
- Pure data pipeline — adding a new track just needs Maps pins, no further hand-curation.

**Approaches we rejected:**

- **Three slow-corner anchors with mean aggregation (the original approach).** T2, T13, T14 were validated against the May-1 L3 reference lap with a ≤8m cross-lap-spread rule. Two issues: (a) the rule only confirmed the corpus clustered near May-1's recorded coords; it didn't validate those coords were at the actual track. We later discovered T2 anchor was 55m off truth and T14 was 30m off. (b) Three anchors all near the start and end of the lap left ~2000m of unanchored middle (T3 through T12), where drift correction was extrapolated.

- **Centerline-based drift (median of per-sample offsets to the centerline).** Tried this. Used every sample in every lap (thousands per lap) for the drift estimate. Strictly better per-lap signal-to-noise, but introduced a subtle bias at slow corners. The centerline is built from drift-corrected laps, so it absorbs any systematic GPS bias present across most recordings. At T13 where the user reliably touches the kerb, anchor-based correction lands the lap at the kerb; centerline-based correction lands the lap on the centerline, which is ~10m off the kerb. For analysis at corners where the line is precise, anchor-based is more correct.

- **Fast-corner anchors excluded.** Originally rejected T1/T7/T9 as anchors because lat-G peak varies across laps at fast corners. With 17 anchors and median aggregation, the noisier ones are absorbed by the median — including them adds spatial coverage at no cost to drift estimate quality.

### 3. A synthetic centerline is the canonical `track_dist_m` ruler

**What we do.** [`data/corpus/ridge_centerline.parquet`](data/corpus/ridge_centerline.parquet) is a 3962-point polyline at 1m resolution. For each 1m grid point along the track, it stores the (lat, long) of the **median** position across all drift-corrected clean-lap samples in a ±2m window. Light Gaussian smoothing (σ=2m) removes residual jitter. This polyline parameterizes `track_dist_m`: every other lap's GPS gets projected onto the nearest centerline point.

**Why.**
- Captures the *physically traversed line* through the track at 1m fidelity. Better than any single recorded lap because it averages out per-lap GPS noise across ~250 contributors.
- Self-consistent: built from Maps-pin-corrected laps, so the centerline inherits the ground-truth alignment. No bootstrap dependency.
- Decouples `track_dist_m` from any specific recording. Replacing one lap or adding new ones doesn't shift the coordinate system meaningfully.

**Note on what "centerline" means here.** This is the **median driver path**, not the asphalt centerline. We have no track-edge data (Google Maps doesn't trace kerbs as machine-readable features), so we cannot derive the geometric centerline of the asphalt. At slow corners where the user reliably clips the inside kerb, the median driver path is close to the inside kerb, not the middle of the asphalt. At fast corners with varied lines, it runs through the middle of those lines. This is exactly what we want for cross-lap analysis (everyone compared against the typical line), but it's not the "physical track center."

**Approaches we rejected:**

- **A single recorded reference lap as the GPS ruler (the original approach).** May-1 L3 (`20260501-101355`) was hand-picked as canonical. Discovered (via Maps pins) that it had silent GPS glitches at T7 (~50m off truth) and T9 (~25m off), and likely smaller errors elsewhere. *Every* lap's `track_dist_m` mapping inherited May-1's contamination at those corners. Switching from a single-recording reference to a corpus-median reference eliminates this entire class of failure mode.

- **Hand-traced polyline from Google Maps satellite.** Considered. Would have required pinning ~50-100 points along the track centerline. The Maps-pin set we have (17 apexes) is enough to constrain the synthetic centerline implicitly via drift correction — the centerline ends up passing close to every pin without explicit constraint. So we get most of the benefit of a hand-traced polyline without the labor.

- **OpenStreetMap polyline.** As above (decision 1): too coarse, region-varying noise of 10-40m.

### 4. `apex_m` is the visual apex on the centerline, not the speed-minimum point

**What we do.** Each corner's `apex_m` field is set to the centerline `track_dist_m` value at the closest centerline point to the corresponding Maps pin. So `apex_m` is *the point on the median driver path nearest the physical inside kerb*. The transit metric `apex_dist_offset_m = min_speed_dist_m - apex_m` now means "how far before the kerb-touch does this lap reach its slowest speed."

**Why.**
- Maps pin is a stable physical reference. Speed-min depends on driver technique and varies by lap, so using it as the per-corner constant would be circular.
- The visual apex is the natural reference for both metrics that compare to it: `apex_dist_offset_m` (trail-brake duration) and `latg_peak_offset_m` (geometric apex placement). Both have meaningful sign conventions: negative = before the kerb-touch, positive = after.
- At slow precise corners where the driver consistently touches the kerb (T13), the typical lap's `apex_dist_offset_m` lands ~0m. At trail-brake corners (T2/T11/T16), it lands -25 to -55m. At throttle-through corners where speed-min is at the corner exit (T1/T7/T10), it lands +40 to +80m. The metric's value at each corner is itself a useful diagnostic of corner type.

**Approaches we rejected:**

- **`apex_m` = May-1's specific speed-min position (the original).** Inherited May-1's specific driving technique and any GPS error at that moment. Made `apex_dist_offset_m` a comparison-against-one-lap metric, not a comparison-against-ground-truth.

- **`apex_m` = corpus-median speed-minimum position.** Considered. Has the issue that for throttle-through corners (T1/T7/T10), the "speed-min in corner range" is at the corner boundary (where braking for the next corner starts), not at any meaningful apex. Apex_m would land at corner edges, making the metric trivially ~0 for all laps and losing the corner-type discrimination.

- **`apex_m` = corpus-median lat-G peak position.** Considered. Geometric-apex-based, would work for most corners. But still corpus-derived rather than ground-truth-anchored — adding or removing laps shifts the reference. The Maps pin doesn't.

### 5. Per-corner ranges (`start_m`, `end_m`) are derived from apex + preserved widths

**What we do.** Each corner's `start_m` and `end_m` are computed as `apex_m ± original_pre/post_apex_width`. Where adjacent corners would overlap with this scheme, the boundary is placed at the midpoint between the two corners' apexes.

**Why.**
- Each apex sits inside its corner's range by construction.
- Adjacent corners don't overlap (which would cause sample-label ambiguity).
- The widths come from the original cluster-analysis curation (where each corner was first hand-defined), preserving the human judgment about "where T3 ends and T4 begins" without re-doing that work.

**Approaches we rejected:**

- **Seed-lap-translated ranges (the original).** When the seed lap had any GPS error at a corner, the translated range was offset. T7's range translated to (1647.5-1745.5), but T7's visual apex Maps pin landed at 1784.5 — *outside* T7's range. So T7 samples were being labeled as T8. Same kind of bug at T13 and T14.

- **Data-derived ranges (lat-G threshold crossings).** Considered. Would compute per-lap "T7 entry" as where lat-G first exceeds 0.3g, "T7 exit" as where it last exceeds 0.3g, then take corpus medians. Cleaner in principle but adds complexity (have to handle hysteresis, choose threshold, deal with corner-to-corner chained lat-G). Deferred until we hit a use case where the current approach is insufficient.

## Pipeline at a glance

```bash
cd app
PYTHONPATH=src python -m lap_analyzer.cli.normalize --track ridge --all
PYTHONPATH=src python -m lap_analyzer.cli.label_corners --track ridge
PYTHONPATH=src python -m lap_analyzer.cli.flag_quality --track ridge
PYTHONPATH=src python -m lap_analyzer.cli.build_corpus --track ridge
```

| Component | Module | Status |
|---|---|---|
| 1. Normalizer | `lap_analyzer.normalize` | ✅ done |
| 2. Track definition | `tracks/ridge.json` | ✅ done |
| 2.5. Corner candidate scan | `lap_analyzer.corners` | ✅ done (historical curation only) |
| 3. Corner labeler | `lap_analyzer.labeler` | ✅ done |
| 3.5. Quality flags | `lap_analyzer.quality` | ✅ done |
| 4. Corpus builder | `lap_analyzer.corpus` | ✅ done |
| 5. Analysis library | — | ❌ not built |
| 6. Visualizer | — | ❌ not built |

The synthetic centerline build is a one-off (run when corpus changes meaningfully); see [`app/notebooks/build_centerline.py`](app/notebooks/build_centerline.py) and [`app/notebooks/install_centerline.py`](app/notebooks/install_centerline.py).

## Quality flags

The pipeline computes a STANDARD reliability tier baked into each transit row. Most analyses should just filter by it; the underlying metrics are kept for cases that need custom thresholds.

| Flag (in `corners.parquet`) | Meaning |
|---|---|
| `transit_reliable` (bool) | This (session, lap, corner) passes STANDARD: `gps_drift_disagreement_m ≤ 20m` AND `track_dist_offset_max_m ≤ 40m` AND `neighborhood_offset_max_m ≤ 40m` (the latter catches glitch spillover from adjacent corners). |
| `lap_reliable` (bool) | Every transit on this (session, lap) is `transit_reliable`. Use this when cross-corner consistency matters (time-delta across a full lap, etc.). |
| Session-level flags in `data/notes/ridge.json` | Hard `exclude` (wet, misfire, single-lap, off-pace) — dropped at normalize time; soft `flag: gps_unreliable` — kept in the data but typically filtered out by analysis. |

At the time of writing, **92% of transits and 84% of clean laps** pass the STANDARD filter — per-corner reliability rate is essentially uniform (91-95%) across all 16 corners, so there's no "T7 is broken" hot spot. Unreliability is per-lap (specific glitched recordings), not per-corner.

For spatial / line / apex analysis: filter by `transit_reliable` (or `lap_reliable` for whole-lap views). For kinematic / time analysis (throttle, brake, speed, lat-G): trust the data even on unreliable laps — OBD channels are unaffected by GPS issues.

Raw metrics kept on the transit row for custom analyses:
- `gps_drift_disagreement_m` — anchor-offset spread per lap
- `track_dist_offset_max_m` — within-corner deviation from the centerline
- `neighborhood_offset_max_m` — max across this corner + both neighbors

## File layout

```
app/
  pyproject.toml
  src/lap_analyzer/
    config.py             # path resolution
    schemas.py            # SessionMeta pydantic
    normalize.py          # Component 1 + per-lap dist rescaling
    corners.py            # Component 2.5 — corner candidate detection (historical)
    labeler.py            # Component 3 — Maps-pin drift correction + centerline projection
    quality.py            # Component 3.5 — corpus-wide quality flags
    cli/                  # one CLI per component
  tracks/
    ridge.json            # 16 corners, 17 Maps-pin anchors, reference path
  docs/
    visualizer-design.md  # design for the not-yet-built Streamlit visualizer
    curation-ridge.md     # historical corner-curation methodology (Component 2.5; superseded by Maps-pin workflow)
  notebooks/
    build_centerline.py   # rebuilds the synthetic centerline from the corpus
    install_centerline.py # installs the centerline as the labeler's reference
    *.py                  # various validation/diagnostic notebooks
data/
  raw/ridge/              # TrackAddict CSV exports
  sessions/ridge/         # normalized + labeled per-session output
    _synthetic_centerline/  # the canonical track_dist_m ruler (wraps ridge_centerline.parquet)
  corpus/
    ridge_centerline.parquet  # the synthetic centerline polyline
  notes/
    ridge.json            # session-level exclude / flag annotations
    ridge_apex_pins.json  # the 17 Maps-pin apex coords (primary geometric truth)
telemetry-pipeline-spec.md  # original design spec
```

## What's not yet built

- **Corpus builder.** A trivial `pd.concat` of every session's `corners.parquet` into `data/corpus/ridge_corners.parquet`. Needed for sub-second cross-session queries.
- **Analysis library.** Pure functions on the corpus: `Corpus.corner_summary("T11")`, `Corpus.corner_compare()`, `Corpus.consistency()`. Emerges naturally from what the visualizer needs.
- **Visualizer.** Streamlit, single Python file. Design captured in [app/docs/visualizer-design.md](app/docs/visualizer-design.md) — envelope-based (median of top-decile laps as the default reference, not a single "best" lap), three-pane layout, time-delta-vs-distance trace, channel overlays. ~150-300 LOC.

## How to add a new track

The architecture is reusable. To add another track:

1. Drop raw TrackAddict CSVs into `data/raw/<track>/`.
2. Create `tracks/<track>.json` with: `track_id`, `name`, `lap_length_internal_m` (guess from official length), `start_finish`, empty `corners` list.
3. Run `normalize.py --track <track> --all`.
4. **Pin every numbered corner's visual apex on Google Maps** and save to `data/notes/<track>_apex_pins.json`. This is the only hand-curated step.
5. Run `corners.py` candidate detection to seed initial corner ranges (or just hand-define them — Maps pins make this easier).
6. Populate `calibration_anchors` from the Maps pins.
7. Populate `corners` (with widths around each apex; the labeler will refine).
8. Run `label_corners.py`, then `build_centerline.py`, then `install_centerline.py`, then `label_corners.py` once more.
9. Run `flag_quality.py`.

The whole bootstrap takes ~1 hour of hands-on work per track plus ~1 minute of compute.
