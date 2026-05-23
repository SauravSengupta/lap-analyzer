# Architecture

This document captures the design rationale behind each architectural choice —
*why* the pipeline looks the way it does, including the approaches that were
tried and rejected. For a runbook on how to use the pipeline, see
[PIPELINE.md](PIPELINE.md). For how to bootstrap a new track, see
[NEW-TRACK.md](NEW-TRACK.md).

Each decision below is the result of trying multiple approaches and converging
on the one that best survived contact with the data. The reasoning is preserved
so future-you (or a future LLM) understands why each component looks the way it
does rather than rederiving it.

---

## Design principles

1. **Deterministic and reproducible.** Same input always produces the same
   output. No random sampling, no non-deterministic algorithms.
2. **OBD speed is the canonical speed channel.** It is not corrected for the
   car's non-OEM tire size. The offset (~6%) is consistent across all laps and
   sessions, so every comparison stays valid. GPS speed is a secondary
   cross-check (`speed_mph_gps`).
3. **Store the raw transit, derive metrics on demand.** Each per-corner-transit
   row carries enough that most downstream questions can be answered without
   touching the raw samples — but the samples stay on disk (with
   `sample_idx_start/end` back-references) for deeper drills.
4. **No auto-generated prose.** The pipeline emits metrics and structured data;
   interpretation happens in the user's head or an LLM conversation. (Tools like
   MyRaceLab emit templated observations — "may suggest there may be an issue
   with your approach" — and we actively avoid that pattern.)
5. **Geometric truth is hand-curated once, per track, from satellite imagery.**
   Everything geometric derives from a small set of Google Maps apex pins (see
   decision 1). Adding a track is a data task, not a code task.

---

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
                                                       │
                                                       ▼
                                              build_corpus (Component 4)
                                              data/corpus/<track>_corners.parquet
                                                       │
                                                       ▼
                                              analysis library + visualizer (Components 5, 6)
```

All geometric truth flows from the Google Maps satellite-derived apex pins (17
for Ridge across its 16 corners; 12 for PIR). Every other geometric field in the
pipeline derives from those pins.

The package is organized one module per component, with a thin CLI wrapper per
component under `lap_analyzer/cli/`:

| Component | Module | What it owns |
|---|---|---|
| 1. Normalizer | `lap_analyzer.normalize` | CSV → tidy parquet + per-lap summary + per-lap dist rescaling |
| 2. Track definition | `tracks/<track>.json` | corner ranges, anchors, reference, in `track_dist_m` units |
| 2.5. Corner candidate scan | `lap_analyzer.corners` | lat-G peak detection (bootstrap-time seeding only) |
| 3. Corner labeler | `lap_analyzer.labeler` | Maps-pin drift correction + centerline projection + transit table |
| — Centerline builder | `lap_analyzer.centerline` | the synthetic `track_dist_m` ruler |
| 3.5. Quality flags | `lap_analyzer.quality` | corpus-wide reliability flags |
| 4. Corpus builder | `lap_analyzer.corpus` | concat sessions → one queryable parquet |
| 5. Analysis library | `lap_analyzer.analysis` | pure functions over the corpus |
| 6. Visualizer | `visualizer/` | Streamlit app + track-specific pages |

---

## Core design decisions

### 1. Maps pins are the only hand-curated source of geometric truth

**What we do.** The user pins each corner's visual apex (inside-kerb-touch
location) on Google Maps satellite imagery. These pins live in
`data/notes/<track>_apex_pins.json` and are the *only* hand-curated geometric
data in the pipeline.

**Why.**
- Survey-grade accuracy (~1–2 m at Ridge; ~5–10 m where the user was less sure
  of the exact kerb-touch). Higher resolution and more sharply defined than any
  GPS recording can provide.
- Immune to recording artifacts — multipath, satellite geometry, hardware bias.
  The kerb is where it is regardless of how a phone records it on a given day.
- Stable across time. No re-validation needed when adding new corpus laps.
- Reproducible. Anyone (or any agent) can look at the satellite image and verify
  a pin location, and disagreements are resolvable by inspection.

**Approaches we rejected:**

- **Hand-curated cluster apexes (the original approach).** Apex positions chosen
  by clustering the corpus's lat-G peaks. Result: apex points reflect what was
  already in the corpus, not where the kerb actually is. When the corpus has
  biased GPS recordings (it does — TrackAddict has localized GPS issues), the
  cluster apex inherits those biases. Maps pins decouple the geometric truth from
  the corpus.
- **OpenStreetMap polyline as ground truth.** OSM has the tracks mapped, but the
  trace has 10–40 m of region-varying georeference error. Adjacent corners can be
  traced from imagery of different vintages with different alignment. OSM is
  internally inconsistent at the sub-30 m precision we need.
- **High-resolution county GIS imagery.** Higher-resolution imagery than Google,
  but the public portal only displays 3 decimal degrees (~78 m precision) in its
  coordinate readout. Useful as a visual reference, not as a coordinate source.

### 2. Per-lap GPS drift correction uses Maps-pin anchors with median aggregation

**What we do.** For each lap, we find the closest GPS sample to each anchor coord
(within a 100 m search radius inside each corner's bounding box). The per-anchor
offset is `(sample − pin)` in meters. The lap's drift estimate is the **median**
of those per-anchor offsets, subtracted from every sample before kd-tree mapping
to the centerline.

**Why.**
- Dense spatial coverage — every section of the lap has a nearby ground-truth
  point.
- Median aggregation is robust to single-anchor failures (a glitched sample at
  one corner, or wildly varying driver line at one corner — the median ignores
  it).
- Each anchor is grounded in physical truth (the Maps pin), so the correction is
  unbiased.
- Pure data pipeline — adding a track just needs Maps pins, no further curation.

A passive centerline-based drift estimate is computed alongside (median of
per-sample offsets to the centerline) and stored as a diagnostic, but it is
**not** used for correction — see the rejected approaches.

**Approaches we rejected:**

- **Three slow-corner anchors with mean aggregation (the original approach).**
  Validated against a single reference lap with a ≤8 m spread rule. Two problems:
  (a) the rule only confirmed the corpus clustered near *that lap's* recorded
  coords; it didn't validate the coords were at the actual track (one anchor
  turned out 55 m off truth, another 30 m off). (b) Three anchors clustered at
  the lap's start/end left ~2000 m of unanchored middle where correction was
  extrapolated.
- **Centerline-based drift (median of per-sample offsets to the centerline).**
  Better per-lap signal-to-noise, but biased at slow corners: the centerline is
  built *from* drift-corrected laps, so it absorbs any systematic GPS bias common
  across recordings. Where the driver reliably touches the kerb, anchor-based
  correction lands the lap at the kerb; centerline-based correction lands it on
  the centerline, ~10 m off the kerb. For corners where the line is precise,
  anchor-based is more correct.
- **Excluding fast-corner anchors.** Originally rejected fast corners as anchors
  because their lat-G peak varies lap-to-lap. With many anchors and median
  aggregation, the noisier ones are absorbed — including them adds spatial
  coverage at no cost.

### 3. A synthetic centerline is the canonical `track_dist_m` ruler

**What we do.** `data/corpus/<track>_centerline.parquet` is a 1 m-resolution
polyline. For each 1 m grid point along the track, it stores the **median**
(lat, long) across all drift-corrected clean-lap samples in a ±2 m window. This
polyline parameterizes `track_dist_m`: every other lap's GPS gets projected onto
its nearest centerline point. (Ridge's centerline, for instance, is a 3962-point
polyline built from 258 clean laps.)

**Why.**
- Captures the *physically traversed line* at 1 m fidelity — better than any
  single recorded lap because it averages out per-lap GPS noise across hundreds
  of contributors.
- Self-consistent: built from Maps-pin-corrected laps, so it inherits the
  ground-truth alignment. No bootstrap dependency once installed.
- Decouples `track_dist_m` from any specific recording. Replacing or adding laps
  doesn't shift the coordinate system meaningfully.

**What "centerline" means here.** This is the **median driver path**, not the
asphalt centerline. We have no track-edge data, so we can't derive the geometric
center of the asphalt. At slow corners where the driver reliably clips the inside
kerb, the median path runs close to the kerb; at fast corners with varied lines,
it runs through the middle of those lines. This is exactly what we want for
cross-lap analysis (everyone compared against the typical line).

**Approaches we rejected:**

- **A single recorded reference lap as the ruler (the original approach).** One
  hand-picked lap was canonical — and it had silent GPS glitches (~50 m off at
  one corner, ~25 m at another). *Every* lap's `track_dist_m` inherited that
  contamination at those corners. A corpus-median reference eliminates this
  entire class of failure.
- **A hand-traced polyline from satellite imagery.** Would require pinning
  50–100 centerline points. The apex pins we already have implicitly constrain
  the synthetic centerline via drift correction (it ends up passing close to
  every pin), so we get most of the benefit without the labor.

#### The lap-wrap refinement (applies to all tracks)

Bootstrapping PIR exposed a single-track assumption in the original centerline
builder. At the start/finish wrap, the kd-tree can assign samples from
physically different positions to the same `track_dist_m`, producing a "zigzag"
in the front-straight centerline. Ridge's wrap happened to be clean enough never
to show it; PIR's wasn't.

The fix (now active for every track):

1. **Corner-protected bin rejection.** Corner `[start_m, end_m]` spans are passed
   to `build_centerline` as *protected ranges*. Bins **inside** corners are never
   rejected — real corner line-variation is legitimate. Bins **outside** corners
   (straights, especially the wrap zone) are rejected when their contributing
   samples disagree laterally (`spread > 6 m`) or pile up (`n > 1.5×` the median
   bin count, a sign of `track_dist_m` folding).
2. **Periodic interpolation** across the lap wrap (was linear) to fill rejected
   bins.
3. **Corner-aware variable Gaussian smoothing** — corners keep a sharp σ=2 m;
   open straights blend toward σ=10 m to erase residual wrap wiggle while
   preserving a straight's genuine low-frequency bow.

The consequence for existing tracks: a Ridge centerline rebuild will shift
slightly (an improvement). `protected_ranges` is optional and defaulted, so
nothing breaks — but it means **a track's corner boxes must be populated before
its centerline is built** (see [NEW-TRACK.md](NEW-TRACK.md)).

### 4. `apex_m` is the visual apex on the centerline, not the speed-minimum point

**What we do.** Each corner's `apex_m` is the centerline `track_dist_m` at the
point nearest the corresponding Maps pin — *the point on the median driver path
nearest the physical inside kerb*. The transit metric `apex_dist_offset_m =
min_speed_dist_m − apex_m` then means "how far before/after the kerb-touch does
this lap reach its slowest speed."

**Why.**
- The Maps pin is a stable physical reference. Speed-min depends on technique and
  varies by lap, so using it as the per-corner constant would be circular.
- The visual apex is the natural reference for both metrics that compare to it:
  `apex_dist_offset_m` (trail-brake duration) and `latg_peak_offset_m` (geometric
  apex placement). Both have meaningful signs: negative = before the kerb,
  positive = after.
- The metric's value is itself a corner-type diagnostic: at slow precise corners
  it lands ≈ 0; at trail-brake corners −25 to −55 m; at throttle-through corners
  +40 to +80 m.

**Approaches we rejected:** `apex_m` = a single lap's speed-min (inherits that
lap's technique + GPS error); = corpus-median speed-min (lands at corner edges
for throttle-through corners, losing discrimination); = corpus-median lat-G peak
(corpus-derived, so it shifts as laps are added — the Maps pin doesn't).

### 5. Per-corner ranges (`start_m`, `end_m`) are derived from apex + preserved widths

**What we do.** Each corner's `start_m`/`end_m` are `apex_m ± original
pre/post-apex width`. Where adjacent corners would overlap, the boundary is the
midpoint between the two apexes.

**Why.** Each apex sits inside its corner's range by construction; adjacent
corners don't overlap (which would make sample labeling ambiguous); and the
widths carry forward the human judgment about "where T3 ends and T4 begins" from
the initial curation without redoing it.

**Approaches we rejected:** seed-lap-translated ranges (when the seed lap had GPS
error, the range was offset — at Ridge this put T7's visual apex *outside* T7's
range, so its samples were labeled T8); and data-derived lat-G-threshold ranges
(cleaner in principle but adds hysteresis/threshold complexity — deferred until a
use case demands it).

---

## Quality flags

The pipeline computes a STANDARD reliability tier and bakes it into each transit
row. Most analyses should just filter by it; the underlying raw metrics are kept
for cases that need custom thresholds.

| Flag (in `corners.parquet`) | Meaning |
|---|---|
| `transit_reliable` (bool) | This (session, lap, corner) passes STANDARD: `gps_drift_disagreement_m ≤ 20 m` AND `track_dist_offset_max_m ≤ 40 m` AND `neighborhood_offset_max_m ≤ 40 m` (the last catches glitch spillover from adjacent corners). |
| `lap_reliable` (bool) | Every transit on this (session, lap) is `transit_reliable`. Use when cross-corner consistency matters (whole-lap time-delta, etc.). |
| Session-level flags in `data/notes/<track>.json` | Hard `exclude` (wet, misfire, single-lap, off-pace) — dropped at normalize time; soft `flag: gps_unreliable` — kept in the data but excluded from the centerline and typically filtered out of spatial analysis; `reference: true` — kept on disk but out of the corpus and all corpus-wide stats. |

Reliability is **per-lap, not per-corner** — it tracks specific glitched
recordings, not a "this corner is broken" hot spot. The per-corner reliability
rate is essentially uniform across all corners. At the time of writing, Ridge
runs ~92% of transits / ~84% of clean laps reliable; PIR runs ~97.8% / ~95.5%.

**How to use them:**
- **Spatial / line / apex analysis** (GPS map, apex offsets): filter by
  `transit_reliable` (or `lap_reliable` for whole-lap views).
- **Kinematic / time analysis** (throttle, brake, speed, lat-G): trust the data
  *even on unreliable laps* — OBD channels are unaffected by GPS issues. A
  glitched lap's speed and throttle traces are still correct.

Raw metrics kept on the transit row for custom thresholds:
`gps_drift_disagreement_m` (anchor-offset spread per lap),
`track_dist_offset_max_m` (within-corner deviation from the centerline),
`neighborhood_offset_max_m` (max across this corner + both neighbors).
