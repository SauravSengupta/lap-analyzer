# Visualizer

A local Streamlit app for reading one lap against the field — where you gained or
lost time, how your line and inputs compare to the typical fast lap, and how a
corner or a multi-corner complex breaks down. It reads the corpus and per-session
samples produced by the [pipeline](PIPELINE.md); no database.

## Running it

```powershell
# PowerShell — sample data
$env:DATA_ROOT = "data/samples"; $env:PYTHONPATH = "."; python -m streamlit run visualizer/app.py
```

```bash
# bash — sample data
DATA_ROOT=data/samples PYTHONPATH=. python -m streamlit run visualizer/app.py
```

Drop `DATA_ROOT` (defaults to `data/`) to run against your own processed data.
The **Track** picker in the sidebar switches between tracks; the default can be
preset with the `LAP_ANALYZER_TRACK` env var.

## Layout

**Sidebar**, top to bottom:

- **Track** — which track's corpus to load.
- **Corner** — a `From`/`To` selection that defines the section in focus: `full
  lap`, a single corner, or a multi-corner *complex* (e.g. `T13`–`T15`). Each
  picker has `<` / `>` steppers. Below it, an **Apex definition** radio (visual /
  speed-min / lat-G peak) and a **"Fastest through …"** panel that names the
  fastest eligible lap through the current section and offers a button to load
  it.
- **Lap** — `Clean laps only` / `Lap reliable only` filters, then a
  Date → Time → Lap cascade to pick the lap to inspect. Lap labels show the lap
  (or section) time and pace decile.

**Main view**, top to bottom:

- **Header** — the selected lap's date/time, lap time, and a ✓/✗ reliability
  mark, plus a metric row that adapts to the selection: a single corner shows
  section time + min/exit speed + max lat-G (each with a delta vs the best); a
  complex shows section time + entry/exit speed; full lap shows no metric row.
- **Channel chart** — stacked, distance-aligned panels for **speed**,
  **throttle**, and **longitudinal G**, sharing an x-axis of `track_dist_m`. Each
  panel overlays three things: the top-decile **envelope** (faint p10–p90 band +
  dotted median), the **best lap** (green), and the **selected lap** (crimson).
  Corners are shaded (darker inside the focused range) with apex lines.
- **Δt panel** — when a reference lap is available, a fourth panel shows the
  cumulative time delta vs the best lap along the section (see below).
- **Per-corner section-time table** — for a complex or full lap, one row per
  corner with its section time and the delta vs the best lap. (A single corner
  has no table — its time is already in the header.)

## The reference model

The default reference is the **top-decile envelope**, not a single "best" lap:
the fastest 10% of clean laps for the track, overlaid as a faint p10–p90 band
with a dotted median, per channel. Single-lap A/B comparisons amplify GPS noise;
the envelope absorbs it and shows the *spread* of fast laps, not one lucky one.

On top of that, a concrete **best lap** is drawn (green) and drives the header
deltas and the Δt panel:

- **Full lap** → the fastest lap that is both clean and `lap_reliable`.
- **A corner or complex** → the fastest *section* time through the `From`..`To`
  range, among laps that are clean and have every corner in the range
  `transit_reliable`.

"Fastest" here is a section/lap time, and eligibility is gated on reliability so
the reference can't be won by a GPS-glitched lap that recorded a physically
impossible shortcut.

## The From/To section model

Selecting `From`/`To` focuses the whole page on that stretch:

- A **single corner** reproduces per-corner analysis.
- A **complex** (range of corners) is treated as one section — useful for linked
  corners like the back chicane, where exit speed out of one feeds entry into the
  next and per-corner times mislead.
- **full lap** compares whole laps.

Section times come from `lap_analyzer.analysis.section_times` /
`range_section_times`, which are thin shells over the **trajectory layer**
(`trajectory.estimate_trajectory` once per lap, then `section_timing` per corner).
The section is timed where the lap's monotone along-track estimate `s_hat` crosses
the two section-bound ruler positions — value and confidence from one pass, so they
can never disagree, and a scalar position crosses `s_hat` exactly once (ghost
crossings are impossible). Each transit carries its `sigma_t_s` and a
`rank_eligible` flag: only tier-A (σ_t ≤ 0.10 s) transits win the "fastest through"
ranking. A coarse-GPS or Mode-4-drift transit gets a wide σ, is shown as an
"estimate ± σ" and **excluded from ranking** — surfaced with a warning rather than
dropped. See [GPS_TRUST.md](GPS_TRUST.md). The channel chart adds ±200 m of context
on each side so the adjacent brake zones are visible.

## The Δt panel

The single most informative product for a track-day driver: the **cumulative
time gap** between the selected lap and the best lap, plotted along the section,
so you can see exactly where time moved. Two curves:

- **Δt total** (orange) — the raw time gap at each track position.
- **Δt speed-only** (dotted) — the gap with racing-line length removed: the laps
  aligned by *distance travelled* rather than track position.

The shaded band between them is what your line is worth in seconds. Where total
runs **below** speed-only, a shorter line is buying back time despite lower
speed; where it runs above, your line is longer than the best lap's. This answers
"am I down because I'm slow, or because my line is longer?"

The two laps are aligned on the **trajectory ruler `s_hat`**
(`trajectory.estimate_trajectory`), not raw `track_dist_m`. Raw `track_dist_m`
jitters sample-to-sample, and its derivative drives the Δt slope, so a few metres
of GPS noise would fabricate phantom wobble. `s_hat` is OBD-smooth and monotone by
construction but still bends onto the centerline at corner scale (the estimator
handles bad GPS internally), so the slope tracks real speed-and-line without
flattening genuine line-length differences and without any glitch pre-filtering.

## Three apex definitions

The visualizer treats three apex positions as first-class and lets you switch via
the **Apex definition** radio; the chosen one draws as a gold apex line per corner
in the focused range:

| Definition | Source | Means |
|---|---|---|
| **Visual** | the corner's `apex_m` (Maps pin on the centerline) | where the inside kerb is — fixed across all laps |
| **Speed-min** | this transit's `min_speed_dist_m` | where this lap was slowest — captures trail-brake duration |
| **Lat-G peak** | this transit's `latg_peak_dist_m` | where load peaks — closest to the "geometric apex" |

See [ARCHITECTURE.md](ARCHITECTURE.md) decision 4 for why the visual apex is a
fixed Maps-pin reference rather than a per-lap quantity.

## Reliability: kinematic vs spatial

The app honors the split described in [PIPELINE.md](PIPELINE.md) and
[ARCHITECTURE.md](ARCHITECTURE.md):

- **Kinematic channels** (speed, throttle, long-G) are trusted on *every* lap —
  OBD data is unaffected by GPS issues — so they always draw.
- **Spatial alignment** (how a lap plots along the track, and which laps are
  eligible as the reference): every trace is drawn on the trajectory ruler `s_hat`,
  which is monotone by construction and places GPS-glitched samples at their honest
  along-track position — no per-lap sample dropping. Stretches where the estimate is
  wide (σ large — sparse or rejected GPS evidence) are shaded and explained in a
  banner. Reference eligibility still gates on the spatial `transit_reliable` /
  `lap_reliable` flags, which certify a lap's **lateral** line position (see
  [GPS_TRUST.md](GPS_TRUST.md)).

## Track-specific pages

`visualizer/app.py` is fully track-agnostic — every corner geometry comes from the
selected track's JSON. Bespoke worked examples live under `visualizer/pages/`
(e.g. the Ridge T8→T11 downshift study) and hard-code one track's corner IDs and
thresholds, so they guard on `current_track()` and render an explainer instead of
crashing when another track is selected. See
[NEW-TRACK.md](NEW-TRACK.md#writing-track-specific-analysis-pages) for how to
write one.

## Load-bearing principles

- **Envelope, not a single best lap**, is the default reference — single-lap
  comparisons amplify GPS noise.
- **The "best lap" definition is reliability-gated and section-aware**, so a
  glitched or off-line lap can't masquerade as the reference.
- **No auto-generated prose.** The app emits charts, metrics, and structured
  tables; interpretation stays with the human (or an LLM conversation). Templated
  "your apex is wider, costing 0.2s" coaching text ages badly and misleads — it's
  deliberately absent.

## Not built (deliberately, for now)

- A **GPS-map pane** colouring the line by a chosen metric (the channel chart +
  Δt panel cover the "where" question for now).
- Auto outlier/cluster detection — push those questions to an LLM conversation
  over the corpus instead.
