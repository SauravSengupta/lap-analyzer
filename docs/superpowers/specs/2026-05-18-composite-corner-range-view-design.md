# Composite Corner-Range View — Design

**Date:** 2026-05-18
**Track:** Ridge Motorsports Park
**Status:** Approved design, pre-implementation

## Goal

Let the main visualizer analyze a *range* of consecutive corners (e.g. T6→T7,
T8→T10, T1→T5, T3→T5), not just a single corner. The time through a compound
corner is often dictated by what happened entering the complex, so the driver
needs to see the whole flow — the cumulative time delta across the span and a
per-corner breakdown of where the time went.

## Constraints / context

- This generalizes the existing single-corner focus in `app/visualizer/app.py`
  (Approach 1). A separate "Complex" page was considered and deferred.
- The page's render path is already driven by one `(display_a, display_b)`
  window pair — a range just supplies a wider one.
- `ridge.json` and the data pipeline are **unchanged** — this is purely an
  additive visualizer feature. No corner merging, no re-labeling, no risk to
  GPS drift correction or the corpus.
- No pytest harness; verification follows the project's `app/notebooks/`
  script pattern plus a Streamlit `AppTest` smoke check.

## Architecture

Two layers:

1. **Analysis library** (`app/src/lap_analyzer/analysis.py`) — new pure
   functions: `section_range_bounds`, and a range section-time computation.
2. **Main visualizer** (`app/visualizer/app.py`) — From/To corner pickers, a
   range-aware window, header, corner shading, and a per-corner section-time
   breakdown table. The existing single-corner transit table is removed.

## Component 1 — Selection UI

The sidebar's single "Corner focus" picker becomes two:

- **From** picker — options `full lap` + every corner in track order. When
  `From = full lap`, the To picker is hidden (full-lap mode, unchanged).
- **To** picker — shown only when From is a corner; options constrained to
  corners *at or after* From; defaults to From.

Modes:
- `From = full lap` → full-lap view (unchanged from today).
- `From == To` (a corner) → single-corner focus, **identical to today** — no
  behavior change on this path.
- `To > From` → a corner range.

The lap picker's per-lap label, which today shows a focused corner's section
time, shows the *range* section time when a range is active.

## Component 2 — Section window & reference lap

`section_range_bounds(track_def, from_id, to_id, pre_m=50.0, post_cap_m=250.0)
-> tuple[float, float]` (new, in `analysis.py`)

- `a = max(0, from_corner.start_m - pre_m)`
- `b = min(next_corner_after_to.start_m, to_corner.end_m + post_cap_m)`
- For `from_id == to_id` it returns exactly the same `(a, b)` as the existing
  `section_bounds()[corner]`, so the single-corner path is genuinely unchanged.

**Reference ("best") lap.** `find_best_lap` becomes range-aware: the reference
is the fastest lap through the whole range, among laps that are clean and have
every corner in `[from..to]` flagged `transit_reliable`. Single-corner and
full-lap behavior is unchanged. "Fastest through" uses the range section time
(Component 4).

## Component 3 — What's rendered

**Header metric row.**
- Single corner (unchanged): `section time | min speed | exit speed | max lat-G`.
- Range: `range section time | entry speed (into From) | exit speed (out of To)`
  — each with a delta vs the reference lap. Entry speed is surfaced because it
  is the "what happened going into the complex" number.

**Envelope plot.** The existing stacked panels (speed / throttle /
longitudinal-G) plus the cumulative Δt-vs-distance curve, over the range window
`[a - 200m, b + 200m]`. Unchanged machinery, wider window. Corner shading:
every corner in the window is shaded; corners *inside* `[from..to]` are shaded
darker. A thin gold apex line is drawn for **each** corner in the range, per
the existing `visual / speed-min / lat-G` apex-definition radio.

**Per-corner section-time breakdown table.** Replaces the removed transit
table. Columns: `section time (selected lap)` and `Δ vs best`, one row per
corner. Shown for:
- **Range** → the corners in `[from..to]`.
- **Full lap** → all corners.
- **Single corner** → not shown (the header metric already gives that number).

Per-corner section times come from the existing `section_times()`.

**Removed:** the existing "Per-corner transit — selected lap vs top-decile
median" table. Its code is **commented out, not deleted** (retained in case it
is wanted back; there is no git history to recover it from otherwise).

## Component 4 — Range section time

The range total is **not** the sum of per-corner section times — adjacent
corners' `section_bounds` windows overlap. It is computed directly across the
range bounds.

A range section-time function (corpus-wide, analogous to `section_times()`)
computes, for every `(session_id, lap)`, the elapsed time between the first
`track_dist_m` crossings of the range bounds `a` and `b`. It reuses the
existing `_first_crossing_t` helper and the OBD-distance sanity check (a lap is
only timed if its OBD-integrated distance over the interval matches `b - a`
within tolerance — rejects GPS-glitched laps). This is the same validated
pattern as `section_times()` and `span_time()`; the implementation should share
that per-lap window-timing logic rather than duplicate it.

## Error handling

- `To < From` is impossible — the To picker only offers corners ≥ From.
- A lap that does not cross both range bounds → no range section time → header
  shows "—" and the lap is excluded from best-lap eligibility, mirroring the
  existing single-corner `_section_time_help` diagnostic.
- GPS-glitched laps are rejected by the OBD-distance check in the range
  section-time computation.
- No eligible reference lap for the range → the sidebar shows the existing
  "No reliable + clean transit" style message.

## Verification

No pytest harness. Follow the project pattern:

- A script in `app/notebooks/` exercises the new pure functions with
  assertions: `section_range_bounds` (including `from == to` equals
  `section_bounds[corner]`), and the range section-time computation on a
  synthetic lap (normal case + GPS-glitch rejection + bound-not-crossed).
- Streamlit `AppTest` smoke test of `app.py` confirms it loads with no
  exception.
- Manual: run the visualizer, exercise single corner (confirm no regression),
  a range, and full lap.

## Out of scope

- The entry-speed-vs-complex-time correlation scatter — a clean fast-follow
  once the range-section machinery exists.
- A separate dedicated "Complex" page (Approach 2) — build later only if the
  in-place generalization proves clunky.
- Any change to `ridge.json` or the data pipeline.
