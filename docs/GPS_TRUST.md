# GPS Trust Model

TrackAddict logs two independent position sources with different, complementary
failure characteristics. This doc says what you can trust for which question, and
how the pipeline handles GPS defects. It is the reference the section-timing code
(`crossing_gap_s` / `CONFIDENCE_GAP_S`) and the reliability tiers point back to.

## Always trustworthy: OBD

The OBD channels — `speed_mph`, `throttle`, `long_g`, `rpm`, and `dist_lap_m`
(OBD-integrated distance) — are high-rate and glitch-free. Anything measured *from
OBD* — a channel value at a sample, or "how far has the car driven" — is
trustworthy. (Exception: ~12% of sessions log without OBD; those fall back to GPS
and are flagged `obd_present=False`.)

## GPS failure taxonomy (three distinct modes)

| Mode | What happens | Signal | Corrupts | Handled by |
|---|---|---|---|---|
| **1. Lateral drift** | GPS sits offset perpendicular to the true track | `track_dist_offset_m` (GPS→centerline distance) | apex / line position | per-lap Maps-pin drift correction + `transit_reliable` |
| **2. Along-track teleport / backward-walk** ("inner-loop glitch") | `track_dist_m` jumps or reverses while OBD stays smooth; localized, 50–200 m | `\|track_dist_m − dist_lap_m\|` spike | the `track_dist_m` ruler; gate crossings near it | `drop_gps_glitches` (samples), fused-axis mask |
| **3. Coarse temporal resolution** | effective GPS rate falls to ~1 Hz; position frozen then jumps ~30 m/s | GPS fix spacing / update rate | *timing* of any GPS position event (gate crossings); the GPS↔OBD offset | `crossing_gap_s` (section timing) |

Mode 3 is the subtle one: the position is spatially *accurate* at each fix but
*temporally* coarse, so an event timed from GPS position fires up to ~0.5 s off.
It is invisible to every spatial metric (a Mode-3 lap can read `track_dist_offset_m`
≈ 30 m and look reliable) yet it compresses/stretches a section time by up to ~2 s.

## The four trust axes

Trust is not one boolean per lap — it depends on the question:

| Question | Trustworthy source | Fails under |
|---|---|---|
| Channel *value* at a sample (speed/throttle/G) | OBD | never |
| *Relative* along-track motion ("how far did I drive") | OBD `dist_lap_m` | never |
| *Absolute* position / line-length (the GPS↔OBD offset) | GPS | Modes 2 & 3 |
| *Timing* of a position event (a gate crossing) | GPS-located + OBD-timed | Mode 3 (irreducible) |

**Principle:** *OBD tells you how far; GPS tells you where — and "where" is only
trustworthy when GPS is sampling well.*

When GPS drops to ~1 Hz, the information needed to place a gate to better than
~±30 m (and thus time a section to better than ~±1 s) is simply not in the log. No
fusion of GPS and OBD recovers information the data does not contain, so a
coarse-GPS transit is **irreducibly uncertain** and must be flagged, not "repaired."

## Section timing

Section times (`analysis.span_time` / `section_times` / `range_section_times`) time
a lap between two gates. Each transit carries a confidence:

- `crossing_gap_s` = the elapsed time between the **good GPS fixes** bracketing a
  gate (good = a fresh, non-teleport fix). Small gap = fine GPS; wide gap = Mode 3
  (coarse) or Mode 2 (a teleport punctured the bracket). `timing_gap_s` is the max
  over the two gates.
- `timing_reliable` = `timing_gap_s < CONFIDENCE_GAP_S` (default 0.4 s).
- **Reliable** → the gate-crossing time is used (line-length preserved).
- **Not reliable** → `section_time_s` is the **OBD-anchored fallback** (time between
  the `dist_lap_m` crossings of the two gate distances) — the robust expected value.
  It is surfaced with a warning and **excluded from "fastest through" ranking**, not
  dropped.

This replaced an earlier spatial guard (`_gates_glitch_free`, a `|track_dist −
dist_lap|` ≥ 50 m near-gate check) that both missed Mode-3 laps and over-dropped
clean fast lines.
