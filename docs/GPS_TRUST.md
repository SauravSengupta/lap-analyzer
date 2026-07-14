# GPS Trust Model

TrackAddict logs two independent position sources with different, complementary
failure characteristics. This doc says what you can trust for which question, and
how the pipeline handles GPS defects.

Since the unified GPS-trust migration (2026-07), **one layer decides whether a
GPS-derived value is trustworthy**: `lap_analyzer/trajectory.py`. Every consumer —
section times, "fastest through" rankings, the visualizer axis and σ-shading,
`corners.parquet` positions, quality tiers, the warning banner — reads the same
per-lap trajectory estimate `(s_hat, σ)` from that module. This file is the
reference it points back to; [ARCHITECTURE.md](ARCHITECTURE.md) decision 6 covers
why it replaced the earlier six-detector approach.

## The reframe: OBD is the backbone, GPS is evidence, every number carries its σ

```
s_hat(t) = dist_lap_m(t) + delta_hat(dist_lap_m)   # the car's monotone position on the canonical ruler
sigma(s)                                            # honest 1-σ, from the SAME estimation pass
```

`delta_hat` is a robust, corridor-weighted, slope-bounded smooth of the offset
evidence `delta = track_dist_m − dist_lap_m`, measured only on fresh,
snap-masked, outlier-rejected GPS fixes. Where GPS evidence is missing or
rejected, `delta_hat` relaxes toward the OBD prior (`δ=0`) and σ grows — which
reproduces the old OBD-anchored fallback *exactly* in the zero-evidence limit,
with the measured ±0.55 s fallback noise appearing as σ ≈ 0.4–0.5 s instead of a
silent flag.

The load-bearing property: **a value and its confidence can never disagree.**
They come from one evidence pass (design R1), and a monotone `s_hat` crosses any
scalar ruler position exactly once, so ghost crossings (a teleport path clipping a
distant gate) are structurally impossible.

## Always trustworthy: OBD

The OBD channels — `speed_mph`, `throttle`, `long_g`, `rpm`, and `dist_lap_m`
(OBD-integrated distance, per-lap rescaled to the canonical lap length; endpoints
pinned) — are high-rate and glitch-free. Anything measured *from OBD* — a channel
value at a sample, or "how far has the car driven" — is trustworthy. (Exception:
~12% of sessions log without OBD; those fall back to GPS speed, are flagged
`obd_present=False` / `speed_source="gps"`, and the trajectory backbone becomes
`∫ v_gps dt` with a wider σ floor — see Mode notes below.)

## GPS failure taxonomy (four distinct modes)

| Mode | What happens | Scale | Signal | Corrupts | How the trajectory layer handles it |
|---|---|---|---|---|---|
| **1. Lateral drift** | GPS sits offset perpendicular to the true track; often locally varying (12–21 m), so a single per-lap correction can't see one drifted corner | 1–20 m | `track_dist_offset_m`; corridor lateral offset | apex / line position | upstream Maps-pin drift correction; residual absorbed as line signal inside the corridor with honest σ; off-corridor evidence downweighted `exp(−(excess/5m)²)` |
| **2. Along-track teleport / backward-walk** | `track_dist_m` jumps or reverses while OBD stays smooth; plus kd-tree wrong-segment snaps (600–2700 m) | 50–200 m / 600–2700 m | `\|track_dist_m − dist_lap_m\|` spike; `\|δ − lap median\|` | the `track_dist_m` ruler; crossings near it | snap mask (200 m) + Hampel teleport rejection (window 31, 5×MAD, 60 m backstop); `s_hat` bridges under the slope bound, σ grows with the gap |
| **3. Coarse temporal resolution** | effective GPS rate falls to ~1 Hz; position frozen then jumps ~30 m/s (4/64 sessions, cleanly separable below 1.0 Hz) | timing ±0.5 s/gate | GPS fix spacing / update rate | *timing* of any GPS position event | fresh-fix evidence + OBD backbone interpolation; `σ_gap = slope_bound × distance-to-nearest-evidence`. On a **straight** (κ≈0) a 1 Hz gap costs ~0.06 s → **recovered to tier A**; in a **hairpin** it costs 0.2–0.5 s → honestly wide, excluded |
| **4. Locally-drifting but temporally FINE** | fresh fixes drift smoothly off the true path; the along-track error is opposite-signed at the two gates of a high-curvature section, compressing/stretching its time by up to ~1.4 s | 10–40 m | corpus lateral envelope; driven-distance band; line-length consistency residual | section *time* of curved corners | the three physical σ-nets below demote it; `s_hat` reverts toward the honest OBD-anchored value; excluded from tier A |

**Mode 4 is why this project exists.** It passes every spatial and fix-rate guard
— a Mode-4 lap can read `track_dist_offset_m` ≈ 30 m, log at full rate, and look
perfectly reliable — yet it silently compresses a section time. Roughly 24% of laps
have a >40 m excursion somewhere, independent of fix rate. The corpus consequences
before this layer: **77.5% of per-corner top-5 "fastest through" entries were
physically impossible-short**, and one verified T3 ghost read **39.79 s stamped
`timing_reliable=True`** (honest ≈5.2 s) because the old flag and value came from
different passes.

### The three physical σ-nets (Mode 4)

No scalar threshold separates a Mode-4 fake short path from a genuine tight racing
line — T11's real −33 m line and a T8 fake −47 m drift have overlapping magnitudes
(magnitude thresholds were reverted twice for exactly this reason, design R2).
Discrimination comes from *physical* evidence, and every net **inflates σ, never
rejects** (R2):

1. **Corpus lateral envelope** ("the asphalt ribbon"): per-10 m-bin p2/p98 of
   clean-lap signed lateral offset. Evidence outside gets weight `exp(−(excess/5m)²)`.
   Catches off-ribbon drift (a T8 fake runs 18 m outside p2 → weight ≈2e-6) while
   leaving a genuine in-ribbon tight line untouched.
2. **Driven-distance band** at the posterior crossings: `band = (∫|κ| ds)·6 + 7 m`;
   `excess = max(0, |driven − (b−a)| − band)` inflates σ by `excess/v̄`. Catches
   *on-ribbon but odometer-inconsistent* drift the envelope can't (a T8 fake that
   sits in-ribbon yet drove 209 m over a 222 m floor).
3. **Line-length consistency residual**: does the lap's own lateral line explain its
   odometer shortening? `resid = driven − [(b−a) + Σ κ_signed·e_left·Δs]`; clean std
   6.6 m. Catches a fake independently of the envelope.

Each net was demonstrated to catch a real case the others miss; all three run in
production, always emitted as structured audit records (R12) in `checks_json`.

## The four trust axes

Trust is not one boolean per lap — it depends on the question:

| Question | Trustworthy source | Fails under |
|---|---|---|
| Channel *value* at a sample (speed/throttle/G) | OBD | never |
| *Relative* along-track motion ("how far did I drive") | OBD `dist_lap_m` | never |
| *Absolute* along-track position / section time (`s_hat`) | GPS evidence, corridor-weighted | Modes 2–4 → **σ widens honestly** |
| *Lateral* line position (apex offset) | GPS lateral offset | Mode 1 → bounded by envelope width (see Honest limits) |

**Principle:** *OBD tells you how far; GPS tells you where — and every "where"
carries its own σ.* When GPS drops to ~1 Hz in a hairpin, the information needed to
place a gate to better than ~±30 m is simply not in the log; no fusion recovers it,
so the transit is reported wide and excluded from rankings, never "repaired."

## Section timing, ranking, and tiers

`analysis.span_time` / `section_times` / `range_section_times` are thin shells over
`estimate_trajectory` (once per lap) + `section_timing` (per corner). Each row is
**always emitted** (design R10) with `status ∈ {ok, no_coverage, rescale_invalid}`,
a NaN value where uncomputable, and its `sigma_t_s`, `driven_m`, `rank_eligible`,
and `checks_json` audit records.

Section σ is **correlation-aware** (independence is wrong-signed for Mode 4):
`Var(T) = [σ_A² + σ_B² − 2ρσ_Aσ_B] / (v_A·v_B)` with `ρ = RHO_SECTION`. A common-mode
Mode-1 offset partly cancels (sharper straights); anticorrelated Mode-4 error widens.

Quality **tiers are derived from σ, never stored as independent booleans**:

| Tier | σ_t | Meaning |
|---|---|---|
| **A** | ≤ `TIER_A_S` (0.10 s) | rankable — `rank_eligible = True` |
| **B** | ≤ `TIER_B_S` (0.30 s) | shown, shaded, labelled "estimate ± σ"; excluded from ranking |
| **C** | otherwise | listed, excluded |

> The 0.10 s / 0.30 s thresholds are **PROVISIONAL** pending a coverage/precision
> sweep (design §9.1, reserved). Do not treat them as finalized.

Only tier-A (`rank_eligible`) transits win the "fastest through" benchmark — a
Mode-3/4 or coarse-GPS lap has wide σ and is excluded (design R2: σ decides
rankability), but it is surfaced with its estimate ± σ, never silently dropped.

Bridged (tier-B/C) values are the OBD prior dressed up, not a measurement — the UI
labels them "estimate ± σ", never "corrected" (R11).

## Reliability flags: σ-tier vs spatial

Two families of reliability flag now live in `corners.parquet`, and they answer
**different questions** — keep both:

- **σ-tier** (`transit_reliable_traj` = per-corner `rank_eligible`;
  `lap_reliable_traj`): does the trajectory layer trust this corner's **along-track**
  section time? This is the flag the timing/ranking consumers read.
- **Spatial** (`transit_reliable`, `lap_reliable`): is the lap's raw kd-tree
  projection (`track_dist_offset_m` + drift disagreement, the STANDARD tier) clean
  enough to trust its **lateral** line/apex position? These still gate the
  line-position metrics the trajectory layer does **not** certify (see Honest
  limits), so they are retained, not merged into the σ tier.

**The pass rates differ, and that is expected — not a regression.** On the ridge
corpus the spatial flag passes ~93% of transits while the σ tier passes ~76%. The
σ tier is stricter *by construction*: it demotes the Mode-3 coarse-GPS and Mode-4
along-track-drift transits that the spatial flag **structurally cannot see** (a
Mode-4 lap is spatially clean — full fix rate, ~30 m lateral offset — yet its
section time is compressed). The ~17-point gap *is* that Mode-3/4 population, now
visible for the first time. It is the whole point of the layer, not a loss of
coverage.

## Alignment invariant (do not break)

`labeler._attach_trajectory` and `trajectory.estimate_trajectory` **both** sort a
lap's samples by `t` with `kind="stable"`, and the estimator returns `s_hat`/`σ`
positionally aligned to *its* sorted order. The labeler relies on that alignment to
`.assign(s_hat=..., sigma_m=...)` onto its own stably-sorted frame. Real sessions
have tied timestamps (up to ~131 duplicate-`t` rows per session), so a non-stable
sort on either side would silently misalign `s_hat` against the samples. **If you
change the sort in one file, change it in both** (the visualizer's per-lap trace
helpers hold the same contract).

## Honest limits (unchanged by any design)

- In-corridor, odometer-consistent drift is observationally identical to line
  choice — absorbed with honest σ (bounded by envelope width, ~0.1–0.3 s in
  hairpins), not resolved. No detector fixes this without new sensors.
- This layer certifies the **along-track** axis. Lateral line-position metrics
  (apex offsets) get an honest `s` but not yet honest lateral positions — that is
  why the spatial `transit_reliable` flag is kept for line/apex queries, and the
  drift-field diagnostic is the seed for a future lateral certificate.
- Mode-3 hairpins stay irreducibly uncertain (the information is not in the log) —
  reported wide, excluded from rankings.

## History

This model supersedes an earlier confidence-based section-timing design that
computed a per-gate `crossing_gap_s` and a `timing_reliable = gap < CONFIDENCE_GAP_S`
flag from a *separate* pass over masked GPS, alongside independent silos
(`drop_gps_glitches`, the fused-axis mask, a spatial `_gates_glitch_free` check).
Those silos each saw one shadow of the underlying problem and were retired in the
2026-07 migration (PRs 0–5): value and confidence now come from one pass, and
`grep` finds exactly one place — the trajectory layer — that decides whether GPS is
trustworthy. The full story of that redesign — why the silos accumulated, the dead
ends, and what generalises — is in [DESIGN-JOURNEY.md](DESIGN-JOURNEY.md).
