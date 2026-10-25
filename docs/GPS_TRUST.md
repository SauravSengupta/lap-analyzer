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

## Input calibration: GPS speed and position are re-timed upstream

Before any of the above runs, `normalize` removes two measured GPS lags (see
[PIPELINE.md](PIPELINE.md) and ARCHITECTURE.md decision 7): a per-lap lag on
`speed_mph_gps` against OBD speed (typically ~0.45 s), and ONE per-session lag
on GPS **position** (lat, long, altitude, accuracy, sector, and the TrackAddict
Lap counter) measured directly as GPS path length against the OBD odometer
(~0.15–0.18 s; session median 0.18 s on both tracks). Speed lags position by a
further ~0.12 s, so the two are corrected separately. Heading is left as logged.

For **OBD sessions** `dist_lap_m` (OBD-integrated) is unchanged, but
`track_dist_m` and the trajectory corridor are built from position and so move
with the correction: OBD events on `track_dist_m` land at their true position
(~0.18·v earlier) and lap boundaries move earlier by τ_pos·v (~8 m at S/F
speed). The corridor must be rebuilt after re-normalizing (PIPELINE.md). For
**GPS-only** sessions (~12% of sessions, no OBD), `speed_mph_gps` *is* the speed
backbone — `dist_m`/`dist_lap_m` integrate it, and `trajectory.py` and the corner
labeler use it directly — so the speed re-timing puts that backbone on the
same clock as position; their τ_pos uses the `offset` fallback.

The trust layer's bands were effectively tuned on lagged positions. The
correction widens PIR T1's odometer-vs-gate gap by ~5–7 m (entry ~45 m/s, exit
~22 m/s), past T1's 8.1 m driven band, so T1 `rank_eligible` fell 80% → 66%
(accepted-unmet 2026-09-29; every other corner and both tracks' overall rates
are within tolerance). Recalibrating the bands on corrected data, for all
corners, is the open follow-up. A session's `meta.gps_lag` block being absent or
older than `gps-lag-v3` means it is stale.

## Always trustworthy: OBD

The OBD channels — `speed_mph`, `throttle`, `long_g`, `rpm`, and `dist_lap_m`
(OBD-integrated distance, per-lap rescaled to the canonical lap length; endpoints
pinned) — are high-rate and glitch-free. Anything measured *from OBD* — a channel
value at a sample, or "how far has the car driven" — is trustworthy. (Exception:
~12% of sessions log without OBD; those fall back to GPS speed, are flagged
`obd_present=False` / `speed_source="gps"`, and the trajectory backbone becomes
`∫ v_gps dt` with its own rule-derived σ floor, `SIGMA_FLOOR_GPS_M`, currently equal
to the OBD floor — see Mode notes below.)

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

**Short bursts are caught by the car's own motion (jump detector, 2026-10).** Before
the δ̂ fit, every GPS fix is compared with the path shape the GPS-independent sensors
give over ±1.5 s — OBD odometer distance plus heading from lateral g (yaw =
−lat_g·g/v) — rigidly fitted to the neighbouring fixes
(`lap_analyzer/gps_consistency.py`). A fix more than `JUMP_THRESH_M` (4.8 m: clean-lap
p99.9 of 3.82 m + 1 m) off that path is dropped from evidence, and σ grows through the
gap term — nothing is reconstructed. It never judges a fix it can't: no lat_g, slow
car, missing speed nearby, or too few neighbours (1 Hz GPS). This catches the 5–30 m
spikes and 0.4–1 s bursts that the Hampel/backstop filters (15 m floor) pass. On
planted faults at section boundaries, wrong-but-rankable bursts fell from 41.5% to
1.0% and planted slow drift from 13.0% to 7.9%, with tier-A coverage unchanged; about
5% of transits carry at least one flagged fix (`gps_jump_n`). It cannot see drift
slower than ~1.5 s — the phone logs no gyro and lat_g heading drifts 8–20°/min — so
Mode 4 stays with the σ-nets below.

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
2. **Driven-distance band** at the posterior crossings: `band = (∫|κ| ds)·DRIVEN_BAND_HDG_SCALE + DRIVEN_BAND_FLOOR_M` (3.2 and 10.3 m since 2026-10);
   `excess = max(0, |driven − (b−a)| − band)` inflates σ by `excess/v̄`. Catches
   *on-ribbon but odometer-inconsistent* drift the envelope can't (a T8 fake that
   sits in-ribbon yet drove 209 m over a 222 m floor).
3. **Line-length consistency residual**: does the lap's own lateral line explain its
   odometer shortening? `resid = driven − [(b−a) + Σ κ_signed·e_left·Δs]`; clean std
   `CONSISTENCY_STD_M` = 5.1 m (2026-10). Catches a fake independently of the envelope.

Each net was demonstrated to catch a real case the others miss; all three run in
production, always emitted as structured audit records (R12) in `checks_json`.

**Re-derived 2026-10 (W2).** The constants were re-derived by their original rules
(`scripts/derive_trust_constants.py`) on lag-corrected, jump-filtered data, calibration
split only: `RHO_SECTION` 0.45 → 0.66 (Fisher-z mean gate-δ correlation, 28 corners);
driven band `·6 + 7 m` → `·3.2 + 10.3 m` (least squares of per-corner robust bounds
vs heading); `CONSISTENCY_STD_M` 6.6 → 5.1 m; `SIGMA_PRIOR_MID_M` 12.0 → 6.5 m. Held-out validation, W0 → W1 → W2:
burst-at-gate (0.4–1 s, ≥10 m) bad-but-rankable 41.5% → 1.0% → 4.3% (the drop is W1's
jump detector; W2 gave some back, still within the 5% gate); slow-drift
bad-but-rankable 13.0% → 7.9% → 12.9% (W2 gave back most of W1's gain, still not worse
than W0); PIR T1 OBD rank-eligible 0.874 → 0.951. The band rule is a central
least-squares line through per-corner bounds, not an envelope, so roughly half the
corners' clean 3σ bound sits above the band; that is the source of the coverage
trade-off: per-track OBD tier-A coverage −1.6 pp (PIR 0.913 → 0.896, Ridge 0.863 →
0.847). The brake-point scatter gate was missed on PIR (13.091 → 13.132 m, +0.04 m vs
W0; Ridge 13.612 → 13.317 m): accepted-unmet by the user 2026-10-20. Likely (not
proven) cause: the two σ-shrinking changes, `RHO_SECTION` 0.45 → 0.66 and
`SIGMA_PRIOR_MID_M` 12 → 6.5.

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

GPS-only (`gps_backbone`) laps are tiered by their own σ (since 2026-10, W3); there
is no blanket tier-B cap. Evidence: dropping OBD from tier-A OBD laps changes section
times by a median 0.006 s (p90 ≤ 0.033 s). `SIGMA_FLOOR_GPS_M` is derived by rule
(smallest floor in a 0.5 m grid at which GPS-only emulation is ≥ 95% within 2σ); the
rule passed at the grid's first point (1.0 m, equal to the OBD floor), so it did not
constrain the value. On the calibration split (2244 transits) emulation is 0.998
within 2σ, tier-A share 0.966, and 99.9% of tier-A transits within 0.10 s of the OBD
answer; on held-out sessions 0.997 within 2σ and 99.9% of 1653 emulated GPS-only
tier-A transits within 0.10 s. Caveat: the floor rule is validated on emulated GPS-only
laps (OBD laps with OBD channels removed); real GPS-only sessions may have worse GPS,
and on them the driven-band "odometer" is GPS speed, so σ alone is not enough. A
GPS-only section therefore reaches tier A only if, besides a tier-A σ, it passes the
accel-speed check below **and** both σ-nets (driven-band, line-length consistency). On
an OBD lap a failed net only inflates σ; on a GPS-only lap it is GPS disagreeing with
itself, with no independent sensor to settle it, so it withholds tier A (A → B, σ
unchanged). Why: PIR 20260612-103325 L4 T8 ranked GPS-only fastest at 4.245 s (−0.226 s
vs the OBD best) with σ 0.034 s, although its consistency net failed and both
neighbours failed the driven-band net. That is a gate-position fault, which the
accel-speed check (a speed check) cannot see; Ridge 20260517-111618 L3 T12 (−0.127 s)
showed the same pattern. Coverage now (status-ok sections): PIR 83.5% rank-eligible
overall, 61.1% of 624 GPS-only sections (from 0% before W3); Ridge 78.4% overall, 83.6%
of 304 GPS-only. Two GPS-only corner-fastest times remain: PIR 20251004-112120 L5 T11
(−0.174 s, consistent with its neighbours on review) and Ridge 20251018-131451 L3 T9
(−0.031 s, within noise).

GPS-only accel-speed check (W3). Why: a confirmed fake on PIR 20260926-111509 L2 T1,
GPS-only 4.307 s, tier A and corner-fastest. GPS speed said 116.6 to 83.7 mph while
integrated long_g said 116.8 to 49.9 mph; on GPS-only laps the odometer is GPS speed,
so no net had an independent sensor. The check, per section: backbone Δv minus
∫long_g·g dt (`accel_speed_resid_mps`), centred on the corner's clean median, with the
corner's p99 |centred| as tolerance, derived on GPS-only emulation of clean calibration
OBD laps (2244 sections; `scripts/derive_accel_speed_tol.py`, references in
`tracks/{track}.json` `accel_speed_check`). Tolerances run from 2.4 to 31.7 m/s: strong
on low-noise corners (PIR T1 3.9, T8 3.3, T10 3.0; Ridge T16 2.4), weak on high-noise
ones (Ridge T6 31.7, PIR T12 20.8). The fake now measures 15.2 m/s centred against 3.9,
so it is tier B. GPS-only sections with no reference (multi-corner ranges) or without
accelerometer data do not reach tier A; OBD laps carry the record as a diagnostic only. The check
is a speed check only: it catches GPS speed that disagrees with the accelerometer, not
a correct-speed lap whose gate crossings are misplaced (that is the σ-nets' job, above).
It is also weak where the tolerance is wide: 8 of 28 corners have tol ≥ 15 m/s (Ridge T6
31.7, T8 21.1, T11 16.2, T12 15.9; PIR T12 20.8, T6 19.8, T4 18.7, T5 15.0), because the spread is
accelerometer noise that varies between sessions (the effective sample size is the number
of sessions, not sections). Per-session accelerometer calibration is the follow-up.

Only tier-A (`rank_eligible`) transits win the "fastest through" benchmark — a
Mode-3/4 or coarse-GPS lap has wide σ and is excluded (design R2: σ decides
rankability; GPS-only laps must also pass the cross-checks above), but it is surfaced with its estimate ± σ, never silently dropped.

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
