# Design journey: how the pipeline learned to distrust GPS

This is the story of the one part of Lap Analyzer that was rebuilt from scratch,
and why. It's here because the *reasoning* is more useful than the result: if you
build anything that fuses a trustworthy sensor with a flaky one, you will probably
walk into the same wall we did.

For the model as it stands today, read [GPS_TRUST.md](GPS_TRUST.md). This doc is
the "how we got here" — the change in direction, the dead ends, and the lessons.

## The setup: one honest sensor, one liar

Every lap carries two independent position signals. **OBD** (read off the car's
bus) gives speed, throttle, rpm, longitudinal and lateral g, and an integrated
odometer distance — all high-rate and effectively glitch-free. **GPS** gives
absolute position on the map. Everything interesting in cross-lap analysis —
"where exactly was my apex", "how long did T8 take" — needs position, which means
it needs GPS. And GPS lies, in four distinct ways:

1. **Lateral drift** — the whole path sits offset a few metres to one side.
2. **Teleports** — the position jumps or walks backward tens to hundreds of metres.
3. **Coarse fixes** — the receiver drops to ~1 Hz; position is accurate but only
   updated occasionally, so anything *timed* from it is off by up to half a second.
4. **Smooth local drift** — fresh, full-rate fixes that quietly wander off the true
   path. This one is invisible to every obvious check, and it's the villain.

The core asymmetry that drives the whole design: **OBD tells you how far you've
driven; GPS tells you where you are.** The odometer is trustworthy but carries no
map position; GPS carries map position but can't be trusted. Neither alone answers
"how long did T8 take."

## The trap: a detector per symptom

The pipeline grew the way most do — reactively. A GPS problem would surface in the
section-timing numbers, someone would characterise it, and a detector would be
written to catch *that* problem and bolted onto the section-timing code. Then a
different GPS problem would surface, and a different detector went in. Over time
this produced roughly six independent guards:

- an OBD-distance sanity reject,
- a 2-D gate-crossing scheme to replace it,
- an OBD-ratio band on top of that,
- a "was there a glitch near the gate" check,
- a "how coarse were the fixes bracketing the gate" confidence gap,
- plus separate masks living in other consumers: a sample-dropping glitch filter
  for the plots, a spatial reliability flag for the map, a drift correction, a
  fused-axis mask, a warning banner — each with its own signal and its own
  threshold.

Every new failure mode needed a new detector because each detector could only see
one shadow of the underlying problem. The threshold zoo grew: five different
distance limits and a timing cutoff, scattered across five files, one constant name
even colliding at two different values.

## The break: when the flag and the value disagreed

The design flaw wasn't any single detector — it was that **the timing value and the
confidence in it were computed by different code, from different evidence.** The
section time came from one pass (a 2-D gate crossing over raw GPS); the "is this
reliable" flag came from a *separate* pass over differently-masked GPS.

The moment that made this undeniable: a lap whose GPS teleported produced a path
that clipped a distant section's gate, and the crossing picker timed that corner at
**39.79 seconds** — a corner that honestly takes about five. And it was stamped
**`reliable = True`**. The confidence check had run its own independent pass, liked
what *it* saw, and certified a number the timing code never actually used. A flag
that can disagree with its value will always, eventually, bless a wrong one.

The corpus made the scale clear. Across ~6,900 transits: **77.5% of the per-corner
top-5 "fastest through" entries were physically impossible** — the odometer said
the car covered less distance than the inside line of the corner is long. Ranking
doesn't just tolerate the contamination, it *seeks* it: sorting for "fastest" pulls
the compressed-time fakes straight to the top. One in ten same-session lap pairs
showed the paradox of a "half-second faster" corner driven at a *lower* average
speed over the same stretch of odometer.

## The dead end: magnitude thresholds

The obvious fix — and the one we tried and reverted, twice — is to reject laps
whose driven distance is implausibly short. It fails for a subtle, fundamental
reason: **a genuine tight racing line and a Mode-4 GPS drift produce the same
symptom.** A driver who hugs the inside kerb really does cover less distance than
the centerline; so does a GPS track that drifts across the corner. Measured by
magnitude they overlap — a real tight line at one corner came in around −33 m from
the median, a GPS fake at another around −47 m. No scalar threshold sits between
them. When we shipped an OBD-ratio band anyway, **65% of the laps it rejected were
clean, faster-than-median laps** — it was throwing away exactly the good driving we
built the tool to find.

The lesson that unlocked everything: *magnitude is not discrimination.* A shorter
distance is evidence of a tight line **or** a drift, and telling them apart needs
*physical* evidence — is the car still on the asphalt? does its own lateral line
explain the shortening? — not a bigger or smaller number.

(There was a quieter bug hiding underneath, too: the gates that timed the sections
took their orientation from just two centerline points a couple of metres apart, on
a centerline with enough sample-to-sample jitter to rotate a gate by up to 59°. A
rotated gate turns an ordinary 3–5 m lateral line offset into a tenth of a second of
phantom timing error — on *clean* laps. Any real fix had to remove that too.)

## The reframe: one estimate, one sigma

The redesign started by refusing to add a seventh detector. Instead of asking "is
this lap glitched?" (a boolean, per detector, per consumer), it asks a single
question once per lap:

> **Where was the car on the track, and how sure are we — as a distance, with an
> honest ±σ?**

One module estimates, per lap, the car's monotone position on the canonical track
ruler:

```
s_hat(t) = dist_lap_m(t) + delta_hat        # OBD odometer, nudged onto the map
sigma(s)                                     # honest 1-sigma, from the SAME pass
```

The OBD odometer is the smooth, trustworthy backbone. `delta_hat` is a robust,
physically-constrained smooth of the GPS-vs-OBD offset, measured only on the GPS
fixes that survive teleport rejection. Where GPS evidence is good, `delta_hat`
follows it; where it's sparse or rejected, `delta_hat` relaxes back toward the
odometer and **σ grows**. Every consumer — section times, rankings, the plotting
axis, the corner metrics, the quality flags, the banner — reads that one `(s_hat,
σ)`.

Three things fall out of the shape of this, not from any threshold:

- **A value can never again disagree with its confidence,** because they're the
  same pass. The 39.79 s ghost is impossible by construction: `s_hat` is monotone,
  so it crosses any gate position *exactly once* — a teleport can't manufacture a
  second crossing at a distant gate.
- **Bad GPS makes the answer wider, not wrong.** A coarse-fix corner or a drifted
  corner doesn't get dropped or "corrected" — it gets a large σ, is shown as an
  honest "estimate ± σ", and is excluded from rankings. Nothing is silently censored.
- **Discrimination is physical, and only ever inflates σ.** Three nets do the work
  no threshold could: a corpus "asphalt-ribbon" envelope (is the car still on
  known-driven pavement?), a driven-distance band, and a line-length consistency
  check (does the lap's own cornering explain its odometer shortening?). A real
  tight line passes all three; a drift fails at least one. None of them ever
  *rejects* a lap — they widen its uncertainty, and uncertainty decides rankability.

## What it cost, and what it didn't

The honest part: some GPS drift is **observationally identical to a real line
choice** — a fix that wanders smoothly across a corner while staying on plausible
pavement and staying odometer-consistent cannot be distinguished from a driver who
genuinely took that line. The layer doesn't pretend to resolve it; it absorbs it as
a modest honest σ. It certifies the *along-track* axis (how long, how far); it does
**not** yet certify *lateral* position (exact apex offset), which is why the older
spatial reliability flag is kept for line-and-apex questions. Some Mode-3 hairpins
stay irreducibly uncertain because the information simply isn't in the log — those
are reported wide and excluded, not repaired.

The satisfying part: replacing six detectors with one estimator made the codebase
**smaller** — the cleanup PR alone removed a net ~540 lines — and `grep` now finds
exactly one place in the whole repo that decides whether GPS is trustworthy. The
screenshot that kicked this off — a real lap reading "+1.16 s vs best" against a
physically-impossible 7.07 s benchmark — now renders as an honest estimate ± σ
against a benchmark that survives an adversarial audit.

## Lessons, generalised

If you take four things from this:

1. **Compute a value and its confidence from the same evidence, in the same pass.**
   The instant they come from different code, they will eventually disagree, and a
   confidence that blesses a value it didn't produce is worse than no confidence.
2. **Magnitude is not discrimination.** If two different causes produce the same-sized
   symptom, no threshold on that symptom separates them. Find evidence that
   *distinguishes the causes*, not a better cutoff on the effect.
3. **Prefer widening uncertainty to dropping data.** "Reject" throws away real signal
   (65% of ours was good laps) and hides the decision; "inflate σ" keeps everything
   visible and lets the ranking exclude what it should while surfacing the rest.
4. **One trust decision beats N detectors.** Every consumer reading one honest
   estimate is not just less code — it's the only way to guarantee they can't
   contradict each other. A pile of per-symptom guards is a pile of future
   disagreements.

The design that came out of this was chosen deliberately — a panel of competing
approaches, scored by independent adversarial review, with a prototype validated on
real laps before a line of production code was written — but the four lessons above
are what generalise past this one codebase.
