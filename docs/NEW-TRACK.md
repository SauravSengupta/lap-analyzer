# Adding a new track

The pipeline is track-agnostic. Adding a track is mostly mechanical — run the
CLIs in order — with **one genuinely manual step** (pinning each corner's apex on
Google Maps) and **a couple of judgment calls** the data can't make for you.

This walkthrough uses the bootstrap of Portland International Raceway (PIR) as
the worked example, since that's the second track and exposed every single-track
assumption the original Ridge-only code had baked in. Budget ~1 hour of hands-on
time (almost all of it Maps pinning) plus a few minutes of compute.

Run everything from the repo root with `PYTHONPATH=.` (see [PIPELINE.md](PIPELINE.md)).
Examples below use `<slug>` for your track id (e.g. `pir`).

---

## What you need before starting

1. **Raw TrackAddict CSVs** in `data/raw/<slug>/`. Three or more sessions is the
   practical minimum — the pace-decile comparison bands collapse with fewer.
2. **The official track length and direction** (clockwise / counter-clockwise) —
   from the track's website or a track map.
3. **Start/finish coordinates** — pull them from any of the track's CSV headers:

   ```powershell
   Get-Content (Get-ChildItem data/raw/<slug>/*.csv | Select-Object -First 1).FullName -TotalCount 12
   ```

   Look for the `# End Point: <lat>, <long> @ ...` comment line.
4. **A satellite map you can pin on** — Google Maps satellite view. You'll be
   right-clicking corner apexes to read off coordinates.

---

## Step 1 — Minimal track JSON skeleton

Create `tracks/<slug>.json`. Start minimal; the corners and anchors get filled in
later by the seeding step. (The `_doc` field in `tracks/ridge.json` documents
every field — copy it.)

```json
{
  "track_id": "pir",
  "name": "Portland International Raceway",
  "configuration": "full",
  "official_length_m": 3166,
  "direction": "clockwise",
  "start_finish": { "lat": 45.595014, "long": -122.694484 },
  "lap_length_internal_m": 3166,
  "reference_lap": null,
  "calibration_anchors": [],
  "corners": []
}
```

Set `lap_length_internal_m` to the official length for now — Step 3 refines it
from the data.

## Step 2 — First normalize pass

```powershell
python -m lap_analyzer.cli.normalize --track <slug> --all
```

Expect some sessions to report `noobd` (no OBD channels logged — ~3–4% of
sessions; they're skipped, which is fine). Confirm the rest produce
`data/sessions/<slug>/<sid>/{samples.parquet, laps.csv, meta.json}`.

## Step 3 — Set `lap_length_internal_m` from real data

`lap_length_internal_m` is the per-lap rescale target — it should be the
**median GPS/OBD-integrated lap distance**, not the official length. Compute it
from the raw `dist_m` deltas in `samples.parquet`:

```python
import pandas as pd, glob, numpy as np
ds = []
for f in glob.glob("data/sessions/pir/*/samples.parquet"):
    s = pd.read_parquet(f, columns=["lap", "dist_m"])
    per_lap = s.groupby("lap")["dist_m"].agg(lambda x: x.max() - x.min())
    ds += [d for d in per_lap if 2000 < d < 4000]   # flying-lap band for a ~3.2km track
print(round(np.median(ds)))
```

> **Do not use `dist_lap_m` or `laps.csv`'s `lap_dist_m` for this.** Both are
> *rescaled* to whatever `lap_length_internal_m` already is, so they'll just echo
> your current guess. Only raw `dist_m` carries the independent measurement. (This
> tripped up the original plan — see [PIPELINE.md](PIPELINE.md) on `dist_lap_m`.)

PIR resolved to **3198 m** (vs official 3166 — ~1% longer, normal for race-line +
GPS-integration noise). For reference, Ridge's raw median is ~3936 m. Update
`lap_length_internal_m` in the JSON, then **re-normalize so `dist_lap_m`
rescales correctly**:

```powershell
python -m lap_analyzer.cli.normalize --track <slug> --all --force
```

## Step 4 — Maps-pin curation (the one manual step)

This is the only hand-curated geometric input, and everything downstream
(drift correction, the centerline, every `apex_m`) derives from it. Spend the
time here.

For each numbered corner, open Google Maps satellite view, find the **visual
inside-kerb apex** (where you'd clip the kerb on a good lap), right-click it, and
copy the `lat, long`. Save them to `data/notes/<slug>_apex_pins.json` mirroring
`data/notes/ridge_apex_pins.json`:

```json
{
  "_doc": "Ground-truth apex locations from Google Maps satellite imagery. User-pinned visual kerb-touch locations, not recorded GPS. Pin precision ~5-10m. (lat, long) in WGS84.",
  "_source": "Manual pinning via Google Maps right-click, 2026-05-20",
  "T1":  {"lat": 45.59616, "long": -122.69815, "note": null},
  "T2":  {"lat": 45.59653, "long": -122.69839, "note": null}
}
```

Judgment calls that came up pinning PIR (yours will have analogues):

- **Multi-apex / double-apex corners.** PIR's T4 is one driven corner with two
  visual apexes: `T4b` (the real, regularly-hit apex) and `T4a` (a "throwaway"
  apex the driver rarely hits). Pin **both** with distinct keys (`T4a`, `T4b`).
  Step 6 folds them into a single corner with a `secondary_apex_m`, and `T4a` is
  deliberately *excluded* from the calibration anchors (a rarely-hit apex would
  skew drift correction toward a line nobody drives).
- **Full-throttle "corners."** PIR's T8 and T9 are taken flat — there's no
  speed-minimum or sharp lat-G peak to cluster on. Pin the *nominal* apex anyway
  (where the kerb is), and mark it: `"note": "full-throttle through this corner —
  apex more nominal than driven"`. Step 6 handles these specially.
- **Pin precision is ~5–10 m.** You won't be perfect. The drift correction's
  median-over-anchors design tolerates a few sloppy pins. But if Step 9's
  validation shows clustered driving data consistently offset >~5 m from a pin,
  re-check that pin against the satellite image before trusting either — the pin
  might be off, or the corner's racing line might genuinely sit off the kerb.

## Step 5 — Candidate scan

```powershell
python -m lap_analyzer.cli.extract_corner_candidates --track <slug>
```

Writes `data/corpus/<slug>_candidates.csv` — every lat-G peak across every flying
lap. This is the cluster pool the seeder maps your pins onto.

## Step 6 — Seed corners + calibration anchors

`scripts/seed_pir_corners.py` populates `corners[]` and `calibration_anchors[]`
in the track JSON by combining the pins (Step 4) with the candidate clusters
(Step 5). Its method:

1. Project each pin's GPS onto sample-lap trajectories to get its lap-distance
   (median across several clean laps; ~3–6 m precision).
2. Cluster the candidate `peak_dist_m` values **separately for left and right
   turns** (density-peak detection). Splitting by direction is what lets
   chicanes and double-apex corners resolve — pure 1-D distance clustering (the
   old Ridge `scratch/cluster_ridge.py` approach) collapsed adjacent
   opposite-direction peaks into one blob.
3. Map each pin to the nearest **direction-matching** cluster; read corner
   direction from the cluster's lat-G sign.
4. `start_m`/`end_m` = `apex_m ± 80 m` initially (the labeler refines bounds
   later from data).
5. Calibration anchors = the pin coordinates themselves (visual / kerb-touch).

The PIR special cases, encoded in that script, are the template for the judgment
calls above:

- **T4** → primary apex from the `T4b` cluster, `secondary_apex_m` from the `T4a`
  cluster, one corner total. `T4a` omitted from anchors.
- **T9** → no nearby lat-G cluster (full-throttle), so it takes the
  **pin-projected lap-distance directly** instead of a cluster median.

Run it dry first, eyeball the report (it prints each corner's source — cluster vs
pin-projected — and any unmapped clusters), then write:

```powershell
python scripts/seed_pir_corners.py            # dry run, prints report
python scripts/seed_pir_corners.py --write     # mutates tracks/pir.json
```

> **`seed_pir_corners.py` is PIR-specific as written** — hardcoded paths, the
> `EXPECTED` left/right direction list, and the T4/T9 special cases. For your
> track, **copy it** to `scripts/seed_<slug>_corners.py` and adapt the `EXPECTED`
> list and special cases. Generalizing this into a `--track`-driven CLI is
> deferred tooling, not yet built.

## Step 7 — First-pass labeling (bootstrap reference)

`label_corners` assigns `track_dist_m` by projecting each sample onto a
**reference lap** — and the canonical reference is the synthetic centerline,
which doesn't exist yet. Chicken-and-egg. Break it by pointing `reference_lap` at
a **real, clean recorded lap** you trust to be glitch-free, just for this first
pass:

```json
"reference_lap": { "session_id": "20240829-103904", "lap": 2 }
```

```powershell
python -m lap_analyzer.cli.label_corners --track <slug>
```

This produces a first-pass `corners.parquet` and the drift columns on
`samples.parquet` — enough to build a proper centerline in the next step.

## Step 8 — Build + install the synthetic centerline

```powershell
python -m lap_analyzer.cli.build_centerline --track <slug>
python -m lap_analyzer.cli.install_centerline --track <slug>
```

`install_centerline` prints a JSON patch for `reference_lap` — apply it so the
block now points at the synthetic centerline:

```json
"reference_lap": {
  "session_id": "_synthetic_centerline",
  "lap": 1,
  "notes": "Synthetic centerline at 1m resolution built from corpus median."
}
```

> **The lap-wrap centerline gotcha.** `build_centerline` reads the corner boxes
> from your track JSON and treats them as *protected ranges*. This matters: at
> the start/finish wrap, the kd-tree can assign samples from physically different
> positions to the same `track_dist_m`, producing a visible "zigzag" in the
> front-straight centerline. The fix rejects high-spread / piled-up bins **only
> outside corner boxes**, then periodic-interpolates across the wrap and applies
> heavier smoothing on straights. So **your `corners[]` must be populated before
> you build the centerline** (Step 6 before Step 8) — otherwise nothing is
> protected and real corner line-variation gets flattened. PIR is what exposed
> this; Ridge's wrap happened to be clean enough to never show it.

## Step 9 — Second-pass labeling, quality flags, corpus

Re-label against the now-installed centerline, then flag quality and build the
corpus:

```powershell
python -m lap_analyzer.cli.label_corners --track <slug>
python -m lap_analyzer.cli.flag_quality --track <slug>
python -m lap_analyzer.cli.build_corpus --track <slug>
```

---

## Validation checklist

Sanity-check the corpus before trusting it:

```python
import pandas as pd
df = pd.read_parquet("data/corpus/pir_corners.parquet")
print(df.shape, round(df["transit_reliable"].mean(), 3))
print(df.groupby("corner_id").size())
```

- **`transit_reliable` rate in the 0.85–0.95+ ballpark.** PIR came in at 97.8%
  transit-reliable / 95.5% lap-reliable; Ridge at ~92% / ~84%. A markedly lower
  rate means drift correction is struggling — usually bad pins (see below).
- **Every corner has roughly the same transit count** (≈ number of flying laps).
  A corner with far fewer transits is being mislabeled or its box is wrong.
- **`apex_dist_offset_m` should match corner type.** It's `min_speed_dist_m −
  apex_m` — i.e. how far past the visual kerb the car was slowest:
  - Slow precise corners where you reliably clip the kerb → ≈ 0.
  - Trail-brake corners → negative (slowest before the kerb).
  - Throttle-through / full-throttle corners → positive (slowest at exit), or
    noisy for the full-throttle ones.
- **Pin vs data offset.** If a corner's clustered data sits consistently >~5 m
  off its Maps pin, re-examine the pin (Step 4) before adopting either source.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Physical points don't line up across laps; `dist_lap_m` for the same corner varies a lot lap-to-lap | `lap_length_internal_m` wrong — redo Step 3 from raw `dist_m`, re-normalize `--force`. |
| Low `transit_reliable` rate, high `gps_drift_disagreement_m` | Pins too far from the actual kerbs — drift correction has no consistent anchor. Re-pin the worst corners. |
| Front-straight "zigzag" in the centerline | Corner boxes weren't populated before `build_centerline`, so nothing was protected. Populate `corners[]` (Step 6), rebuild. |
| A full-throttle corner has almost no transits / nonsensical apex | No lat-G cluster to seed from. Use the T9 pattern — pin-projected lap-distance directly (Step 6). |
| `label_corners` errors with "no reference_lap defined" | First pass needs a real bootstrap reference (Step 7); set `reference_lap` to a real session+lap before the centerline exists. |

---

## Writing track-specific analysis pages

The main visualizer (`visualizer/app.py`) is fully track-agnostic — it derives
all corner geometry from the selected track's JSON. But bespoke worked examples
(like the Ridge T8→T11 downshift study in `visualizer/pages/`) hard-code corner
IDs and tuned thresholds for one track. **Guard them** so selecting another track
renders an explainer instead of crashing:

```python
from shared import current_track
track = current_track()
if track != "ridge":
    st.info("**This is a Ridge-only worked example.** … "
            "See docs/NEW-TRACK.md for how to write one for your own track.")
    st.stop()
```

See `visualizer/pages/1_Ridge_Downshift_T8-T11.py` for the full pattern. To write
your own page: copy that file, swap in your track's corner IDs, gate on
`current_track() == "<slug>"`, and lean on the analysis helpers in
`lap_analyzer/analysis.py` (`section_times`, `span_time`, `derive_gear`, etc.),
which are all track-agnostic.
