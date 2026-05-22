# GitHub Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the personal `lap-analyzer` repo so it can live on GitHub as a usable open-source project, with a second track (PIR) wired in to flush out single-track assumptions, and ship with sample data + docs so a new user can launch the visualizer in 5 minutes from a fresh clone.

**Architecture:** Three phases of work — (1) make the pipeline genuinely track-agnostic in place (without moving files) so we can prove it on PIR, (2) bootstrap PIR end-to-end, (3) flatten the `app/` wrapper, add sample data + gitignore + docs + license. Tests are explicitly deferred to a follow-up TDD session per user request.

**Tech Stack:** Python 3.11+, pandas/pyarrow, Streamlit, plotly, scipy. Windows PowerShell host (no bash-style env-var syntax). Repo is not currently in git; `git init` + push is the last step.

---

## 📍 RESUME HERE (last updated 2026-05-22)

**Status:** Phases 0-3 done. **At CHECK-IN 3 passed — ready for Phase 4 (sample data + finalize .gitignore).** CHECK-IN 2 resolved: ORP deferred to post-publish, PIR locked.

**GIT IS NOW INITIALIZED (changed from the original plan).** User asked to set up git early to de-risk the Phase 3 restructure. Local-only, no remote yet (Phase 7 adds remote). Commits so far: `e3b51de` baseline (Phases 0-2), `5d9f6cc` Phase 3 restructure. `.gitignore` excludes data/ (samples carved back in Phase 4), .venv/, .claude/, scratch/, scripts/*.png. The old "comment out, don't delete" convention is relaxed (history preserved) — but don't retroactively delete pre-existing commented blocks without asking.

**Phase 3 done — repo flattened:** `app/src/lap_analyzer/`→`lap_analyzer/`, `app/visualizer/`→`visualizer/`, `app/tracks/`→`tracks/`, `app/docs/*`→`docs/`, `app/pyproject.toml`→root. config.py APP_ROOT parents[2]→[1], DATA_ROOT default `../data`→`data`. Run cmd now `PYTHONPATH=. python -m streamlit run visualizer/app.py` from root. Notebooks triaged: `scripts/` kept = seed_pir_corners, plot_pir_map, verify_ridge, obd_distance_spread, corner_envelope; everything else → gitignored `scratch/` (incl cluster_ridge as superseded). `app/` and empty `tests/` removed entirely. Smoke-tested: identical 1608 PIR transits, visualizer imports, keeper scripts run.

**Decisions locked in (Phase 0):**
- Repo: `lap-analyzer` · License: MIT · Data tier: B (samples + Release-asset corpus)

**Phase 1 deliverables (done, byte-equality verified on Ridge at the time):**
- `labeler.py` latitude parametrized; `centerline.py` + `cli/build_centerline.py` + `cli/install_centerline.py` (new); visualizer `current_track()` + sidebar picker + `LAP_ANALYZER_TRACK` override.

**Phase 2 deliverables (PIR is fully live):**
- `tracks/pir.json` — 12 corners (T1-T12) + 12 calibration_anchors. Direction `clockwise` (user-confirmed). `lap_length_internal_m=3198` (median GPS-integrated, NOT official 3166). `_doc` schema field added (also to ridge.json).
- `data/notes/pir_apex_pins.json` — 13 user Maps pins (T1-T12 + T4a/T4b variants). T4 modeled as one corner: primary apex from T4b cluster (841m), secondary from T4a (739m). T9 is full-throttle with no lat-G cluster → uses pin-projected apex (1676m); the R@1798m cluster is the slow back-straight curve, intentionally unmodeled.
- `app/notebooks/seed_pir_corners.py` (new) — seeds corners[]+calibration_anchors[] from candidate clusters + pins. `--write` flips dry-run→write. Default dry-run.
- `app/notebooks/plot_pir_map.py` (new) — renders the corner-zone map → `pir_track_map.png`. Use to visualize any corner-box change.
- PIR corpus built: **97.8% transit_reliable, 95.5% lap_reliable**, 1608 transits / 134 flying laps (every corner = 134 transits). 15 sessions processed, 3 noobd.
- Corner starts pulled back to throttle-lift (braking) points per user: T1→126, T4→648, T7→1284, T10→2198.

**Phase 2 code change with cross-track impact — `centerline.py` algorithm changed (applies to ALL tracks):**
- Root issue PIR exposed: lap-wrap `track_dist_m` misalignment created a front-straight "zigzag" in the centerline. Single-track assumption — Ridge's wrap happened to be clean enough to not show it.
- Fix: (1) reject high-spread/folded bins ONLY outside corner boxes (corners passed in via new optional `protected_ranges` arg to `build_centerline`; CLI derives from track corners); (2) periodic interpolation across the lap wrap (was linear); (3) corner-aware variable Gaussian smoothing (corners σ=2, straights blend to σ=10). New constants: `SPREAD_REJECT_M=6`, `N_FOLD_MULT=1.5`, `STRAIGHT_SMOOTH_SIGMA_M=10`, `STRAIGHT_BLEND_TRANSITION_M=25`.
- **Consequence: next Ridge centerline rebuild will shift slightly (improvement). `protected_ranges` is optional/defaulted so nothing breaks.**

**Plan corrections made this phase:**
- Task 2.2 Step 2 was WRONG about `lap_dist_m` (it's a copy of `lap_length_internal_m`, not OBD-integrated). Corrected in-place; correct method uses raw `dist_m` deltas from samples.parquet.

**Side effects from Phase 1 still relevant:** T8-T11 page renamed to `1_Ridge_Downshift_T8-T11.py` with `track!="ridge"` guard (verified: shows explainer + `st.stop()`, no crash on PIR) — Phase 5 owes `docs/NEW-TRACK.md` "writing track-specific pages". Backup-file filter in `available_tracks()` → backups move to `scratch/` in Phase 3 Task 3.6.

**Next action when resuming:** Phase 4 Task 4.1 — pick one Ridge + one PIR sample session (clean, top-third pace, all 4 outputs present), confirm with user, copy under `data/samples/`. Then Task 4.2 finalizes `.gitignore` (carve `!data/samples/` back in — note .gitignore already exists from the early git setup, so this is an EDIT not a create), Task 4.3 verifies sample-only mode (`DATA_ROOT=data/samples`). ORP deferred to post-publish.

**Execution conventions:**
- Subagent-driven was the CHECK-IN 0 decision, but Phase 2 ran mostly inline because the work was exploratory/iterative (data analysis with the user in the loop) — that's the right call when tasks need a tight feedback loop rather than a fixed spec.
- No git operations. Project gets `git init`'d at Phase 7.
- Comment out, don't delete. Tests deferred to a follow-up TDD session.
- PowerShell host: `$env:VAR = "val"; python ...`, never bash-style.

**Outstanding nits surfaced but deferred:**
- Pre-existing `use_container_width` Streamlit deprecation warnings — out of scope.
- "Direction: clockwise" is user-confirmed but the north-up map traces counter-clockwise; only affects diagnostics, user OK with it.

---

## Deferred work (out of scope for this plan)

- **Tests.** User wants a dedicated TDD session: review code → write spec → fresh session writes tests from spec without seeing code → run. This plan must NOT touch test scaffolding beyond keeping the empty `tests/` dir in the new layout.
- **ORP track.** Defer until after PIR ships; revisited at check-in 2.
- **Pluggable telemetry adapters.** TrackAddict CSV remains the only supported input. The plan documents the column contract; alternative-source support is a future feature.

## Workflow conventions

- **No git during migration.** Per user preference, retired code is commented out with a one-line reason, not deleted. Git is initialized once at the very end (Phase 7).
- **Check-ins are mandatory.** Five explicit check-in points are baked into the plan. Stop at each, report what changed and what the user needs to look at, and wait for the go-ahead before proceeding.
- **Smoke-test before each check-in.** Every check-in is gated on the pipeline + visualizer still running clean on real data.

---

## Phase 0: Pre-flight decisions (CHECK-IN 0)

Before any code changes, surface three decisions for the user. Do not proceed past this phase without explicit answers.

### Task 0.1: Surface pre-flight decisions

- [ ] **Step 1: Use AskUserQuestion to gather the three decisions**

Ask three questions in a single `AskUserQuestion` call:

1. **Repo name** — recommend `lap-analyzer`. Alternatives: `trackday-telemetry`, `corner-by-corner`.
2. **License** — recommend MIT. Alternatives: Apache-2.0, GPL-3.0.
3. **Data publishing tier** — recommend tier B:
   - **A. Code-only.** Ship no data; `data/samples/` empty; user clones, brings own CSVs.
   - **B. Tiny sample committed + full corpus as release asset.** Ship 1 Ridge session + 1 PIR session (~10MB) in `data/samples/`. Publish full processed corpus (~30MB normalized parquet) as a GitHub Release asset. Raw CSVs stay local.
   - **C. Everything committed.** Raw + processed + corpus all in the repo (~660MB). Not recommended.

- [ ] **Step 2: Record the answers in this plan**

Edit this plan to fill in the three blanks below before Phase 1 starts.

- **Repo name:** `lap-analyzer`
- **License:** MIT
- **Data tier:** B — tiny sample (1 Ridge + 1 PIR session) committed under `data/samples/`, full processed corpus attached as v0.1 GitHub Release asset, raw CSVs stay local

---

## Phase 1: Make the pipeline track-agnostic (in place)

Goal: prove PIR works without moving files yet. If we hit pipeline bugs, they're easier to diagnose with the layout we know.

### Task 1.1: Parametrize labeler latitude

The labeler hardcodes `_M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))` ([labeler.py:36](app/src/lap_analyzer/labeler.py)). At Ridge's latitude this is correct; at PIR (~45.6°) it's ~2% off on E-W drift. Compute from the track's start_finish lat instead.

**Files:**
- Modify: `app/src/lap_analyzer/labeler.py:34-36`

- [ ] **Step 1: Replace the module-level constants with a helper**

```python
# Replace lines 34-36:

def _m_per_deg(lat_deg: float) -> tuple[float, float]:
    """Local meters-per-degree at a given latitude. Used for GPS drift correction.
    Lat is constant; lon shrinks with cos(lat). 1% accuracy is fine for drift math."""
    return 111_132.0, 111_132.0 * float(np.cos(np.radians(lat_deg)))
```

- [ ] **Step 2: Thread it through `compute_lap_drift` and any other consumer**

Grep for uses of `_M_PER_DEG_LAT` and `_M_PER_DEG_LON` in `labeler.py`. Replace each call site to derive `(m_per_deg_lat, m_per_deg_lon)` from the track's `start_finish.lat` once at function entry, then use locally.

The track dict is already in scope where drift is computed; if not, pass it through.

- [ ] **Step 3: Smoke test Ridge unchanged**

Run:
```powershell
cd app
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.label_corners --track ridge --session 20240518-111554
```
Expected: completes without error. Compare `data/sessions/ridge/20240518-111554/corners.parquet` before/after — `gps_drift_disagreement_m` should match within 0.01m (we replaced a Ridge-latitude constant with a Ridge-latitude derivation).

### Task 1.2: Convert `build_centerline.py` notebook to a CLI module

The notebook ([build_centerline.py:33-42](app/notebooks/build_centerline.py)) hardcodes `D:\Projects\lap-analyzer\data\sessions\ridge` and a Ridge-specific `LAP_LENGTH_M`. Convert to a proper CLI under `lap_analyzer.cli.build_centerline` that takes `--track`.

**Files:**
- Create: `app/src/lap_analyzer/cli/build_centerline.py`
- Modify: `app/notebooks/build_centerline.py` (leave a one-line shim pointing at the new module, comment out the old body)

- [ ] **Step 1: Read the existing notebook end-to-end**

Read `app/notebooks/build_centerline.py` fully. Identify: the pure-function "build centerline from sessions dir" core, the hardcoded paths/constants, and the diagnostic plotting section. The plotting section at the bottom (regions-of-interest list line 200+) is Ridge-specific and not needed for PIR — keep it but gate it on `--diagnostics`.

- [ ] **Step 2: Extract the core to `lap_analyzer/labeler.py` or a new module**

Put the build-centerline logic in `lap_analyzer/centerline.py` (new file) so both the new CLI and the existing notebook shim can call it. Signature:

```python
def build_centerline(
    track: str,
    sessions_dir: Path,
    notes_path: Path,
    lap_length_m: float,
    track_lat_deg: float,
    out_path: Path,
) -> pd.DataFrame:
    """Build the synthetic centerline polyline from drift-corrected clean laps.
    Reads every session under sessions_dir, drops sessions with exclude/gps_unreliable
    flags from notes, drift-corrects each lap against the track's apex pins, then
    takes the median (lat,lon) position in 1m windows along the lap. Returns the
    centerline DataFrame and writes it to out_path."""
```

All the hardcoded constants become arguments. Use `tracks/<track>.json` and `data/notes/<track>.json` lookups via the existing `config.py` helpers.

- [ ] **Step 3: Write the CLI wrapper**

```python
# app/src/lap_analyzer/cli/build_centerline.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from ..centerline import build_centerline
from ..config import corpus_dir, data_root, sessions_dir, tracks_dir

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build synthetic centerline polyline from drift-corrected clean laps.")
    p.add_argument("--track", required=True, help="Track slug (e.g. 'ridge')")
    p.add_argument("--out", help="Override output parquet path")
    args = p.parse_args(argv)

    track_def = json.loads((tracks_dir() / f"{args.track}.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else corpus_dir() / f"{args.track}_centerline.parquet"
    notes = data_root() / "notes" / f"{args.track}.json"

    df = build_centerline(
        track=args.track,
        sessions_dir=sessions_dir(args.track),
        notes_path=notes,
        lap_length_m=float(track_def["lap_length_internal_m"]),
        track_lat_deg=float(track_def["start_finish"]["lat"]),
        out_path=out,
    )
    print(f"Wrote {len(df)} centerline points to {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Comment out (do not delete) the body of the old notebook**

Per user convention, keep the file with the body commented:

```python
# app/notebooks/build_centerline.py
"""Legacy notebook — superseded by `python -m lap_analyzer.cli.build_centerline --track ridge`.
Diagnostic plots from the notebook era are preserved below in a comment block; uncomment
to run them against a freshly built centerline."""
# Original notebook contents commented below for reference (2026-05-19).
# from pathlib import Path
# ...  (entire original body)
```

- [ ] **Step 5: Smoke test on Ridge**

```powershell
cd app
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.build_centerline --track ridge --out C:\Users\Saurav\AppData\Local\Temp\ridge_centerline_test.parquet
```

Compare the new file against the existing `data/corpus/ridge_centerline.parquet`:

```powershell
python -c "import pandas as pd; a=pd.read_parquet('C:/Users/Saurav/AppData/Local/Temp/ridge_centerline_test.parquet'); b=pd.read_parquet('../data/corpus/ridge_centerline.parquet'); print(len(a), len(b), (a-b).abs().max() if a.shape==b.shape else 'shape mismatch')"
```

Expected: same length, max abs diff < 1e-6 (we just refactored, no logic change). If different, the refactor introduced a bug — investigate before moving on.

### Task 1.3: Convert `install_centerline.py` notebook to a CLI module

Same treatment as Task 1.2 for [install_centerline.py](app/notebooks/install_centerline.py).

**Files:**
- Create: `app/src/lap_analyzer/cli/install_centerline.py`
- Modify: `app/notebooks/install_centerline.py` (shim + commented body)

- [ ] **Step 1: Read the existing notebook**

Identify the logic: read `data/corpus/<track>_centerline.parquet`, write `data/sessions/<track>/_synthetic_centerline/{samples.parquet, meta.json}` shaped like a real session, print the json patch the user should apply to `tracks/<track>.json`'s `reference_lap`.

- [ ] **Step 2: Write the CLI**

```python
# app/src/lap_analyzer/cli/install_centerline.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from ..config import corpus_dir, sessions_dir, tracks_dir
# ... (port the notebook logic, parametrized on --track)

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Install a built centerline as the labeler's _synthetic_centerline reference session.")
    p.add_argument("--track", required=True)
    args = p.parse_args(argv)
    # body: read corpus_dir()/f"{args.track}_centerline.parquet",
    #       write sessions_dir(args.track)/"_synthetic_centerline"/{samples.parquet, meta.json},
    #       print the JSON patch
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Comment out the old notebook body**

Same convention as Task 1.2 Step 4.

- [ ] **Step 4: Smoke test on Ridge**

```powershell
cd app
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.install_centerline --track ridge
```

Expected: prints a JSON patch matching `tracks/ridge.json`'s current `reference_lap` block. The on-disk `_synthetic_centerline` should be byte-identical (or within float-format tolerance) to before.

### Task 1.4: Parametrize the visualizer's track

The Streamlit app hardcodes `TRACK = "ridge"` in [shared.py:12](app/visualizer/shared.py). Make it env-driven with a sidebar override.

**Files:**
- Modify: `app/visualizer/shared.py:12`
- Modify: `app/visualizer/app.py` (add sidebar track picker)

- [ ] **Step 1: Replace the constant with an env-driven function**

```python
# shared.py: replace `TRACK = "ridge"` with:
import os
def current_track() -> str:
    return os.environ.get("LAP_ANALYZER_TRACK", "ridge")
```

All call sites that used `TRACK` should now call `current_track()`. The `@st.cache_data`-decorated loaders (`corpus`, `laps`, `track_def`, `samples`) need to take `track: str` as their first arg so the cache keys correctly on track switch.

- [ ] **Step 2: Add a sidebar track picker**

In `app.py`, before the existing "Corner" header in the sidebar:

```python
import os
from pathlib import Path
from lap_analyzer.config import tracks_dir

with st.sidebar:
    available_tracks = sorted(p.stem for p in tracks_dir().glob("*.json") if not p.stem.startswith("."))
    default_track = os.environ.get("LAP_ANALYZER_TRACK", "ridge")
    if default_track not in available_tracks:
        default_track = available_tracks[0]
    track = st.selectbox("Track", available_tracks, index=available_tracks.index(default_track), key="picked_track")
    # Push the choice into the env so shared.current_track() sees it on next rerun.
    os.environ["LAP_ANALYZER_TRACK"] = track
```

Note: the track picker needs to invalidate cached loaders on change. The cleanest fix is to make the cached functions take `track` explicitly (rather than reading from env inside), and call them as `_corpus(track)`, `_laps(track)`, etc.

- [ ] **Step 3: Audit `app.py` for hidden track assumptions**

Grep `app/visualizer/` for "ridge", "T1" through "T16", or numeric ranges that assume Ridge's lap geometry. Replace any with track-def-derived values.

- [ ] **Step 4: Smoke test**

```powershell
cd app
$env:PYTHONPATH = "src"; python -m streamlit run visualizer/app.py
```

Verify: app loads, defaults to Ridge, all panels render. (PIR isn't bootstrapped yet — the track picker will only show Ridge.)

### Task 1.5: Phase 1 verification

- [ ] **Step 1: End-to-end Ridge smoke test**

Run the full pipeline on a clean session:
```powershell
cd app
$env:PYTHONPATH = "src"
python -m lap_analyzer.cli.normalize --track ridge --all
python -m lap_analyzer.cli.label_corners --track ridge
python -m lap_analyzer.cli.build_centerline --track ridge
python -m lap_analyzer.cli.install_centerline --track ridge
python -m lap_analyzer.cli.label_corners --track ridge
python -m lap_analyzer.cli.flag_quality --track ridge
python -m lap_analyzer.cli.build_corpus --track ridge
```

Expected: each step completes, final `corners.parquet` byte-identical (or within float tolerance) to its pre-refactor state for at least 3 spot-checked sessions.

- [ ] **Step 2: Visualizer smoke test**

Launch the visualizer, verify the same lap renders identically to before (Ridge T11 downshift page especially — it does the most cross-channel math).

---

## CHECK-IN 1 — Phase 1 complete, ready for PIR

Report to user:
- Which files moved/changed.
- Ridge byte-equality results.
- Anything weird discovered along the way.
- Ask: ready to bootstrap PIR?

---

## Phase 2: Bootstrap PIR

This phase has interleaved user-manual and agent-automatic steps. The agent cannot pin corners on Google Maps; the user does that. The plan stops at each manual step and waits.

### Task 2.1: PIR skeleton

**Files:**
- Create: `app/tracks/pir.json` (skeleton — minimal, no corners yet)

- [ ] **Step 1: Look up PIR official metadata**

Portland International Raceway. Length ~1.967mi (3,166m). Direction: clockwise. Need start/finish lat/long — extract from any PIR CSV header (`# End Point: lat, long`).

```powershell
Get-Content (Get-ChildItem ../data/raw/pir/*.csv | Select-Object -First 1).FullName -TotalCount 5
```

- [ ] **Step 2: Write the minimal `tracks/pir.json`**

```json
{
  "track_id": "pir",
  "name": "Portland International Raceway",
  "configuration": "full",
  "official_length_m": 3166,
  "direction": "clockwise",
  "start_finish": { "lat": <from CSV header>, "long": <from CSV header> },
  "lap_length_internal_m": <best guess, will refine after first normalize pass — start with official_length_m>,
  "reference_lap": null,
  "calibration_anchors": [],
  "corners": []
}
```

### Task 2.2: First normalize pass on PIR

- [ ] **Step 1: Run normalize**

```powershell
cd app
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.normalize --track pir --all
```

Expected: 18 PIR sessions processed. Some may report `noobd` (per memory: ~3-4% have no OBD).

- [ ] **Step 2: Check `lap_length_internal_m`**

⚠️ **Plan was wrong (corrected 2026-05-20):** `lap_dist_m` in `laps.csv` does NOT reflect OBD-integrated lap length — it's just a copy of the configured `lap_length_internal_m`. Don't be misled by the column name.

The real signal lives in `samples.parquet`'s raw `dist_m` (cumulative GPS-integrated distance through the session). Compute per-lap distance as `df.groupby('lap')['dist_m'].agg(lambda s: s.max() - s.min())`, filter to flying laps (e.g. `2000 < d < 4000` for PIR's ~3.2km length), take the median, update `tracks/pir.json`'s `lap_length_internal_m`, then **re-normalize with `--force`** so per-lap `dist_lap_m` rescales correctly.

Resolved value for PIR: 3198m (vs official 3166m — ~1% longer, normal for race-line + GPS integration noise). For comparison, Ridge raw median is 3936m vs configured 3962m (config'd ~26m above median, historical — left as-is since rescaling is internally consistent either way).

### Task 2.3: PIR Maps-pin curation (USER MANUAL STEP)

- [ ] **Step 1: STOP. Report to user.**

Tell user: "PIR has been normalized. Next step is creating the Maps-pin apex file. PIR has roughly 12 numbered corners (T1-T12). For each, pin the visual inside-kerb apex on Google Maps satellite view at [maps.google.com](https://maps.google.com), and record the lat/long. Save them as `data/notes/pir_apex_pins.json` mirroring the schema of `data/notes/ridge_apex_pins.json`. This is the only manual curation step — it takes ~30-60 minutes."

Provide the existing `ridge_apex_pins.json` as a schema reference:

```powershell
Get-Content ../data/notes/ridge_apex_pins.json -TotalCount 20
```

Wait for user to complete this step before resuming.

- [ ] **Step 2: Verify the file when user confirms done**

```powershell
Test-Path ../data/notes/pir_apex_pins.json
Get-Content ../data/notes/pir_apex_pins.json | ConvertFrom-Json | Select-Object -ExpandProperty pins | Measure-Object
```

Expected: ~12 entries.

### Task 2.4: Seed PIR corner ranges via candidate detection

- [ ] **Step 1: Run candidate scan**

```powershell
cd app
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.extract_corner_candidates --track pir
```

Output: `data/corpus/pir_candidates.csv` — lat-G peaks across all flying laps.

- [ ] **Step 2: Cluster candidates into corners**

This step has been historically a user-judgment task ([app/docs/curation-ridge.md](app/docs/curation-ridge.md) describes the original methodology). With Maps pins available, we can shortcut: for each pin, find the nearest cluster of candidate peaks in `peak_dist_m`, take its median as the corner's `apex_m`, and define `start_m`/`end_m` as `apex_m ± 80m` initially (the labeler will refine via `compute_corner_bounds_from_data` later).

Write a one-off script `app/notebooks/seed_pir_corners.py` that does this and writes the populated `corners` block to `tracks/pir.json`. Reference the existing `app/notebooks/cluster_ridge.py` for the clustering pattern.

- [ ] **Step 3: Populate `calibration_anchors` from the pins**

Each pin becomes a calibration anchor entry in `tracks/pir.json`. Copy the schema from `ridge.json`'s `calibration_anchors` block.

### Task 2.5: First-pass PIR labeling

- [ ] **Step 1: Run labeler**

```powershell
cd app
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.label_corners --track pir
```

This runs without a centerline yet — labeler should detect missing `_synthetic_centerline` and fall back to a bootstrap mode (or, if it doesn't, modify the labeler to allow a `--bootstrap` flag that uses the apex pins directly as a coarse ruler for first-pass `track_dist_m` assignment). Verify this path exists; if not, add it.

### Task 2.6: Build + install PIR centerline

- [ ] **Step 1: Build**

```powershell
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.build_centerline --track pir
```

- [ ] **Step 2: Install**

```powershell
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.install_centerline --track pir
```

- [ ] **Step 3: Apply the printed `reference_lap` patch to `tracks/pir.json`**

### Task 2.7: Second-pass PIR labeling + quality flags

- [ ] **Step 1: Re-label with centerline**

```powershell
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.label_corners --track pir
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.flag_quality --track pir
$env:PYTHONPATH = "src"; python -m lap_analyzer.cli.build_corpus --track pir
```

- [ ] **Step 2: Sanity-check the PIR corpus**

```powershell
python -c "import pandas as pd; df = pd.read_parquet('../data/corpus/pir_corners.parquet'); print(df.shape, df['transit_reliable'].mean(), df.groupby('corner_id').size())"
```

Expected: every corner has roughly the same number of transits (modulo dropped laps), and `transit_reliable` mean is in the 0.85-0.95 ballpark.

### Task 2.8: PIR visualizer smoke test

- [ ] **Step 1: Launch with PIR**

```powershell
cd app
$env:LAP_ANALYZER_TRACK = "pir"; $env:PYTHONPATH = "src"; python -m streamlit run visualizer/app.py
```

Or use the sidebar track switcher. Verify the visualizer renders PIR laps end-to-end without errors. Especially: the time-delta panel, the per-corner section-time table, and the corner shading on the channel plot.

- [ ] **Step 2: Document any PIR-specific bugs in a punch list**

If the visualizer breaks anywhere on PIR, file each issue in this plan as a new task under Task 2.9. Single-track assumptions discovered here are exactly the reason for doing PIR before publishing.

### Task 2.9: PIR fix punch list (filled at runtime)

- [ ] *(reserved — populate during Task 2.8)*

---

## CHECK-IN 2 — PIR live, decide on ORP

Report to user:
- PIR transit-reliable rate.
- Any single-track assumptions found and fixed.
- Side-by-side visualizer screenshots of one Ridge lap + one PIR lap.
- Ask:
  - Bootstrap ORP now or defer? (6 ORP CSVs, ~2 sessions each direction → small corpus, may not be worth the manual pinning time for v1.)
  - Any PIR data quality concerns before proceeding to restructure?

---

## Phase 3: Repo restructure

The `app/` wrapper buys nothing; flatten it. The package is the project.

### Task 3.1: Move the Python package

**Files:**
- Move: `app/src/lap_analyzer/` → `lap_analyzer/`

- [ ] **Step 1: Move the directory**

```powershell
Move-Item app/src/lap_analyzer lap_analyzer
```

(After move, `app/src/` is empty — leave it for now, remove in Task 3.4.)

- [ ] **Step 2: Update `lap_analyzer/config.py`**

```python
# Before: APP_ROOT = Path(__file__).resolve().parents[2]   # app/src/lap_analyzer -> app/
# After:
APP_ROOT = Path(__file__).resolve().parents[1]              # lap_analyzer/ -> repo root
```

Also update the `DATA_ROOT` default from `"../data"` to `"data"`:

```python
def data_root() -> Path:
    return (APP_ROOT / os.environ.get("DATA_ROOT", "data")).resolve()
```

### Task 3.2: Move the visualizer

- [ ] **Step 1: Move**

```powershell
Move-Item app/visualizer visualizer
```

- [ ] **Step 2: Update the `cd` paths in docstrings**

`visualizer/app.py` docstring and the README both say `cd app`. After restructure, you stay at repo root. Search-and-replace:

```powershell
# from repo root after move:
# grep for any "cd app" reference and update to "" (no cd needed)
```

### Task 3.3: Move tracks/ and docs/

- [ ] **Step 1: Move tracks**

```powershell
Move-Item app/tracks tracks
```

- [ ] **Step 2: Move app/docs into top-level docs/**

```powershell
Move-Item app/docs/* docs/
Remove-Item app/docs
```

(`docs/` already exists at root with `superpowers/` inside. Merging keeps that.)

- [ ] **Step 3: Verify `tracks_dir()` in `config.py`**

`tracks_dir()` should now resolve to `<repo_root>/tracks` automatically via the updated `APP_ROOT`. Spot-check:

```powershell
python -c "import sys; sys.path.insert(0, '.'); from lap_analyzer.config import tracks_dir; print(tracks_dir())"
```

Expected: prints `D:\Projects\lap-analyzer\tracks`.

### Task 3.4: Move pyproject.toml + update

- [ ] **Step 1: Move and edit**

```powershell
Move-Item app/pyproject.toml pyproject.toml
```

Edit:
```toml
[tool.hatch.build.targets.wheel]
packages = ["lap_analyzer"]   # was: ["src/lap_analyzer"]

[tool.pytest.ini_options]
testpaths = ["tests"]   # unchanged; will become meaningful in the TDD session
```

- [ ] **Step 2: Delete now-empty `app/` subdirectories**

```powershell
Remove-Item -Recurse app/src
Remove-Item app   # only if everything else moved out cleanly
```

If `app/` still contains `notebooks/` or `tests/`, see Task 3.5/3.6 first.

### Task 3.5: Cull notebooks

The notebooks dir has ~40 scripts. Most are one-off diagnostics with committed PNG outputs. Triage into three buckets:

**KEEP** (move to `scripts/` — they're part of the bootstrap or actively useful):
- `fetch_osm_ridge.py` (generalize → `scripts/fetch_osm.py` later)
- `seed_pir_corners.py` (from Task 2.4)
- Anything the user identifies as "I run this regularly"

**ARCHIVE** (move to `scratch/` which becomes gitignored — preserves history without shipping):
- All `diag_*.py`, `verify_*.py`, `t11_*.py`, `t13_lines.py`, etc.
- All `*.png` files
- All `*.csv` produced by these notebooks (`ridge_clusters.csv`, etc.)
- The `proposed_corner_bounds.json` (one-off curation artifact)

**DELETE** (per user "comment out don't delete" rule, instead move to scratch):
- `__pycache__/`

- [ ] **Step 1: Create the new dirs**

```powershell
New-Item -ItemType Directory -Path scripts, scratch
```

- [ ] **Step 2: Move keepers to `scripts/`**

```powershell
Move-Item app/notebooks/fetch_osm_ridge.py scripts/
# ... (one Move-Item per keeper)
```

- [ ] **Step 3: Move everything else to `scratch/`**

```powershell
Move-Item app/notebooks/* scratch/
Remove-Item app/notebooks
```

- [ ] **Step 4: Stop here and surface the triage list to user before deciding**

Actually: stop before Step 2 and ask the user which notebooks they want kept in `scripts/` vs archived to `scratch/`. The agent can guess from filenames but the user has tacit knowledge.

### Task 3.6: Remove backup/junk files

- [ ] **Step 1: List candidates for removal**

```powershell
Get-ChildItem -Recurse -Include *.bak, *.pre-*.json | Select-Object FullName
```

Expected: `tracks/ridge.json.bak`, `tracks/ridge.pre-apex-pin-update.json`, `tracks/ridge.pre-notes-refresh.json`.

- [ ] **Step 2: Move them to `scratch/`**

Per user's "comment out don't delete" preference, move to `scratch/` (which will be gitignored) rather than deleting outright:

```powershell
Move-Item tracks/*.bak scratch/
Move-Item tracks/ridge.pre-*.json scratch/
```

### Task 3.7: Smoke test from new layout

- [ ] **Step 1: Full pipeline from repo root**

```powershell
# from D:\Projects\lap-analyzer (no cd app):
$env:PYTHONPATH = "."
python -m lap_analyzer.cli.normalize --track ridge --all
python -m lap_analyzer.cli.label_corners --track ridge
python -m lap_analyzer.cli.flag_quality --track ridge
python -m lap_analyzer.cli.build_corpus --track ridge
```

Expected: same outputs as before, no path errors.

- [ ] **Step 2: Visualizer from repo root**

```powershell
$env:PYTHONPATH = "."; python -m streamlit run visualizer/app.py
```

Expected: launches, Ridge + PIR both selectable in sidebar.

---

## CHECK-IN 3 — Restructure done

Report to user:
- New layout (paste `tree -L 2` output, agent equivalent: recursive `Get-ChildItem`).
- Whether sample pipeline still produces byte-equal output.
- Notebook triage list (keep vs scratch).
- Ask: any layout objections? Want anything else moved?

---

## Phase 4: Sample data + gitignore

Per pre-flight decision in Phase 0, ship the chosen data tier.

### Task 4.1: Choose sample sessions

- [ ] **Step 1: Pick one Ridge session + one PIR session**

Criteria for the sample sessions:
- Clean (mostly `is_clean=True` laps).
- Reasonably good driving (top-third pace) so the visualizer shows a realistic-looking lap.
- All four pipeline outputs present: `samples.parquet`, `corners.parquet`, `laps.csv`, `meta.json`.

Suggest one Ridge session (e.g. the fastest May-1 session) and one PIR session (e.g. a fastest 1:33 session). Ask user to confirm before copying.

- [ ] **Step 2: Copy them under `data/samples/`**

```powershell
New-Item -ItemType Directory -Path data/samples/ridge, data/samples/pir
Copy-Item -Recurse data/sessions/ridge/<chosen-sid> data/samples/ridge/
Copy-Item -Recurse data/sessions/pir/<chosen-sid> data/samples/pir/
```

- [ ] **Step 3: Also copy the centerlines**

The visualizer needs `data/corpus/<track>_centerline.parquet` and the synthetic-centerline session. Decide whether sample mode points at a sample corpus or the real one. Cleanest: include the centerline (it's ~200KB per track) under `data/samples/_corpus/`.

- [ ] **Step 4: Add a "sample mode" doc snippet**

Quickstart instructions for a fresh clone:

```bash
git clone <repo>
cd lap-analyzer
pip install -e ".[viz]"
$env:DATA_ROOT = "data/samples"; python -m streamlit run visualizer/app.py
```

### Task 4.2: Write `.gitignore`

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Write it**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# IDE
.vscode/
.idea/

# Project data — too big and personal
data/raw/
data/sessions/
data/corpus/
data/notes/

# But ship the samples
!data/samples/

# Scratch (legacy notebooks, backups)
scratch/

# Streamlit
.streamlit/

# Env
.env

# OS
.DS_Store
Thumbs.db
```

### Task 4.3: Verify sample-only mode works

- [ ] **Step 1: Simulate a fresh clone**

```powershell
# Move the real data out of the way temporarily:
Move-Item data data.real
New-Item -ItemType Directory data
Copy-Item -Recurse data.real/samples data/samples
```

- [ ] **Step 2: Launch visualizer in sample mode**

```powershell
$env:DATA_ROOT = "data/samples"; $env:PYTHONPATH = "."; python -m streamlit run visualizer/app.py
```

Expected: launches, shows the one Ridge sample session, switches to PIR sample session.

- [ ] **Step 3: Restore the real data**

```powershell
Remove-Item -Recurse data
Move-Item data.real data
```

---

## CHECK-IN 4 — Sample mode works

Report to user:
- Sample session sizes (MB committed).
- Screenshot of visualizer running in sample mode.
- Ask: ready to write docs?

---

## Phase 5: Documentation

Split the current README's two concerns (architecture rationale + user runbook) into separate docs.

### Task 5.1: Write LICENSE

**Files:**
- Create: `LICENSE`

- [ ] **Step 1: Drop in the chosen license text**

Use the SPDX-standard text for the license picked in Phase 0. For MIT:

```
MIT License

Copyright (c) 2026 <user's name from git config or ask>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Task 5.2: New top-level README

**Files:**
- Modify: `README.md` (complete rewrite — preserve nothing; the old content is preserved in `docs/ARCHITECTURE.md` by Task 5.3)

- [ ] **Step 1: Write the new README**

Sections, in order:
1. **One-paragraph elevator pitch.** "Lap Analyzer takes TrackAddict CSV exports and produces per-corner transit metrics across all your sessions — so you can ask 'what did I do differently in T11 today vs my fastest?' across hundreds of laps. Built for personal track-day data analysis."
2. **Screenshot** of the visualizer in sample mode. Place file at `docs/screenshot.png`.
3. **Quickstart** — `git clone`, `pip install`, `DATA_ROOT=data/samples streamlit run`. Show the actual one-liner.
4. **Supported tracks** — Ridge Motorsports Park, Portland International Raceway. Note that adding a new track is documented in `docs/NEW-TRACK.md`.
5. **Inputs** — TrackAddict CSV. Link to `docs/PIPELINE.md` for the column contract.
6. **Project layout** — short tree, link to `docs/ARCHITECTURE.md` for design rationale.
7. **Contributing** — link to a one-line CONTRIBUTING.md or inline if minimal.
8. **License** — line pointing at LICENSE.

Length target: ~80-120 lines.

### Task 5.3: docs/ARCHITECTURE.md

**Files:**
- Create: `docs/ARCHITECTURE.md`

- [ ] **Step 1: Move the design-decision content from the old README**

Sections to move verbatim (preserve all five "Core design decisions" subsections):
- "Architectural overview" diagram
- All of "Core design decisions" (1-5)
- "Quality flags" detail (the table + thresholds)

Drop from the move:
- "Pipeline at a glance" (moves to PIPELINE.md)
- "File layout" (moves to README's "Project layout" section)
- "What's not yet built" (stale — both items are now built)
- "How to add a new track" (moves to NEW-TRACK.md)

Add at the top: "This document captures the design rationale behind each architectural choice. For a runbook on how to use the pipeline, see [PIPELINE.md](PIPELINE.md). For how to bootstrap a new track, see [NEW-TRACK.md](NEW-TRACK.md)."

### Task 5.4: docs/PIPELINE.md

**Files:**
- Create: `docs/PIPELINE.md`

- [ ] **Step 1: Write the runbook**

Sections:
1. **Pipeline overview** — the 5 CLI scripts and what each produces.
2. **CLI reference** — each script with full `--help` text and example invocations.
3. **TrackAddict CSV column contract** — the 28 columns (use the existing list from the current README's spec or extract from a real CSV's header). Document required vs optional columns. Note that missing OBD columns trigger a `noobd` skip; missing GPS columns are a hard fail.
4. **Output schema** — `samples.parquet`, `corners.parquet`, `laps.csv`, `meta.json` — list each column with one-line description.
5. **Common operations** — re-running a single session, force re-normalization, adding session-level notes (`data/notes/<track>.json` exclude/flag schema).

### Task 5.5: docs/NEW-TRACK.md

**Files:**
- Create: `docs/NEW-TRACK.md`

- [ ] **Step 1: Write the bootstrap walkthrough**

This is essentially the current README's "How to add a new track" section, expanded with the lessons learned from bootstrapping PIR in Phase 2. Sections:

1. **Inputs you need** — raw CSVs, track length, direction, start/finish coords.
2. **Step-by-step bootstrap** — every command in order, with expected output for each step.
3. **The Maps-pin step in detail** — how to choose visual apexes, link to Google Maps satellite, schema for the pins JSON.
4. **Validation checklist** — what to look for in `corners.parquet` to know your pins are good (transit_reliable rate, apex_dist_offset_m landing in expected ranges per corner type).
5. **Troubleshooting** — common failure modes (lap_length_internal_m off → start_finish in wrong place; pins too far from actual kerbs → drift correction goes haywire).

### Task 5.6: CONTRIBUTING.md (optional, short)

**Files:**
- Create: `CONTRIBUTING.md`

- [ ] **Step 1: Write it**

```markdown
# Contributing

This is a personal project published in the hope it's useful. PRs welcome but not actively solicited.

If you're trying to add support for your own track, follow [docs/NEW-TRACK.md](docs/NEW-TRACK.md). If you hit a track-bootstrap snag, open an issue with: raw CSV header, the pin JSON you produced, and the error.

Tests are currently sparse — that's known and being worked on.
```

---

## CHECK-IN 5 — Docs done, final review

Report to user:
- All docs written, link to each.
- Total committed size (`du -sh` equivalent).
- Ask: read through docs and verify accuracy. Anything missing?

---

## Phase 6: Pre-publish polish

### Task 6.1: Final repo audit

- [ ] **Step 1: Hunt for absolute Windows paths**

```powershell
# from repo root:
# (using Grep tool, not bash, per assistant conventions)
```

Use the Grep tool to search for `D:\\Projects` or `C:\\Users` across `*.py`, `*.md`, `*.json`. Any hit is a paste-artifact bug — fix to relative paths.

- [ ] **Step 2: Hunt for `__pycache__` and `.pyc` files**

```powershell
Get-ChildItem -Recurse -Include __pycache__, *.pyc -Force | Remove-Item -Recurse -Force
```

- [ ] **Step 3: Verify no secrets in any committed file**

There's no `.env` in the repo today, but double-check:
```powershell
Get-ChildItem -Recurse -Force -Include .env, .env.*, *secret*, *credential*
```

### Task 6.2: Take the README screenshot

- [ ] **Step 1: Launch visualizer in sample mode**

```powershell
$env:DATA_ROOT = "data/samples"; $env:PYTHONPATH = "."; python -m streamlit run visualizer/app.py
```

- [ ] **Step 2: Take a screenshot of a good lap view**

User-driven. Save as `docs/screenshot.png`. Suggest a Ridge T8-T11 range view since it shows the most of the visualizer's distinctive features (Δt panel, corner shading, envelope).

### Task 6.3: Dry-run a fresh-clone install

- [ ] **Step 1: Create a throwaway sibling directory**

```powershell
New-Item -ItemType Directory C:\Users\Saurav\AppData\Local\Temp\lap-analyzer-fresh
Copy-Item -Recurse -Exclude data.real, scratch, __pycache__, *.pyc D:\Projects\lap-analyzer\* C:\Users\Saurav\AppData\Local\Temp\lap-analyzer-fresh\
```

(Only copies what would be in git — approximates `git clone`.)

- [ ] **Step 2: Install from scratch in a clean venv**

```powershell
cd C:\Users\Saurav\AppData\Local\Temp\lap-analyzer-fresh
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

- [ ] **Step 3: Launch the visualizer following the README quickstart verbatim**

If anything doesn't work or the README is wrong, fix the README. The quickstart MUST work cold.

---

## CHECK-IN 6 — Ready to push

Report to user:
- Final repo size.
- Fresh-clone test passed.
- Screenshot in place.
- Ask: ready for `git init` + push?

---

## Phase 7: Initialize git and push (user-driven)

### Task 7.1: Initialize git

- [ ] **Step 1: User runs**

```powershell
cd D:\Projects\lap-analyzer
git init
git add .
git status   # verify .gitignore is working (no data/raw, no scratch)
git commit -m "Initial commit"
```

Agent's role: verify the staged file list looks right before the user commits. Flag anything surprising.

### Task 7.2: Create the GitHub repo and push

- [ ] **Step 1: User creates the repo**

Either via `gh repo create <name> --public --source=. --remote=origin --push` or via the GitHub UI + `git remote add`. The user owns this step — agent's role is to confirm the description, topics, and visibility before the user runs it.

### Task 7.3: (Optional) Tag a v0.1 release with corpus assets

If the user chose Data Tier B in Phase 0, attach the full processed corpus as a release asset:

```powershell
# bundle the corpus:
Compress-Archive -Path data/corpus/* -DestinationPath ridge-pir-corpus-v0.1.zip
gh release create v0.1 --title "v0.1 — initial public release" --notes "..." ridge-pir-corpus-v0.1.zip
```

---

## Post-publish: Tests session (separate plan)

Per user request, tests are a separate TDD-flavored session:

1. **Session A** (this agent, fresh context): review the published code → write a test spec to `docs/superpowers/specs/test-coverage-spec.md`. Spec lists invariants and behaviors per module without prescribing test code.
2. **Session B** (fresh agent, sees only the spec): write tests under `tests/` matching the spec, no peek at implementation.
3. **Session C** (this agent or user): run the tests, debug failures, decide what's a real bug vs a spec ambiguity.

That session has its own plan and is not part of this one.

---

## Self-review notes

Coverage check against user's stated wants:
- ✅ Separate visualizer from scripts conceptually — Phase 3 puts them in sibling top-level dirs, both consuming `lap_analyzer/` library.
- ✅ Detailed run instructions — `docs/PIPELINE.md` (5.4) + README quickstart (5.2).
- ✅ Expected source columns — `docs/PIPELINE.md` section 3 covers the TrackAddict contract.
- ✅ Personal data decision — Phase 0 surfaces it, Phase 4 implements whichever tier user picks.
- ✅ Sample-data starter — Task 4.1-4.3.
- ✅ Multi-track validation (PIR) — Phase 2 fully covered.
- ✅ Tests deferred to a separate TDD session — noted at top, scaffolded at bottom.
- ✅ Check-ins at key milestones — 6 explicit checkpoints.

Things this plan deliberately does NOT do:
- Refactor analysis logic for cleanliness.
- Add new visualizer features.
- Add CI/CD.
- Add type-checking config.
- Add a CHANGELOG (premature for v0).

These are good follow-ups for separate sessions.
