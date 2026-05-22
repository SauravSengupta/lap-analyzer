# T8–T11 Downshift Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a visualizer page that compares T8→T11 laps split by whether a downshift was made entering T8.

**Architecture:** Three layers, no data-pipeline changes. (1) Pure functions in `analysis.py` derive a per-sample gear from the `rpm/speed_mph` ratio and classify each lap's T8 entry. (2) A new `shared.py` holds visualizer loaders extracted from `app.py`. (3) A new Streamlit page in `pages/` renders grouped channel envelopes, a time-delta curve, and a section-time comparison.

**Tech Stack:** Python, pandas, numpy, Streamlit, Plotly. Verification via a script in `app/notebooks/` (the project has no pytest harness).

**Note on commits:** This repo is not git-initialized. Either run `git init` in `D:\Projects\lap-analyzer` first, or skip the `git commit` steps. All other steps stand alone.

**Spec:** `docs/superpowers/specs/2026-05-17-t8-t11-downshift-comparison-design.md`

---

## File Structure

- **Modify** `app/src/lap_analyzer/analysis.py` — add `find_gear_bands`, `gear_bands`, `_confirm_runs`, `derive_gear`, `span_time`, `classify_t8_section`.
- **Create** `app/notebooks/verify_gear_classification.py` — synthetic-input assertion checks + corpus eyeball.
- **Create** `app/visualizer/shared.py` — cached loaders/helpers extracted from `app.py`.
- **Modify** `app/visualizer/app.py` — import shared loaders instead of defining them inline.
- **Create** `app/visualizer/pages/1_Downshift_T8-T11.py` — the comparison page.

Run commands assume working directory `D:\Projects\lap-analyzer` and the project venv at `app\.venv`.

---

### Task 1: `find_gear_bands` + `gear_bands` — derive gear band centers

**Files:**
- Modify: `app/src/lap_analyzer/analysis.py`
- Test: `app/notebooks/verify_gear_classification.py` (create)

- [ ] **Step 1: Write the failing test**

Create `app/notebooks/verify_gear_classification.py`:

```python
"""Verification for gear derivation + T8 downshift classification.

Run: app\\.venv\\Scripts\\python.exe app/notebooks/verify_gear_classification.py
The project has no pytest harness; this script is the test harness. Each
check_* function asserts on synthetic input; main() also prints a corpus
eyeball at the end (added in Task 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def check_find_gear_bands() -> None:
    from lap_analyzer.analysis import find_gear_bands

    rng = np.random.default_rng(0)
    ratios = np.concatenate([
        rng.normal(54.0, 1.0, 3000),
        rng.normal(99.0, 1.2, 3000),
        rng.normal(180.0, 2.0, 3000),
    ])
    bands = find_gear_bands(ratios)
    assert len(bands) == 3, f"expected 3 bands, got {bands}"
    assert abs(bands[0] - 54.0) < 2.0, bands
    assert abs(bands[1] - 99.0) < 2.0, bands
    assert abs(bands[2] - 180.0) < 3.0, bands
    print("check_find_gear_bands OK", bands)


def main() -> None:
    check_find_gear_bands()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py`
Expected: FAIL — `ImportError: cannot import name 'find_gear_bands'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/src/lap_analyzer/analysis.py`:

```python
# --- gear derivation --------------------------------------------------------

def find_gear_bands(
    ratios: np.ndarray,
    ratio_lo: float = 30.0,
    ratio_hi: float = 260.0,
    n_bins: int = 230,
    min_peak_frac: float = 0.005,
    min_separation: float = 8.0,
) -> np.ndarray:
    """Cluster pooled rpm/speed ratios into gear band centers.

    Histograms the ratios, takes strict local maxima above `min_peak_frac` of
    the tallest bin, merges maxima closer than `min_separation` ratio units
    (keeping the first), and refines each surviving center as the mean ratio of
    samples within +/- min_separation/2. Returns sorted band centers (ascending
    ratio; smallest ratio = tallest gear).
    """
    r = ratios[(ratios >= ratio_lo) & (ratios <= ratio_hi)]
    if r.size == 0:
        return np.array([])
    hist, edges = np.histogram(r, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    thresh = min_peak_frac * hist.max()
    peak = np.zeros(n_bins, dtype=bool)
    peak[1:-1] = (
        (hist[1:-1] >= hist[:-2])
        & (hist[1:-1] > hist[2:])
        & (hist[1:-1] >= thresh)
    )
    raw = centers[peak]
    merged: list[float] = []
    for c in raw:
        if merged and (c - merged[-1]) < min_separation:
            continue
        merged.append(float(c))
    refined: list[float] = []
    for c in merged:
        near = r[np.abs(r - c) < min_separation / 2]
        refined.append(float(near.mean()) if near.size else c)
    return np.array(sorted(refined))


def gear_bands(track: str) -> np.ndarray:
    """Corpus-wide gear band centers for a track (IO wrapper over find_gear_bands)."""
    root = sessions_dir(track)
    pooled: list[np.ndarray] = []
    for sp in sorted(root.rglob("samples.parquet")):
        try:
            df = pd.read_parquet(sp, columns=["rpm", "speed_mph"])
        except Exception:
            continue
        m = (df["rpm"] > 1200) & (df["speed_mph"] > 8)
        if m.any():
            pooled.append((df["rpm"][m] / df["speed_mph"][m]).to_numpy())
    if not pooled:
        return np.array([])
    return find_gear_bands(np.concatenate(pooled))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py`
Expected: PASS — prints `check_find_gear_bands OK [ ~54  ~99  ~180 ]`.

- [ ] **Step 5: Commit**

```bash
git add app/src/lap_analyzer/analysis.py app/notebooks/verify_gear_classification.py
git commit -m "feat: derive gear bands from rpm/speed ratio"
```

---

### Task 2: `derive_gear` — per-sample gear with dwell filter

**Files:**
- Modify: `app/src/lap_analyzer/analysis.py`
- Test: `app/notebooks/verify_gear_classification.py`

- [ ] **Step 1: Write the failing test**

Add to `verify_gear_classification.py` (before `main`):

```python
def _synthetic_samples(ratios: list[float], speed: float = 80.0,
                       dt: float = 0.046) -> pd.DataFrame:
    """Samples df with constant speed; rpm chosen to hit each target ratio.

    ratios entries may be None to mark an OBD dropout (rpm = 0).
    """
    n = len(ratios)
    rpm = np.array([0.0 if r is None else r * speed for r in ratios])
    return pd.DataFrame({
        "t": np.arange(n) * dt,
        "rpm": rpm,
        "speed_mph": np.full(n, speed),
        "track_dist_m": np.arange(n) * 2.0,
        "dist_lap_m": np.arange(n) * 2.0,
    })


def check_derive_gear() -> None:
    from lap_analyzer.analysis import derive_gear

    bands = np.array([54.0, 99.0, 180.0])  # gear idx: 54->2, 99->1, 180->0

    # A 3-sample blip at ratio 99 inside a long ratio-54 stint is below dwell
    # and must be smoothed away -> gear stays 2 (tallest) throughout.
    blip = [54.0] * 100 + [99.0] * 3 + [54.0] * 97
    g = derive_gear(_synthetic_samples(blip), bands)
    confirmed = g.dropna()
    assert (confirmed == 2).all(), f"blip not smoothed: {sorted(confirmed.unique())}"

    # A real, sustained downshift 54 -> 180 must register: gear 2 then 0.
    shift = [54.0] * 100 + [180.0] * 100
    g2 = derive_gear(_synthetic_samples(shift), bands).to_numpy()
    assert g2[20] == 2, g2[20]
    assert g2[-1] == 0, g2[-1]

    # OBD dropout (rpm == 0) -> NaN gear.
    drop = [54.0] * 50 + [None] * 50 + [54.0] * 50
    g3 = derive_gear(_synthetic_samples(drop), bands).to_numpy()
    assert np.isnan(g3[75]), g3[75]
    print("check_derive_gear OK")
```

Update `main`:

```python
def main() -> None:
    check_find_gear_bands()
    check_derive_gear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py`
Expected: FAIL — `ImportError: cannot import name 'derive_gear'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/src/lap_analyzer/analysis.py`:

```python
def _confirm_runs(raw: np.ndarray, n_dwell: int) -> np.ndarray:
    """Set values in any run of identical values shorter than n_dwell to NaN.

    A run is a maximal stretch of equal (or all-NaN) values. Used to drop the
    brief wrong-gear snap while the clutch is slipping during a shift.
    """
    out = raw.astype(float).copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and (
            (out[j] == out[i]) or (np.isnan(out[j]) and np.isnan(out[i]))
        ):
            j += 1
        if not np.isnan(out[i]) and (j - i) < n_dwell:
            out[i:j] = np.nan
        i = j
    return out


def derive_gear(
    samples: pd.DataFrame, bands: np.ndarray, min_dwell_s: float = 0.3
) -> pd.Series:
    """Per-sample gear index derived from the rpm/speed ratio.

    Snaps each qualifying sample (rpm > 1200, speed_mph > 8) to the nearest
    band, drops sub-dwell runs (clutch-slip transients) and forward-fills the
    held gear across them, then re-NaNs samples with no OBD (rpm == 0). Gear
    index 0..len(bands)-1; index 0 = lowest gear, higher index = taller gear.
    Returned Series is aligned to `samples.index`.
    """
    s = samples.sort_values("t")
    rpm = s["rpm"].to_numpy(dtype=float)
    spd = s["speed_mph"].to_numpy(dtype=float)
    valid = (rpm > 1200) & (spd > 8)
    raw = np.full(len(s), np.nan)
    if bands.size and valid.any():
        rv = rpm[valid] / spd[valid]
        nearest = np.argmin(np.abs(rv[:, None] - bands[None, :]), axis=1)
        raw[valid] = (bands.size - 1) - nearest
    t = s["t"].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.046
    n_dwell = max(1, int(round(min_dwell_s / dt)))
    confirmed = _confirm_runs(raw, n_dwell)
    gear = pd.Series(confirmed, index=s.index).ffill()
    gear[rpm == 0] = np.nan
    return gear.reindex(samples.index)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py`
Expected: PASS — prints `check_derive_gear OK`.

- [ ] **Step 5: Commit**

```bash
git add app/src/lap_analyzer/analysis.py app/notebooks/verify_gear_classification.py
git commit -m "feat: per-sample gear derivation with dwell filter"
```

---

### Task 3: `span_time` — single-lap section time between two distances

**Files:**
- Modify: `app/src/lap_analyzer/analysis.py`
- Test: `app/notebooks/verify_gear_classification.py`

- [ ] **Step 1: Write the failing test**

Add to `verify_gear_classification.py` (before `main`):

```python
def check_span_time() -> None:
    from lap_analyzer.analysis import span_time

    # track_dist advances 40 m/s; dist_lap_m mirrors it (OBD agrees with GPS).
    dist = np.arange(0.0, 3000.0, 2.0)
    s = pd.DataFrame({"t": dist / 40.0, "track_dist_m": dist, "dist_lap_m": dist})
    got = span_time(s, 1830.5, 2574.5)
    assert got is not None and abs(got - (744.0 / 40.0)) < 0.05, got

    # GPS glitch: dist_lap_m (OBD) says the car traveled far less than the span.
    s_glitch = s.copy()
    s_glitch["dist_lap_m"] = s_glitch["dist_lap_m"] * 0.5
    assert span_time(s_glitch, 1830.5, 2574.5) is None

    # Bound never reached -> None.
    assert span_time(s, 1830.5, 9999.0) is None
    print("check_span_time OK")
```

Update `main`:

```python
def main() -> None:
    check_find_gear_bands()
    check_derive_gear()
    check_span_time()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py`
Expected: FAIL — `ImportError: cannot import name 'span_time'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/src/lap_analyzer/analysis.py`:

```python
def span_time(
    lap_samples: pd.DataFrame,
    dist_a: float,
    dist_b: float,
    obd_tol_m: float = 15.0,
) -> float | None:
    """Elapsed seconds for one lap between two track_dist_m positions.

    Interpolates t at the first crossing of dist_a and the first later crossing
    of dist_b. Returns None if either bound isn't crossed, or if the OBD-
    integrated distance over the interval differs from (dist_b - dist_a) by more
    than obd_tol_m (rejects GPS-glitched laps). Mirrors section_times().
    """
    s = lap_samples.sort_values("t")
    xs = s["track_dist_m"].to_numpy()
    ts = s["t"].to_numpy()
    ds = s["dist_lap_m"].to_numpy()
    if len(xs) < 2:
        return None
    ta = _first_crossing_t(xs, ts, dist_a)
    if ta is None:
        return None
    tb = _first_crossing_t(xs, ts, dist_b, after_t=ta)
    if tb is None:
        return None
    obd_dist = float(np.interp(tb, ts, ds) - np.interp(ta, ts, ds))
    if abs(obd_dist - (dist_b - dist_a)) > obd_tol_m:
        return None
    return float(tb - ta)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py`
Expected: PASS — prints `check_span_time OK`.

- [ ] **Step 5: Commit**

```bash
git add app/src/lap_analyzer/analysis.py app/notebooks/verify_gear_classification.py
git commit -m "feat: single-lap span_time helper"
```

---

### Task 4: `classify_t8_section` — label a lap's T8 entry

**Files:**
- Modify: `app/src/lap_analyzer/analysis.py`
- Test: `app/notebooks/verify_gear_classification.py`

- [ ] **Step 1: Write the failing test**

Add to `verify_gear_classification.py` (before `main`):

```python
_TRACK_DEF_STUB = {
    "corners": [
        {"id": "T8", "start_m": 1830.5, "end_m": 2036.5, "apex_m": 1876.5},
        {"id": "T9", "start_m": 2093.5, "end_m": 2296.5, "apex_m": 2168.5},
        {"id": "T10", "start_m": 2340.5, "end_m": 2501.5, "apex_m": 2425.5},
        {"id": "T11", "start_m": 2574.5, "end_m": 2727.5, "apex_m": 2643.5},
    ]
}


def _samples_over_span(ratio_fn, lo: float = 1760.0, hi: float = 2620.0,
                       speed: float = 80.0, dt: float = 0.046) -> pd.DataFrame:
    """Samples spanning [lo, hi] in track_dist_m; ratio_fn(dist) -> ratio."""
    dist = np.arange(lo, hi, 2.0)
    ratios = np.array([ratio_fn(d) for d in dist])
    return pd.DataFrame({
        "t": np.arange(len(dist)) * dt,
        "rpm": ratios * speed,
        "speed_mph": np.full(len(dist), speed),
        "track_dist_m": dist,
        "dist_lap_m": dist,
    })


def check_classify_t8_section() -> None:
    from lap_analyzer.analysis import classify_t8_section

    bands = np.array([54.0, 99.0, 180.0])

    # Downshift inside the T8 entry zone [1780.5, 1876.5]: ratio 54 -> 180 at 1850.
    ds = _samples_over_span(lambda d: 180.0 if d >= 1850.0 else 54.0)
    r = classify_t8_section(ds, _TRACK_DEF_STUB, bands)
    assert r["label"] == "downshift", r

    # No downshift anywhere: constant ratio.
    nd = _samples_over_span(lambda d: 54.0)
    r = classify_t8_section(nd, _TRACK_DEF_STUB, bands)
    assert r["label"] == "no-downshift", r

    # Downshift outside the T8 zone but inside the window (~2400 m, traffic).
    tr = _samples_over_span(lambda d: 180.0 if d >= 2400.0 else 54.0)
    r = classify_t8_section(tr, _TRACK_DEF_STUB, bands)
    assert r["label"] == "excluded" and r["excluded_reason"] == "traffic", r

    # No OBD across the whole window -> excluded/no-obd.
    no_obd = _samples_over_span(lambda d: 54.0)
    no_obd["rpm"] = 0.0
    r = classify_t8_section(no_obd, _TRACK_DEF_STUB, bands)
    assert r["label"] == "excluded" and r["excluded_reason"] == "no-obd", r
    print("check_classify_t8_section OK")
```

Update `main`:

```python
def main() -> None:
    check_find_gear_bands()
    check_derive_gear()
    check_span_time()
    check_classify_t8_section()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py`
Expected: FAIL — `ImportError: cannot import name 'classify_t8_section'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/src/lap_analyzer/analysis.py`:

```python
def classify_t8_section(
    samples: pd.DataFrame,
    track_def: dict,
    bands: np.ndarray,
    min_dwell_s: float = 0.3,
) -> dict:
    """Label one lap's T8 entry by downshift behaviour.

    Scans the detection window [T8.start_m - 50, T10.end_m] in track_dist_m.
    The window ends at T10 — before the T11 braking zone — so the routine
    downshift for T11 braking is never counted. Returns a dict with:
      label           - 'downshift' | 'no-downshift' | 'excluded'
      excluded_reason - 'traffic' | 'no-obd' | 'ambiguous' | None
      downshift_dist_m- track_dist_m of the single downshift, or None
      t8_apex_gear    - last confirmed gear at/before the T8 apex, or None

    downshift = exactly one downshift, inside the T8 entry zone
    [T8.start_m - 50, T8.apex_m]. no-downshift = zero downshifts in the window.
    Everything else (downshift inside the window but outside the zone = traffic,
    multiple downshifts, no OBD, too few samples) is excluded.
    """
    corners = {c["id"]: c for c in track_def["corners"]}
    t8 = corners["T8"]
    win_lo = t8["start_m"] - 50.0
    win_hi = corners["T10"]["end_m"]
    zone_hi = t8["apex_m"]

    w = samples[
        (samples["track_dist_m"] >= win_lo)
        & (samples["track_dist_m"] <= win_hi)
    ].sort_values("t")
    result = {
        "label": "excluded",
        "excluded_reason": "ambiguous",
        "downshift_dist_m": None,
        "t8_apex_gear": None,
    }
    if len(w) < 10:
        return result
    if (w["rpm"] == 0).all():
        result["excluded_reason"] = "no-obd"
        return result

    gear = derive_gear(w, bands, min_dwell_s=min_dwell_s).to_numpy()
    x = w["track_dist_m"].to_numpy()

    apex_gear = gear[x <= t8["apex_m"]]
    apex_gear = apex_gear[~np.isnan(apex_gear)]
    result["t8_apex_gear"] = float(apex_gear[-1]) if apex_gear.size else None

    valid = ~np.isnan(gear)
    gv = gear[valid]
    xv = x[valid]
    if gv.size < 2:
        return result

    downshift_locs = xv[1:][np.diff(gv) < 0]
    if downshift_locs.size == 0:
        result.update(label="no-downshift", excluded_reason=None)
        return result
    if downshift_locs.size > 1:
        return result

    loc = float(downshift_locs[0])
    result["downshift_dist_m"] = loc
    if loc <= zone_hi:
        result.update(label="downshift", excluded_reason=None)
    else:
        result["excluded_reason"] = "traffic"
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py`
Expected: PASS — prints `check_classify_t8_section OK`.

- [ ] **Step 5: Commit**

```bash
git add app/src/lap_analyzer/analysis.py app/notebooks/verify_gear_classification.py
git commit -m "feat: classify T8 entry by downshift behaviour"
```

---

### Task 5: Corpus eyeball in the verification script

**Files:**
- Modify: `app/notebooks/verify_gear_classification.py`

- [ ] **Step 1: Add the corpus eyeball function**

Add to `verify_gear_classification.py` (before `main`):

```python
def eyeball_corpus() -> None:
    """Print real gear bands + a ratio histogram + classification counts."""
    from lap_analyzer.analysis import (
        classify_t8_section, gear_bands, lap_index, load_corpus, load_samples,
    )
    from lap_analyzer.config import tracks_dir
    import json

    track = "ridge"
    bands = gear_bands(track)
    print("\n=== gear bands (ratio centers) ===")
    print(np.round(bands, 1))

    td = json.loads((tracks_dir() / f"{track}.json").read_text(encoding="utf-8"))

    laps = lap_index(load_corpus(track), track=track)
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    examples: dict[str, list] = {"downshift": [], "no-downshift": []}
    for r in laps.itertuples(index=False):
        try:
            s = load_samples(track, r.session_id, int(r.lap))
        except Exception:
            continue
        res = classify_t8_section(s, td, bands)
        counts[res["label"]] = counts.get(res["label"], 0) + 1
        if res["excluded_reason"]:
            reasons[res["excluded_reason"]] = reasons.get(res["excluded_reason"], 0) + 1
        if res["label"] in examples and len(examples[res["label"]]) < 5:
            examples[res["label"]].append(
                (r.session_id, int(r.lap), res["downshift_dist_m"], res["t8_apex_gear"])
            )

    print("\n=== classification counts ===")
    for k, v in sorted(counts.items()):
        print(f"  {k:14s} {v}")
    print("  excluded reasons:", reasons)
    print("\n=== example laps ===")
    for label, rows in examples.items():
        print(f"  {label}:")
        for sid, lap, loc, ag in rows:
            loc_s = f"{loc:.0f}m" if loc is not None else "—"
            print(f"    {sid} L{lap}  downshift@{loc_s}  t8_apex_gear={ag}")
```

Update `main`:

```python
def main() -> None:
    check_find_gear_bands()
    check_derive_gear()
    check_span_time()
    check_classify_t8_section()
    eyeball_corpus()
```

- [ ] **Step 2: Run the script**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py`
Expected: all four `check_* OK` lines, then gear band centers (roughly the well-separated values seen in the spec), classification counts, excluded-reason breakdown, and example laps. Confirm the bands look separated and the counts are plausible (most laps `downshift` or `no-downshift`, a minority `excluded`).

- [ ] **Step 3: Commit**

```bash
git add app/notebooks/verify_gear_classification.py
git commit -m "test: corpus eyeball for gear classification"
```

---

### Task 6: Extract `shared.py` and rewire `app.py`

**Files:**
- Create: `app/visualizer/shared.py`
- Modify: `app/visualizer/app.py:35-94` (TRACK constant + cached loaders + helpers)

- [ ] **Step 1: Create `app/visualizer/shared.py`**

```python
"""Shared cached loaders and helpers for the visualizer pages."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from lap_analyzer.analysis import lap_index, load_corpus, load_samples
from lap_analyzer.config import tracks_dir

TRACK = "ridge"


@st.cache_data(show_spinner=False)
def corpus() -> pd.DataFrame:
    return load_corpus(TRACK)


@st.cache_data(show_spinner=False)
def laps() -> pd.DataFrame:
    return lap_index(corpus(), track=TRACK)


@st.cache_data(show_spinner=False)
def track_def() -> dict:
    return json.loads((tracks_dir() / f"{TRACK}.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner="loading samples")
def samples(session_id: str, lap: int) -> pd.DataFrame:
    """One lap's samples. long_g negated to automotive convention (+ = accel).

    `rpm` is included so the downshift page can derive gear; app.py ignores it.
    """
    s = load_samples(TRACK, session_id, lap)[
        ["t", "lap", "track_dist_m", "dist_lap_m", "speed_mph", "speed_mph_gps",
         "throttle_norm", "long_g", "lat_g", "rpm"]
    ].copy()
    s["long_g"] = -s["long_g"]
    return s


def drop_gps_glitches(s: pd.DataFrame) -> pd.DataFrame:
    """Drop samples whose GPS projection disagrees with OBD-integrated distance.

    Primary signal: |track_dist_m - dist_lap_m|. On clean data these track
    within ~20-30m; on a TrackAddict inner-loop glitch they diverge by hundreds.
    The cummax pass mops up residual non-monotonic samples.
    """
    s = s.sort_values("t").reset_index(drop=True)
    small = s["track_dist_m"] < 500
    if small.any():
        s = s.iloc[small.idxmax():].reset_index(drop=True)
    diff = (s["track_dist_m"] - s["dist_lap_m"]).abs()
    s = s[diff < 50].reset_index(drop=True)
    rmax = s["track_dist_m"].cummax()
    return s[s["track_dist_m"] >= rmax - 1.0]


def session_hhmm(sid: str) -> str:
    return f"{sid[9:11]}:{sid[11:13]}"


def format_lap_time(s: float) -> str:
    """Lap time as M:SS.SS (e.g. 132.62s -> '2:12.62')."""
    m, rem = divmod(s, 60)
    return f"{int(m)}:{rem:05.2f}"
```

- [ ] **Step 2: Rewire `app.py` to import from `shared.py`**

In `app/visualizer/app.py`, delete the `TRACK = "ridge"` line (line 35) and the inline definitions of `_corpus`, `_laps`, `_track_def`, `_samples`, `_drop_gps_glitches` (lines 49-94), and the `session_hhmm` / `format_lap_time` definitions (lines 176-183). Replace the import block / removed region by adding this import after the existing `from lap_analyzer.config import tracks_dir` line:

```python
from shared import TRACK, drop_gps_glitches as _drop_gps_glitches
from shared import corpus as _corpus, laps as _laps, track_def as _track_def
from shared import samples as _samples, session_hhmm, format_lap_time
```

Leave everything else in `app.py` unchanged — `_insert_gap_breaks`, `_top_decile_traces`, `_envelope`, `_section_times`, and all rendering code keep working through the aliases.

- [ ] **Step 3: Verify the existing app still runs**

Run (PowerShell):

```powershell
cd app
$env:PYTHONPATH = "src"; python -m streamlit run visualizer/app.py
```

Expected: the visualizer loads in the browser exactly as before — corner selector, lap cascade, envelope plot, transit table all render. Stop the server (Ctrl+C) once confirmed.

- [ ] **Step 4: Commit**

```bash
git add app/visualizer/shared.py app/visualizer/app.py
git commit -m "refactor: extract shared visualizer loaders into shared.py"
```

---

### Task 7: New page — bands, controls, classification, summary header

**Files:**
- Create: `app/visualizer/pages/1_Downshift_T8-T11.py`

- [ ] **Step 1: Create the page skeleton**

```python
"""Visualizer page: T8-T11 downshift comparison.

Splits laps by whether a downshift was made entering T8, then compares the
T8 -> T11 section. See docs/superpowers/specs/2026-05-17-t8-t11-downshift-comparison-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from shared import (TRACK, corpus, drop_gps_glitches, format_lap_time, laps,
                    samples, session_hhmm, track_def)
from lap_analyzer.analysis import (classify_t8_section, derive_gear, gear_bands,
                                   section_times, span_time)

st.set_page_config(page_title="T8-T11 Downshift", layout="wide")
st.title("T8-T11 Downshift Comparison")

_td = track_def()
_corners = {c["id"]: c for c in _td["corners"]}
T8, T10, T11 = _corners["T8"], _corners["T10"], _corners["T11"]
SPAN_A = T8["start_m"]


@st.cache_data(show_spinner="deriving gear bands (one-time)")
def _bands() -> list[float]:
    return gear_bands(TRACK).tolist()


bands = np.array(_bands())
if bands.size == 0:
    st.error("No gear bands could be derived — OBD rpm/speed data is missing.")
    st.stop()

# --- controls ---------------------------------------------------------------

with st.sidebar:
    st.header("Filters")
    clean_only = st.checkbox("Clean laps only", value=True)
    top_n_deciles = st.slider("Pace: include top N deciles", 1, 10, 10,
                              help="Decile 0 = fastest 10%. 10 = all laps.")
    endpoint_choice = st.radio("Section endpoint", ["T11 entry", "end of T10"])

endpoint_m = T11["start_m"] if endpoint_choice == "T11 entry" else T10["end_m"]


# --- classify every lap -----------------------------------------------------

@st.cache_data(show_spinner="classifying laps (one-time per endpoint)")
def _classify_all(endpoint_m: float, bands_key: tuple) -> pd.DataFrame:
    b = np.array(bands_key)
    td = track_def()
    rows = []
    for r in laps().itertuples(index=False):
        try:
            s = samples(r.session_id, int(r.lap))
        except Exception:
            continue
        res = classify_t8_section(s, td, b)
        rows.append({
            "session_id": r.session_id,
            "lap": int(r.lap),
            "date": r.date,
            "lap_pace_decile": r.lap_pace_decile,
            "is_clean": getattr(r, "is_clean", None),
            "label": res["label"],
            "excluded_reason": res["excluded_reason"],
            "downshift_dist_m": res["downshift_dist_m"],
            "t8_apex_gear": res["t8_apex_gear"],
            "section_time_s": span_time(s, SPAN_A, endpoint_m),
        })
    return pd.DataFrame(rows)


cls = _classify_all(endpoint_m, tuple(bands.tolist()))

filt = cls.copy()
if clean_only:
    filt = filt[filt["is_clean"].fillna(False).astype(bool)]
filt = filt[filt["lap_pace_decile"].fillna(99) < top_n_deciles]

ds = filt[filt["label"] == "downshift"]
nd = filt[filt["label"] == "no-downshift"]
exc = filt[filt["label"] == "excluded"]

# --- summary header ---------------------------------------------------------

st.subheader("Groups")
c1, c2, c3 = st.columns(3)
c1.metric("Downshift laps", len(ds))
c2.metric("No-downshift laps", len(nd))
c3.metric("Excluded", len(exc))

if len(exc):
    breakdown = ", ".join(
        f"{k}={v}" for k, v in exc["excluded_reason"].value_counts().items()
    )
    st.caption(f"Excluded breakdown: {breakdown}")

st.markdown("**Pace mix** — lap count per pace decile (0 = fastest). "
            "Check the two groups are pace-matched before trusting the comparison.")
pace_mix = (filt[filt["label"].isin(["downshift", "no-downshift"])]
            .pivot_table(index="label", columns="lap_pace_decile",
                         values="lap", aggfunc="count", fill_value=0))
st.dataframe(pace_mix, use_container_width=True)

if len(ds) < 5 or len(nd) < 5:
    st.warning("One or both groups have fewer than 5 laps after filtering — "
               "loosen the filters; bands below may be unreliable.")
```

- [ ] **Step 2: Verify the page renders**

Run (PowerShell):

```powershell
cd app
$env:PYTHONPATH = "src"; python -m streamlit run visualizer/app.py
```

In the browser, open the "Downshift T8-T11" page from the sidebar nav. Expected: gear bands derive (one-time spinner), the three group metrics show non-zero counts, the excluded breakdown caption appears, and the pace-mix table renders. Toggle the filters and confirm counts update. Stop the server once confirmed.

- [ ] **Step 3: Commit**

```bash
git add app/visualizer/pages/1_Downshift_T8-T11.py
git commit -m "feat: T8-T11 downshift page — classification + summary"
```

---

### Task 8: New page — grouped channel envelopes + time-delta panel

**Files:**
- Modify: `app/visualizer/pages/1_Downshift_T8-T11.py` (append after the summary header)

- [ ] **Step 1: Append the envelope + delta plotting code**

Append to `app/visualizer/pages/1_Downshift_T8-T11.py`:

```python
# --- grouped channel envelopes ---------------------------------------------

DISPLAY_LO = SPAN_A - 200.0
DISPLAY_HI = endpoint_m + 200.0
DS_COLOR = "crimson"
ND_COLOR = "#2e8b57"


@st.cache_data(show_spinner="pooling group traces")
def _group_traces(keys: tuple, bands_key: tuple,
                  lo: float, hi: float) -> pd.DataFrame:
    """Pool samples for a set of (session_id, lap) keys across [lo, hi], with a
    derived `gear` column and GPS glitches removed."""
    b = np.array(bands_key)
    frames = []
    for sid, lap in keys:
        try:
            s = drop_gps_glitches(samples(sid, int(lap)))
        except Exception:
            continue
        s = s[(s["track_dist_m"] >= lo) & (s["track_dist_m"] <= hi)].copy()
        if s.empty:
            continue
        s["gear"] = derive_gear(s, b)
        frames.append(s[["track_dist_m", "speed_mph", "throttle_norm", "gear"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _envelope(traces: pd.DataFrame, channel: str,
              lo: float, hi: float, n: int = 160) -> pd.DataFrame:
    """Bin a pooled trace into p10/p50/p90 over track_dist_m."""
    if traces.empty:
        return pd.DataFrame(columns=["x", "p10", "p50", "p90"])
    t = traces[["track_dist_m", channel]].dropna()
    if t.empty:
        return pd.DataFrame(columns=["x", "p10", "p50", "p90"])
    edges = np.linspace(lo, hi, n + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cuts = pd.cut(t["track_dist_m"], edges, labels=False, include_lowest=True)
    grp = t.groupby(cuts)[channel]
    return pd.DataFrame({
        "x": centers,
        "p10": grp.quantile(0.10).reindex(range(n)).to_numpy(),
        "p50": grp.median().reindex(range(n)).to_numpy(),
        "p90": grp.quantile(0.90).reindex(range(n)).to_numpy(),
    }).dropna()


def _keys(group: pd.DataFrame) -> tuple:
    return tuple((r.session_id, r.lap) for r in group.itertuples(index=False))


def _median_time_curve(keys: tuple, grid: np.ndarray) -> np.ndarray | None:
    """Median (over laps) of cumulative time vs track_dist_m, anchored at grid[0]."""
    rows = []
    for sid, lap in keys:
        try:
            s = drop_gps_glitches(samples(sid, int(lap)))
        except Exception:
            continue
        s = s.sort_values("track_dist_m").drop_duplicates(
            subset="track_dist_m", keep="first")
        if len(s) < 2:
            continue
        ts = np.interp(grid, s["track_dist_m"].to_numpy(), s["t"].to_numpy())
        rows.append(ts - ts[0])
    if not rows:
        return None
    return np.median(np.vstack(rows), axis=0)


ds_keys = _keys(ds)
nd_keys = _keys(nd)
ds_traces = _group_traces(ds_keys, tuple(bands.tolist()), DISPLAY_LO, DISPLAY_HI)
nd_traces = _group_traces(nd_keys, tuple(bands.tolist()), DISPLAY_LO, DISPLAY_HI)

PANELS = [("speed_mph", "Speed (mph)"),
          ("throttle_norm", "Throttle"),
          ("gear", "Gear (0=low)")]

# Time-delta: no-downshift minus downshift. Positive = no-downshift slower.
grid = np.arange(SPAN_A, endpoint_m + 0.5, 2.0)
ds_curve = _median_time_curve(ds_keys, grid) if ds_keys else None
nd_curve = _median_time_curve(nd_keys, grid) if nd_keys else None
show_delta = ds_curve is not None and nd_curve is not None

n_rows = len(PANELS) + (1 if show_delta else 0)
fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05)


def _add_group(env: pd.DataFrame, row: int, color: str, name: str,
               draw_band: bool, show_legend: bool) -> None:
    if env.empty:
        return
    rgba = "rgba(220,20,60,0.15)" if color == DS_COLOR else "rgba(46,139,87,0.15)"
    if draw_band:
        fig.add_trace(go.Scatter(x=env["x"], y=env["p90"], mode="lines",
                                 line=dict(width=0), showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=env["x"], y=env["p10"], mode="lines",
                                 line=dict(width=0), fill="tonexty", fillcolor=rgba,
                                 showlegend=False), row=row, col=1)
    fig.add_trace(go.Scatter(x=env["x"], y=env["p50"], mode="lines",
                             line=dict(color=color, width=2.5),
                             name=name, showlegend=show_legend), row=row, col=1)


ds_band = len(ds) >= 5
nd_band = len(nd) >= 5
for i, (ch, label) in enumerate(PANELS, start=1):
    _add_group(_envelope(ds_traces, ch, DISPLAY_LO, DISPLAY_HI), i, DS_COLOR,
               f"downshift (n={len(ds)})", ds_band, i == 1)
    _add_group(_envelope(nd_traces, ch, DISPLAY_LO, DISPLAY_HI), i, ND_COLOR,
               f"no-downshift (n={len(nd)})", nd_band, i == 1)
    fig.update_yaxes(title_text=label, row=i, col=1)

# Corner shading for the T8-T11 corners.
for c in _td["corners"]:
    if c["end_m"] < DISPLAY_LO or c["start_m"] > DISPLAY_HI:
        continue
    for i in range(1, n_rows + 1):
        fig.add_vrect(x0=c["start_m"], x1=c["end_m"], fillcolor="lightgray",
                      opacity=0.12, line_width=0, layer="below", row=i, col=1)
    fig.add_annotation(x=(c["start_m"] + c["end_m"]) / 2, y=1.0, yref="y domain",
                       row=1, col=1, text=c["id"], showarrow=False,
                       yanchor="bottom", font=dict(size=10, color="gray"))

if show_delta:
    delta_row = n_rows
    delta = nd_curve - ds_curve
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"),
                  row=delta_row, col=1)
    fig.add_trace(go.Scatter(x=grid, y=delta, mode="lines",
                             line=dict(color="#ff8c00", width=2.5),
                             showlegend=False, fill="tozeroy",
                             fillcolor="rgba(255,140,0,0.10)"), row=delta_row, col=1)
    fig.update_yaxes(title_text="Δt: no-downshift − downshift (s)",
                     row=delta_row, col=1)

fig.update_xaxes(title_text="track_dist_m", row=n_rows, col=1)
fig.update_layout(height=240 * n_rows, margin=dict(l=20, r=20, t=30, b=30),
                  legend=dict(orientation="h", y=1.06, x=0))
st.plotly_chart(fig, use_container_width=True)
st.caption("Bands are p10–p90 per group (drawn only when a group has ≥5 laps); "
           "lines are group medians. Δt panel: positive = the no-downshift group "
           "is slower at that point. A downshift that pays off shows Δt rising "
           "across the T9–10 hill.")
```

- [ ] **Step 2: Verify the plots render**

Run the visualizer (`cd app; $env:PYTHONPATH = "src"; python -m streamlit run visualizer/app.py`) and open the Downshift page. Expected: a stacked figure with Speed / Throttle / Gear panels, each showing a crimson (downshift) and green (no-downshift) median line plus shaded bands, corner shading labelled T8–T11, and an orange Δt panel at the bottom. The Gear panel should show a clear step down for the downshift group in the T8 zone. Stop the server once confirmed.

- [ ] **Step 3: Commit**

```bash
git add app/visualizer/pages/1_Downshift_T8-T11.py
git commit -m "feat: T8-T11 downshift page — channel envelopes + time delta"
```

---

### Task 9: New page — section-time table + single-lap overlay

**Files:**
- Modify: `app/visualizer/pages/1_Downshift_T8-T11.py` (append after the plot)

- [ ] **Step 1: Append the section-time table and single-lap overlay**

Append to `app/visualizer/pages/1_Downshift_T8-T11.py`:

```python
# --- section-time comparison ------------------------------------------------

st.subheader(f"Section time — T8 entry → {endpoint_choice}")


def _median_or_nan(s: pd.Series) -> float:
    v = s.dropna()
    return float(v.median()) if len(v) else float("nan")


ds_med = _median_or_nan(ds["section_time_s"])
nd_med = _median_or_nan(nd["section_time_s"])
section_tbl = pd.DataFrame({
    "group": ["downshift", "no-downshift"],
    "laps timed": [int(ds["section_time_s"].notna().sum()),
                   int(nd["section_time_s"].notna().sum())],
    "median section (s)": [round(ds_med, 3), round(nd_med, 3)],
})
st.dataframe(section_tbl, use_container_width=True, hide_index=True)
if np.isfinite(ds_med) and np.isfinite(nd_med):
    diff = nd_med - ds_med
    faster = "downshift" if diff > 0 else "no-downshift"
    st.caption(f"Median delta: {abs(diff):.3f}s — the **{faster}** group is "
               f"faster across T8→{endpoint_choice} (median lap).")


@st.cache_data(show_spinner="computing per-corner section times (one-time)")
def _per_corner_times() -> pd.DataFrame:
    return section_times(TRACK, track_def())


per_corner = _per_corner_times()
label_map = cls.set_index(["session_id", "lap"])["label"]
pc = per_corner[per_corner["corner_id"].isin(["T8", "T9", "T10"])].copy()
pc["label"] = pc.set_index(["session_id", "lap"]).index.map(label_map)
pc = pc[pc["label"].isin(["downshift", "no-downshift"])]
if not pc.empty:
    per_corner_tbl = (pc.pivot_table(index="corner_id", columns="label",
                                     values="section_time_s", aggfunc="median")
                      .reindex(["T8", "T9", "T10"]).round(3))
    st.markdown("**Per-corner median section time (s)**")
    st.dataframe(per_corner_tbl, use_container_width=True)

# --- single-lap overlay -----------------------------------------------------

st.subheader("Overlay individual laps")
st.caption("Pick one lap from each group to overlay its raw trace on the bands.")


def _lap_options(group: pd.DataFrame) -> list:
    rows = group.sort_values("section_time_s", na_position="last")
    return [(r.session_id, r.lap) for r in rows.itertuples(index=False)]


def _lap_label(key) -> str:
    sid, lap = key
    row = cls[(cls["session_id"] == sid) & (cls["lap"] == lap)].iloc[0]
    sec = row["section_time_s"]
    sec_s = f"{sec:.2f}s" if pd.notna(sec) else "—"
    return f"{row['date']} {session_hhmm(sid)} L{lap} · {sec_s}"


oc1, oc2 = st.columns(2)
with oc1:
    ds_opts = _lap_options(ds)
    ds_pick = st.selectbox("Downshift lap", [None] + ds_opts,
                           format_func=lambda k: "— none —" if k is None else _lap_label(k))
with oc2:
    nd_opts = _lap_options(nd)
    nd_pick = st.selectbox("No-downshift lap", [None] + nd_opts,
                           format_func=lambda k: "— none —" if k is None else _lap_label(k))

if ds_pick is not None or nd_pick is not None:
    ofig = make_subplots(rows=len(PANELS), cols=1, shared_xaxes=True,
                         vertical_spacing=0.05)
    for i, (ch, label) in enumerate(PANELS, start=1):
        _add_group(_envelope(ds_traces, ch, DISPLAY_LO, DISPLAY_HI), i, DS_COLOR,
                   "downshift band", ds_band, False)
        _add_group(_envelope(nd_traces, ch, DISPLAY_LO, DISPLAY_HI), i, ND_COLOR,
                   "no-downshift band", nd_band, False)
        for pick, color, name in ((ds_pick, DS_COLOR, "downshift lap"),
                                  (nd_pick, ND_COLOR, "no-downshift lap")):
            if pick is None:
                continue
            sid, lap = pick
            s = drop_gps_glitches(samples(sid, int(lap)))
            s = s[(s["track_dist_m"] >= DISPLAY_LO)
                  & (s["track_dist_m"] <= DISPLAY_HI)].copy()
            s["gear"] = derive_gear(s, bands)
            s = s.sort_values("track_dist_m")
            ofig.add_trace(go.Scatter(x=s["track_dist_m"], y=s[ch], mode="lines",
                                      line=dict(color=color, width=2, dash="dot"),
                                      name=name, showlegend=(i == 1)),
                           row=i, col=1)
        ofig.update_yaxes(title_text=label, row=i, col=1)
    ofig.update_xaxes(title_text="track_dist_m", row=len(PANELS), col=1)
    ofig.update_layout(height=240 * len(PANELS), margin=dict(l=20, r=20, t=30, b=30),
                       legend=dict(orientation="h", y=1.08, x=0))
    st.plotly_chart(ofig, use_container_width=True)
```

- [ ] **Step 2: Verify the table and overlay render**

Run the visualizer and open the Downshift page. Expected: a section-time table with per-group median + lap counts, a delta caption naming the faster group, a per-corner (T8/T9/T10) median table, and two selectboxes. Pick a lap in each — a second figure appears overlaying the two picked laps (dotted) on the group bands. Stop the server once confirmed.

- [ ] **Step 3: Commit**

```bash
git add app/visualizer/pages/1_Downshift_T8-T11.py
git commit -m "feat: T8-T11 downshift page — section-time table + lap overlay"
```

---

## Self-Review

**Spec coverage:**
- Gear derivation (`derive_gear` + bands) — Tasks 1–2. ✓
- Three-way lap classification — Task 4. ✓
- Section span & timing (`span_time`, per-corner via `section_times`) — Tasks 3, 9. ✓
- `pages/` page + `shared.py` refactor — Tasks 6–9. ✓
- Controls (clean toggle, pace slider, endpoint radio) — Task 7. ✓
- Group summary header + pace mix — Task 7. ✓
- Channel envelopes (speed/throttle/gear) + time-delta — Task 8. ✓
- Section-time comparison table + per-corner breakdown — Task 9. ✓
- Single-lap overlay — Task 9. ✓
- Error handling: OBD dropout (`no-obd` exclusion), GPS glitches (`drop_gps_glitches`, `span_time` OBD check), thin groups (≥5 guard), empty filters (counts simply show 0; the thin-group warning covers the misleading-band case) — Tasks 4, 7, 8. ✓
- Verification script — Tasks 1–5. ✓

**Placeholder scan:** No TBD/TODO; every code step carries complete code.

**Type consistency:** `find_gear_bands`/`gear_bands` return `np.ndarray`; `derive_gear` returns a `pd.Series`; `classify_t8_section` returns the dict documented in Task 4 and consumed identically in Tasks 5 and 7; `span_time` returns `float | None`. Shared loaders are named `corpus`/`laps`/`track_def`/`samples`/`drop_gps_glitches` in `shared.py` and imported under those names (with `_`-prefixed aliases inside `app.py` only). Page helper names (`_envelope`, `_group_traces`, `_add_group`, `_keys`, `_median_time_curve`) are defined once in Task 8 and reused in Task 9.

**Note for executor:** Task 6 deletes specific line ranges from `app.py`; line numbers are from the current file — re-locate the named definitions if the file has shifted. Verify `app.py` still renders before moving on.
