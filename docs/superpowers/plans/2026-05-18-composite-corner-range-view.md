# Composite Corner-Range View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the main visualizer analyze a range of consecutive corners (From/To), showing a combined envelope, the cumulative time-delta curve, and a per-corner section-time breakdown.

**Architecture:** Two new pure functions in `analysis.py` (`section_range_bounds`, `range_section_times`); the rest is a generalization of `app/visualizer/app.py`'s single-corner focus path — the page already renders from one `(window_start, window_end)` pair, so a range just supplies a wider one.

**Tech Stack:** Python, pandas, numpy, Streamlit, Plotly.

**Note on commits:** This repo is not git-initialized. Either run `git init` first, or skip the `git commit` steps — all other steps stand alone.

**Spec:** `docs/superpowers/specs/2026-05-18-composite-corner-range-view-design.md`

**Two Python environments:** `app\.venv\Scripts\python.exe` has pandas/numpy/scipy but NOT streamlit/plotly — use it for the analysis-function checks. Bare `python` on PATH has streamlit 1.57 + plotly 6.7 — use it for the Streamlit `AppTest` smoke test.

---

## File Structure

- **Modify** `app/src/lap_analyzer/analysis.py` — append `section_range_bounds` and `range_section_times`.
- **Create** `app/notebooks/verify_range_section.py` — assertion checks for the two new functions (the project has no pytest).
- **Modify** `app/visualizer/app.py` — From/To pickers, section model, range-aware `find_best_lap`, header, envelope window, corner shading, apex lines, per-corner breakdown table; the old transit table commented out.

Run commands assume working directory `D:\Projects\lap-analyzer`.

---

### Task 1: `section_range_bounds` — section window for a corner range

**Files:**
- Modify: `app/src/lap_analyzer/analysis.py`
- Test: `app/notebooks/verify_range_section.py` (create)

- [ ] **Step 1: Write the failing test**

Create `app/notebooks/verify_range_section.py`:

```python
"""Verification for the composite corner-range section functions.

Run: app\.venv\Scripts\python.exe app/notebooks/verify_range_section.py
The project has no pytest harness; this script is the test harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_STUB = {
    "lap_length_internal_m": 3800.0,
    "corners": [
        {"id": "T8", "start_m": 1830.5, "end_m": 2036.5, "apex_m": 1876.5},
        {"id": "T9", "start_m": 2093.5, "end_m": 2296.5, "apex_m": 2168.5},
        {"id": "T10", "start_m": 2340.5, "end_m": 2501.5, "apex_m": 2425.5},
        {"id": "T11", "start_m": 2574.5, "end_m": 2727.5, "apex_m": 2643.5},
    ],
}


def check_section_range_bounds() -> None:
    from lap_analyzer.analysis import section_bounds, section_range_bounds

    # from == to must equal the single-corner section_bounds entry.
    sb = section_bounds(_STUB)
    assert section_range_bounds(_STUB, "T9", "T9") == sb["T9"], (
        section_range_bounds(_STUB, "T9", "T9"), sb["T9"])

    # A range T8..T10: a = T8.start-50; b = min(T11.start, T10.end+250).
    a, b = section_range_bounds(_STUB, "T8", "T10")
    assert a == 1830.5 - 50.0, a
    assert b == min(2574.5, 2501.5 + 250.0), b

    # Last corner (no next corner): b capped by lap length.
    a, b = section_range_bounds(_STUB, "T11", "T11")
    assert b == min(3800.0, 2727.5 + 250.0), b

    # to before from is rejected.
    try:
        section_range_bounds(_STUB, "T10", "T8")
        raise AssertionError("expected ValueError for reversed range")
    except ValueError:
        pass
    print("check_section_range_bounds OK")


def main() -> None:
    check_section_range_bounds()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_range_section.py`
Expected: FAIL — `ImportError: cannot import name 'section_range_bounds'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/src/lap_analyzer/analysis.py`:

```python
def section_range_bounds(
    track_def: dict,
    from_id: str,
    to_id: str,
    pre_m: float = 50.0,
    post_cap_m: float = 250.0,
) -> tuple[float, float]:
    """Section window spanning corners from_id..to_id inclusive.

    a = from_corner.start_m - pre_m; b = min(start of the corner after to_id,
    to_corner.end_m + post_cap_m). For from_id == to_id this returns exactly the
    same (a, b) as section_bounds(track_def)[from_id]. Raises ValueError if
    to_id precedes from_id in track order.
    """
    corners = sorted(track_def["corners"], key=lambda c: c["start_m"])
    ids = [c["id"] for c in corners]
    i_from = ids.index(from_id)
    i_to = ids.index(to_id)
    if i_to < i_from:
        raise ValueError(f"to_id {to_id!r} precedes from_id {from_id!r}")
    lap_len = float(track_def.get("lap_length_internal_m", corners[-1]["end_m"] + 200))
    next_start = corners[i_to + 1]["start_m"] if (i_to + 1) < len(corners) else lap_len
    a = max(0.0, corners[i_from]["start_m"] - pre_m)
    b = float(min(next_start, corners[i_to]["end_m"] + post_cap_m))
    return (a, b)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_range_section.py`
Expected: PASS — prints `check_section_range_bounds OK`.

- [ ] **Step 5: Commit** — SKIP (repo not git-initialized).

---

### Task 2: `range_section_times` — per-lap elapsed time across a range

**Files:**
- Modify: `app/src/lap_analyzer/analysis.py`
- Test: `app/notebooks/verify_range_section.py`

- [ ] **Step 1: Write the failing test**

Add to `verify_range_section.py` (before `main`):

```python
def check_range_section_times() -> None:
    """Smoke test against the real Ridge corpus (IO function — no synthetic input)."""
    import json
    from lap_analyzer.analysis import range_section_times
    from lap_analyzer.config import tracks_dir

    td = json.loads((tracks_dir() / "ridge.json").read_text(encoding="utf-8"))
    df = range_section_times("ridge", td, "T8", "T10")
    assert list(df.columns) == ["session_id", "lap", "section_time_s"], df.columns
    assert len(df) > 50, f"expected many timed laps, got {len(df)}"
    assert (df["section_time_s"] > 0).all(), "section times must be positive"
    assert df["section_time_s"].between(5, 60).mean() > 0.9, (
        "T8-T10 section times should mostly be 5-60s; got "
        f"{df['section_time_s'].describe()}")
    # from == to must agree with the per-corner section_times for that corner.
    from lap_analyzer.analysis import section_times
    one = range_section_times("ridge", td, "T9", "T9").set_index(["session_id", "lap"])
    per = section_times("ridge", td)
    per_t9 = per[per["corner_id"] == "T9"].set_index(["session_id", "lap"])
    common = one.index.intersection(per_t9.index)
    assert len(common) > 50, f"too few overlapping laps: {len(common)}"
    diff = (one.loc[common, "section_time_s"] - per_t9.loc[common, "section_time_s"]).abs()
    assert diff.max() < 0.01, f"from==to disagrees with section_times: max diff {diff.max()}"
    print("check_range_section_times OK")
```

Update `main`:

```python
def main() -> None:
    check_section_range_bounds()
    check_range_section_times()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_range_section.py`
Expected: FAIL — `ImportError: cannot import name 'range_section_times'`.

- [ ] **Step 3: Write minimal implementation**

Append to `app/src/lap_analyzer/analysis.py`:

```python
def range_section_times(
    track: str,
    track_def: dict,
    from_id: str,
    to_id: str,
    pre_m: float = 50.0,
    post_cap_m: float = 250.0,
    obd_distance_tolerance_m: float = 15.0,
) -> pd.DataFrame:
    """For every (session_id, lap), elapsed time across the from_id..to_id
    section window. Returns long-form: session_id, lap, section_time_s.

    Uses span_time() — the OBD-distance-validated per-lap window timer — so a
    lap is emitted only when it cleanly crosses both bounds and its OBD-
    integrated distance matches the nominal span (rejects GPS-glitched laps).
    For from_id == to_id this matches section_times() for that corner.
    """
    a, b = section_range_bounds(track_def, from_id, to_id, pre_m, post_cap_m)
    rows: list[tuple] = []
    root = sessions_dir(track)
    for sid_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sp = sid_dir / "samples.parquet"
        if not sp.exists():
            continue
        sid = sid_dir.name
        try:
            s = pd.read_parquet(sp, columns=["lap", "t", "track_dist_m", "dist_lap_m"])
        except Exception:
            continue
        for lap_n, g in s.groupby("lap"):
            st_ = span_time(g, a, b, obd_tol_m=obd_distance_tolerance_m)
            if st_ is not None:
                rows.append((sid, int(lap_n), st_))
    return pd.DataFrame(rows, columns=["session_id", "lap", "section_time_s"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `app\.venv\Scripts\python.exe app/notebooks/verify_range_section.py`
Expected: PASS — prints `check_section_range_bounds OK` then `check_range_section_times OK`.

- [ ] **Step 5: Commit** — SKIP (repo not git-initialized).

---

### Task 3: `app.py` — selection model, eligibility helper, range-aware best lap & sidebar

**Files:**
- Modify: `app/visualizer/app.py`

This task makes the "input" half of the page range-aware. The page will not run cleanly until Task 4 is also done; Task 4 ends with the smoke test.

- [ ] **Step 1: Add the two new analysis imports**

In `app/visualizer/app.py`, replace the import block:

```python
from lap_analyzer.analysis import (
    _first_crossing_t,
    lap_summary,
    section_bounds,
    section_times,
    top_decile_laps,
)
```

with:

```python
from lap_analyzer.analysis import (
    _first_crossing_t,
    lap_summary,
    range_section_times,
    section_bounds,
    section_range_bounds,
    section_times,
    top_decile_laps,
)
```

- [ ] **Step 2: Add the `_range_section_times` cached helper**

In `app/visualizer/app.py`, immediately after the `_section_times` function (the block ending at `return section_times(TRACK, _track_def())`), add:

```python


@st.cache_data(show_spinner="computing range section times")
def _range_section_times(from_id: str, to_id: str,
                         _version: int = _SECTION_TIMES_VERSION) -> pd.DataFrame:
    """Per-(session, lap) elapsed time across the from_id..to_id section window.
    Keyed on (from_id, to_id); recomputed when the selected range changes."""
    return range_section_times(TRACK, _track_def(), from_id, to_id)
```

- [ ] **Step 3: Replace `find_best_lap` with the range-aware version + eligibility helper**

In `app/visualizer/app.py`, replace the entire `find_best_lap` function (from `def find_best_lap(corner: str)` through its final `return` line) with:

```python
def eligible_section_keys(range_corners: list[str]) -> set[tuple[str, int]]:
    """(session_id, lap) pairs that are clean AND have every corner in
    range_corners flagged transit_reliable. Used for best-lap and decile pools."""
    clean_keys = set(map(tuple,
        laps[laps["is_clean"].fillna(False).astype(bool)][["session_id", "lap"]]
            .astype({"lap": int}).itertuples(index=False, name=None)))
    rel = corpus[corpus["corner_id"].isin(range_corners) & corpus["transit_reliable"]]
    cnt = rel.groupby(["session_id", "lap"]).size()
    fully_reliable = {(sid_, int(lap_)) for (sid_, lap_), n in cnt.items()
                      if n == len(range_corners)}
    return fully_reliable & clean_keys


def find_best_lap(range_corners: list[str], is_full_lap: bool,
                  range_sec_t: pd.DataFrame) -> tuple[str, int, float] | None:
    """Best reference lap for the chart + deltas.

    - is_full_lap: fastest overall clean + lap_reliable lap → (sid, lap, lap_time_s).
    - otherwise: fastest range section_time among laps that are clean AND have
      every corner in range_corners transit_reliable → (sid, lap, section_time_s).
      A single-corner range has one id, reproducing the old per-corner behavior.
    Returns None if no eligible lap.
    """
    if is_full_lap:
        rel = laps["lap_reliable"].fillna(False).astype(bool)
        elig = laps[laps["is_clean"].fillna(False).astype(bool) & rel]
        if elig.empty:
            return None
        r = elig.sort_values("lap_time_s").iloc[0]
        return (r["session_id"], int(r["lap"]), float(r["lap_time_s"]))
    eligible_keys = eligible_section_keys(range_corners)
    if not eligible_keys:
        return None
    eligible = range_sec_t[
        range_sec_t.set_index(["session_id", "lap"]).index.isin(eligible_keys)
    ]
    if eligible.empty:
        return None
    r = eligible.sort_values("section_time_s").iloc[0]
    return (r["session_id"], int(r["lap"]), float(r["section_time_s"]))
```

- [ ] **Step 4: Replace the corner picker with From/To pickers + the section model**

In `app/visualizer/app.py`, replace this block:

```python
with st.sidebar:
    st.header("Corner")
    corner_choice = pick_with_arrows(
        "Corner focus", ["full lap"] + corner_order, "pick_corner"
    )
    apex_def = st.radio(
        "Apex definition",
        ["visual", "speed-min", "lat-G peak"],
        captions=["Maps pin (kerb)", "per-lap min_speed_dist_m", "per-lap latg_peak_dist_m"],
    )
```

with:

```python
with st.sidebar:
    st.header("Corner")
    from_choice = pick_with_arrows("From", ["full lap"] + corner_order, "pick_from")
    if from_choice == "full lap":
        to_choice = "full lap"
    else:
        to_choice = pick_with_arrows(
            "To", corner_order[corner_order.index(from_choice):], "pick_to"
        )
    apex_def = st.radio(
        "Apex definition",
        ["visual", "speed-min", "lat-G peak"],
        captions=["Maps pin (kerb)", "per-lap min_speed_dist_m", "per-lap latg_peak_dist_m"],
    )

    # Resolve the From/To selection into a section model used across the page.
    is_full_lap = (from_choice == "full lap")
    if is_full_lap:
        range_corners = corner_order
        section_lo = section_hi = None
        section_label = "full lap"
    else:
        i_from = corner_order.index(from_choice)
        i_to = corner_order.index(to_choice)
        range_corners = corner_order[i_from:i_to + 1]
        section_lo, section_hi = section_range_bounds(track_def, from_choice, to_choice)
        section_label = (from_choice if from_choice == to_choice
                         else f"{from_choice}–{to_choice}")
    is_single = (not is_full_lap) and len(range_corners) == 1
    is_range = (not is_full_lap) and len(range_corners) > 1
    range_sec_t = (pd.DataFrame(columns=["session_id", "lap", "section_time_s"])
                   if is_full_lap else _range_section_times(from_choice, to_choice))
```

- [ ] **Step 5: Replace the sidebar "Fastest through" info block**

In `app/visualizer/app.py`, replace this block:

```python
    if corner_choice != "full lap":
        best = find_best_lap(corner_choice)
        if best is not None:
            best_sid, best_lap, best_section_s = best
            best_date = laps[(laps["session_id"] == best_sid) & (laps["lap"] == best_lap)].iloc[0]["date"]
            st.info(
                f"**Fastest through {corner_choice}**  \n"
                f"{best_date} {session_hhmm(best_sid)} L{best_lap} — "
                f"{best_section_s:.2f}s in section"
            )
            if st.button(f"⤵ Load fastest {corner_choice} lap", use_container_width=True):
                reset_cascade_to(best_date, best_sid, best_lap)
                st.rerun()
        else:
            st.info(f"No reliable + clean transit recorded at {corner_choice}.")
    else:
        best = find_best_lap("full lap")
        if best is not None:
            best_sid, best_lap, best_lap_time = best
            best_date = laps[(laps["session_id"] == best_sid) & (laps["lap"] == best_lap)].iloc[0]["date"]
            st.info(
                f"**Fastest full lap** (clean + lap_reliable)  \n"
                f"{best_date} {session_hhmm(best_sid)} L{best_lap} — "
                f"{format_lap_time(best_lap_time)}"
            )
            if st.button("⤵ Load fastest full lap", use_container_width=True):
                reset_cascade_to(best_date, best_sid, best_lap)
                st.rerun()
        else:
            st.info("No clean + lap_reliable laps available.")
```

with:

```python
    if not is_full_lap:
        best = find_best_lap(range_corners, False, range_sec_t)
        if best is not None:
            best_sid, best_lap, best_section_s = best
            best_date = laps[(laps["session_id"] == best_sid) & (laps["lap"] == best_lap)].iloc[0]["date"]
            st.info(
                f"**Fastest through {section_label}**  \n"
                f"{best_date} {session_hhmm(best_sid)} L{best_lap} — "
                f"{best_section_s:.2f}s in section"
            )
            if st.button(f"⤵ Load fastest {section_label} lap", use_container_width=True):
                reset_cascade_to(best_date, best_sid, best_lap)
                st.rerun()
        else:
            st.info(f"No reliable + clean lap recorded through {section_label}.")
    else:
        best = find_best_lap(range_corners, True, range_sec_t)
        if best is not None:
            best_sid, best_lap, best_lap_time = best
            best_date = laps[(laps["session_id"] == best_sid) & (laps["lap"] == best_lap)].iloc[0]["date"]
            st.info(
                f"**Fastest full lap** (clean + lap_reliable)  \n"
                f"{best_date} {session_hhmm(best_sid)} L{best_lap} — "
                f"{format_lap_time(best_lap_time)}"
            )
            if st.button("⤵ Load fastest full lap", use_container_width=True):
                reset_cascade_to(best_date, best_sid, best_lap)
                st.rerun()
        else:
            st.info("No clean + lap_reliable laps available.")
```

- [ ] **Step 6: Replace the lap-label section/decile block**

In `app/visualizer/app.py`, replace this block:

```python
    # When a corner is focused, build per-lap section time + corner-decile lookups
    # (decile ranks each clean+reliable@corner lap's section_time across the corpus).
    sec_for_session: dict[int, float] = {}
    corner_decile_for_session: dict[int, int] = {}
    if corner_choice != "full lap":
        s_session = sec_t[(sec_t["session_id"] == picked_sid)
                          & (sec_t["corner_id"] == corner_choice)]
        sec_for_session = dict(zip(s_session["lap"].astype(int), s_session["section_time_s"]))

        # Corpus-wide ranking pool: clean + transit_reliable at this corner
        clean_keys = set(map(tuple, laps[laps["is_clean"].fillna(False).astype(bool)]
                             [["session_id", "lap"]].astype({"lap": int})
                             .itertuples(index=False, name=None)))
        reliable_keys = set(map(tuple, corpus[
            (corpus["corner_id"] == corner_choice) & corpus["transit_reliable"]
        ][["session_id", "lap"]].astype({"lap": int})
          .itertuples(index=False, name=None)))
        eligible_keys = clean_keys & reliable_keys
        elig = sec_t[(sec_t["corner_id"] == corner_choice)
                     & sec_t.set_index(["session_id", "lap"]).index.isin(eligible_keys)]
        if len(elig) >= 10:
            deciles = pd.qcut(elig["section_time_s"], 10, labels=False, duplicates="drop")
            for sid_e, lap_e, d in zip(elig["session_id"], elig["lap"].astype(int), deciles):
                if sid_e == picked_sid and pd.notna(d):
                    corner_decile_for_session[int(lap_e)] = int(d)

    def lap_label(lp: int) -> str:
        row = view[(view["session_id"] == picked_sid) & (view["lap"] == lp)]
        if row.empty:
            return f"L{lp}"
        r = row.iloc[0]
        if corner_choice == "full lap":
            lt = f"{r['lap_time_s']:.2f}s"
            decile = int(r["lap_pace_decile"]) if pd.notna(r["lap_pace_decile"]) else -1
            d_txt = f"  d{decile}" if decile >= 0 else ""
            return f"L{lp}  {lt}{d_txt}"
        # Corner-focused: show section time + corner-decile only
        sec = f"{sec_for_session[lp]:.2f}s" if lp in sec_for_session else "—"
        cd = corner_decile_for_session.get(lp)
        cd_txt = f"  d{cd}" if cd is not None else ""
        return f"L{lp}  {corner_choice}: {sec}{cd_txt}"
```

with:

```python
    # When a section is focused, build per-lap range section time + section-decile
    # lookups (decile ranks each clean + fully-reliable lap's range section_time).
    sec_for_session: dict[int, float] = {}
    section_decile_for_session: dict[int, int] = {}
    if not is_full_lap:
        s_session = range_sec_t[range_sec_t["session_id"] == picked_sid]
        sec_for_session = dict(zip(s_session["lap"].astype(int), s_session["section_time_s"]))

        eligible_keys = eligible_section_keys(range_corners)
        elig = range_sec_t[
            range_sec_t.set_index(["session_id", "lap"]).index.isin(eligible_keys)
        ]
        if len(elig) >= 10:
            deciles = pd.qcut(elig["section_time_s"], 10, labels=False, duplicates="drop")
            for sid_e, lap_e, d in zip(elig["session_id"], elig["lap"].astype(int), deciles):
                if sid_e == picked_sid and pd.notna(d):
                    section_decile_for_session[int(lap_e)] = int(d)

    def lap_label(lp: int) -> str:
        row = view[(view["session_id"] == picked_sid) & (view["lap"] == lp)]
        if row.empty:
            return f"L{lp}"
        r = row.iloc[0]
        if is_full_lap:
            lt = f"{r['lap_time_s']:.2f}s"
            decile = int(r["lap_pace_decile"]) if pd.notna(r["lap_pace_decile"]) else -1
            d_txt = f"  d{decile}" if decile >= 0 else ""
            return f"L{lp}  {lt}{d_txt}"
        # Section-focused: show range section time + section-decile.
        sec = f"{sec_for_session[lp]:.2f}s" if lp in sec_for_session else "—"
        cd = section_decile_for_session.get(lp)
        cd_txt = f"  d{cd}" if cd is not None else ""
        return f"L{lp}  {section_label}: {sec}{cd_txt}"
```

---

### Task 4: `app.py` — header, envelope window, shading, apex lines, breakdown table

**Files:**
- Modify: `app/visualizer/app.py`

- [ ] **Step 1: Replace `best_ref` and the header metric row**

In `app/visualizer/app.py`, replace this block:

```python
# Reference lap for chart overlay + header deltas. Picked once per render.
best_ref = find_best_lap(corner_choice)

# Corner-relevant stats — deltas are vs the best lap for this corner.
# Full-lap view shows no metric row; the per-corner transit table carries detail.
if corner_choice != "full lap":
    t_row = corpus[(corpus["session_id"] == sid) & (corpus["lap"] == lap)
                   & (corpus["corner_id"] == corner_choice)]
    s_row = sec_t[(sec_t["session_id"] == sid) & (sec_t["lap"] == lap)
                  & (sec_t["corner_id"] == corner_choice)]
    section_t_val = float(s_row.iloc[0]["section_time_s"]) if not s_row.empty else None

    # Reference values: best lap's transit row + section time
    ref_section = None
    ref_transit = None
    if best_ref is not None:
        b_sid, b_lap, ref_section = best_ref
        b_t = corpus[(corpus["session_id"] == b_sid) & (corpus["lap"] == b_lap)
                     & (corpus["corner_id"] == corner_choice)]
        if not b_t.empty:
            ref_transit = b_t.iloc[0]

    is_self = best_ref is not None and best_ref[0] == sid and best_ref[1] == lap
    section_label = f"{corner_choice} section"

    c1, c2, c3, c4 = st.columns(4)
    help_text = _section_time_help(sid, lap, corner_choice) if section_t_val is None else None
    c1.metric(
        section_label,
        f"{section_t_val:.2f}s" if section_t_val is not None else "—",
        delta=(None if is_self or section_t_val is None or ref_section is None
               else f"{section_t_val - ref_section:+.2f}s vs best"),
        delta_color="inverse",  # lower = faster → green when negative
        help=help_text,
    )
    if not t_row.empty:
        r = t_row.iloc[0]
        def _delta(col, fmt):
            if is_self or ref_transit is None:
                return None
            return f"{r[col] - ref_transit[col]:+{fmt}} vs best"
        c2.metric("Min speed", f"{r['min_speed_mph']:.1f} mph", delta=_delta("min_speed_mph", ".1f"))
        c3.metric("Exit speed", f"{r['exit_speed_mph']:.1f} mph", delta=_delta("exit_speed_mph", ".1f"))
        c4.metric("Max lat-G", f"{r['max_lat_g']:.2f}", delta=_delta("max_lat_g", ".2f"))
```

with:

```python
# Reference lap for chart overlay + header deltas. Picked once per render.
best_ref = find_best_lap(range_corners, is_full_lap, range_sec_t)


def _range_section_time(sid_: str, lap_: int) -> float | None:
    """Range section time for one (sid, lap) from range_sec_t, or None."""
    r = range_sec_t[(range_sec_t["session_id"] == sid_) & (range_sec_t["lap"] == lap_)]
    return float(r.iloc[0]["section_time_s"]) if not r.empty else None


# Header metric row. Full-lap view shows no metric row. Single corner keeps the
# 4-metric row; a range shows section time + entry/exit speed of the complex.
if not is_full_lap:
    section_t_val = _range_section_time(sid, lap)
    is_self = best_ref is not None and best_ref[0] == sid and best_ref[1] == lap
    ref_section = best_ref[2] if best_ref is not None else None
    section_delta = (None if is_self or section_t_val is None or ref_section is None
                     else f"{section_t_val - ref_section:+.2f}s vs best")

    if is_single:
        t_row = corpus[(corpus["session_id"] == sid) & (corpus["lap"] == lap)
                       & (corpus["corner_id"] == from_choice)]
        ref_transit = None
        if best_ref is not None:
            b_t = corpus[(corpus["session_id"] == best_ref[0]) & (corpus["lap"] == best_ref[1])
                         & (corpus["corner_id"] == from_choice)]
            if not b_t.empty:
                ref_transit = b_t.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            f"{section_label} section",
            f"{section_t_val:.2f}s" if section_t_val is not None else "—",
            delta=section_delta, delta_color="inverse",
            help=(_section_time_help(sid, lap, from_choice)
                  if section_t_val is None else None),
        )
        if not t_row.empty:
            r = t_row.iloc[0]
            def _delta(col, fmt):
                if is_self or ref_transit is None:
                    return None
                return f"{r[col] - ref_transit[col]:+{fmt}} vs best"
            c2.metric("Min speed", f"{r['min_speed_mph']:.1f} mph", delta=_delta("min_speed_mph", ".1f"))
            c3.metric("Exit speed", f"{r['exit_speed_mph']:.1f} mph", delta=_delta("exit_speed_mph", ".1f"))
            c4.metric("Max lat-G", f"{r['max_lat_g']:.2f}", delta=_delta("max_lat_g", ".2f"))
    else:
        # Range: section time + entry speed (into From) + exit speed (out of To).
        entry_row = corpus[(corpus["session_id"] == sid) & (corpus["lap"] == lap)
                           & (corpus["corner_id"] == from_choice)]
        exit_row = corpus[(corpus["session_id"] == sid) & (corpus["lap"] == lap)
                          & (corpus["corner_id"] == to_choice)]
        ref_entry = ref_exit = None
        if best_ref is not None:
            be = corpus[(corpus["session_id"] == best_ref[0]) & (corpus["lap"] == best_ref[1])
                        & (corpus["corner_id"] == from_choice)]
            bx = corpus[(corpus["session_id"] == best_ref[0]) & (corpus["lap"] == best_ref[1])
                        & (corpus["corner_id"] == to_choice)]
            ref_entry = be.iloc[0] if not be.empty else None
            ref_exit = bx.iloc[0] if not bx.empty else None
        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"{section_label} section",
            f"{section_t_val:.2f}s" if section_t_val is not None else "—",
            delta=section_delta, delta_color="inverse",
            help=None if section_t_val is not None else
                 "Lap doesn't cleanly cross the range bounds (or failed OBD-distance "
                 "validation — likely a GPS glitch in the section).",
        )

        # Entry speed
        if not entry_row.empty:
            ev = float(entry_row.iloc[0]["entry_speed_mph"])
            ed = (None if is_self or ref_entry is None
                  else f"{ev - float(ref_entry['entry_speed_mph']):+.1f} vs best")
            c2.metric(f"Entry speed ({from_choice})", f"{ev:.1f} mph", delta=ed)
        else:
            c2.metric(f"Entry speed ({from_choice})", "—")
        # Exit speed
        if not exit_row.empty:
            xv = float(exit_row.iloc[0]["exit_speed_mph"])
            xd = (None if is_self or ref_exit is None
                  else f"{xv - float(ref_exit['exit_speed_mph']):+.1f} vs best")
            c3.metric(f"Exit speed ({to_choice})", f"{xv:.1f} mph", delta=xd)
        else:
            c3.metric(f"Exit speed ({to_choice})", "—")
```

- [ ] **Step 2: Make the envelope display window range-aware**

In `app/visualizer/app.py`, replace:

```python
if corner_choice != "full lap":
    a_, b_ = sec_bounds_all[corner_choice]
    display_a = max(0.0, a_ - DISPLAY_BUFFER_M)
    display_b = b_ + DISPLAY_BUFFER_M
    lap_s = lap_s[(lap_s["track_dist_m"] >= display_a)
                  & (lap_s["track_dist_m"] <= display_b)]
    if best_lap_s is not None:
        best_lap_s = best_lap_s[(best_lap_s["track_dist_m"] >= display_a)
                                & (best_lap_s["track_dist_m"] <= display_b)]
```

with:

```python
if not is_full_lap:
    display_a = max(0.0, section_lo - DISPLAY_BUFFER_M)
    display_b = section_hi + DISPLAY_BUFFER_M
    lap_s = lap_s[(lap_s["track_dist_m"] >= display_a)
                  & (lap_s["track_dist_m"] <= display_b)]
    if best_lap_s is not None:
        best_lap_s = best_lap_s[(best_lap_s["track_dist_m"] >= display_a)
                                & (best_lap_s["track_dist_m"] <= display_b)]
```

- [ ] **Step 3: Make the Δt grid range-aware**

In `app/visualizer/app.py`, replace:

```python
    if corner_choice != "full lap":
        grid_a, grid_b = display_a, display_b
    else:
```

with:

```python
    if not is_full_lap:
        grid_a, grid_b = display_a, display_b
    else:
```

- [ ] **Step 4: Make the per-channel envelope filter range-aware**

In `app/visualizer/app.py`, replace:

```python
for i, (ch, label) in enumerate(PANELS, start=1):
    env_i = _envelope(ch)
    if corner_choice != "full lap":
        env_i = env_i[(env_i["track_dist_m"] >= display_a)
                      & (env_i["track_dist_m"] <= display_b)]
```

with:

```python
for i, (ch, label) in enumerate(PANELS, start=1):
    env_i = _envelope(ch)
    if not is_full_lap:
        env_i = env_i[(env_i["track_dist_m"] >= display_a)
                      & (env_i["track_dist_m"] <= display_b)]
```

- [ ] **Step 5: Make corner shading + apex lines range-aware**

In `app/visualizer/app.py`, replace this block:

```python
# Corner shading + labels.
# - full lap: every corner shaded lightly with its T# label (no apex line, no focus).
# - corner focused: all visible corners shaded; the focused one darker + apex line.
if corner_choice == "full lap":
    visible_corners = list(track_def["corners"])
else:
    visible_corners = [c for c in track_def["corners"]
                       if c["end_m"] >= display_a and c["start_m"] <= display_b]

for c in visible_corners:
    is_focused = (c["id"] == corner_choice)
    opacity = 0.30 if is_focused else 0.10
    for i in range(1, n_rows + 1):
        fig.add_vrect(x0=c["start_m"], x1=c["end_m"], fillcolor="lightgray",
                      opacity=opacity, line_width=0, layer="below", row=i, col=1)
    cx = (c["start_m"] + c["end_m"]) / 2
    fig.add_annotation(
        x=cx, y=1.0, yref="y domain", row=1, col=1, text=c["id"],
        font=dict(color="black" if is_focused else "gray",
                  size=12 if is_focused else 10),
        showarrow=False, yanchor="bottom", yshift=2,
    )

# Apex line only when a specific corner is focused
if corner_choice != "full lap":
    c = corner_apex[corner_choice]
    apex_x: float | None = None
    if apex_def == "visual":
        apex_x = float(c["apex_m"])
    else:
        t_row_for_apex = corpus[(corpus["session_id"] == sid) & (corpus["lap"] == lap)
                                & (corpus["corner_id"] == corner_choice)]
        if not t_row_for_apex.empty:
            col_for_apex = "min_speed_dist_m" if apex_def == "speed-min" else "latg_peak_dist_m"
            apex_x = float(t_row_for_apex[col_for_apex].iloc[0])
            if np.isnan(apex_x):
                apex_x = None
    if apex_x is not None:
        for i in range(1, n_rows + 1):
            extras = (dict(annotation_text=f"apex ({apex_def})", annotation_position="top")
                      if i == 1 else {})
            fig.add_vline(x=apex_x, line=dict(color="gold", width=2), row=i, col=1, **extras)
```

with:

```python
# Corner shading + labels.
# - full lap: every corner shaded lightly with its T# label.
# - section focused: all visible corners shaded; corners inside the From..To
#   range darker, with an apex line each.
if is_full_lap:
    visible_corners = list(track_def["corners"])
else:
    visible_corners = [c for c in track_def["corners"]
                       if c["end_m"] >= display_a and c["start_m"] <= display_b]

for c in visible_corners:
    in_range = (not is_full_lap) and (c["id"] in range_corners)
    opacity = 0.30 if in_range else 0.10
    for i in range(1, n_rows + 1):
        fig.add_vrect(x0=c["start_m"], x1=c["end_m"], fillcolor="lightgray",
                      opacity=opacity, line_width=0, layer="below", row=i, col=1)
    cx = (c["start_m"] + c["end_m"]) / 2
    fig.add_annotation(
        x=cx, y=1.0, yref="y domain", row=1, col=1, text=c["id"],
        font=dict(color="black" if in_range else "gray",
                  size=12 if in_range else 10),
        showarrow=False, yanchor="bottom", yshift=2,
    )

# Apex line for each corner in the focused range.
if not is_full_lap:
    for cid in range_corners:
        c = corner_apex[cid]
        apex_x: float | None = None
        if apex_def == "visual":
            apex_x = float(c["apex_m"])
        else:
            t_row_for_apex = corpus[(corpus["session_id"] == sid) & (corpus["lap"] == lap)
                                    & (corpus["corner_id"] == cid)]
            if not t_row_for_apex.empty:
                col_for_apex = "min_speed_dist_m" if apex_def == "speed-min" else "latg_peak_dist_m"
                apex_x = float(t_row_for_apex[col_for_apex].iloc[0])
                if np.isnan(apex_x):
                    apex_x = None
        if apex_x is not None:
            for i in range(1, n_rows + 1):
                extras = (dict(annotation_text=cid, annotation_position="top")
                          if i == 1 else {})
                fig.add_vline(x=apex_x, line=dict(color="gold", width=1.5),
                              row=i, col=1, **extras)
```

- [ ] **Step 6: Replace the transit table with the per-corner section-time breakdown**

In `app/visualizer/app.py`, replace this entire block:

```python
# --- transit table ----------------------------------------------------------

st.subheader("Per-corner transit — selected lap vs top-decile median")

lap_t = corpus[(corpus["session_id"] == sid) & (corpus["lap"] == lap)]
# pool / pool_t already computed above for header metrics
metric_cols = ["min_speed_mph", "apex_speed_mph", "exit_speed_mph",
               "max_lat_g", "apex_dist_offset_m", "latg_peak_offset_m",
               "time_in_corner_s"]
ref = pool_t.groupby("corner_id")[metric_cols].median()
mine = lap_t.set_index("corner_id")[metric_cols + ["transit_reliable"]]
delta = (mine[metric_cols] - ref).round(2)
delta.columns = [f"Δ {c}" for c in delta.columns]
table = pd.concat([mine[metric_cols].round(2), delta], axis=1)
table.insert(0, "reliable", mine["transit_reliable"].map({True: "✓", False: "✗"}))
table = table.reindex([c for c in corner_order if c in table.index])

st.dataframe(table, use_container_width=True, height=560)

st.caption(
    f"Reference pool: top-decile clean laps ({len(pool)} laps). "
    "Spatial views (envelope band, line position) use the STANDARD-reliable subset; "
    "kinematic channels (speed/throttle/brake/lat-G) are trusted even on unreliable laps."
)
```

with:

```python
# --- per-corner section-time breakdown --------------------------------------

# Shown for a range or full lap (one row per corner in scope); a single corner
# has no table — its section time is already in the header metric.
#
# The old "Per-corner transit — selected lap vs top-decile median" table is
# retired (kept here, commented, in case it is wanted back):
#
# st.subheader("Per-corner transit — selected lap vs top-decile median")
# lap_t = corpus[(corpus["session_id"] == sid) & (corpus["lap"] == lap)]
# metric_cols = ["min_speed_mph", "apex_speed_mph", "exit_speed_mph",
#                "max_lat_g", "apex_dist_offset_m", "latg_peak_offset_m",
#                "time_in_corner_s"]
# ref = pool_t.groupby("corner_id")[metric_cols].median()
# mine = lap_t.set_index("corner_id")[metric_cols + ["transit_reliable"]]
# delta = (mine[metric_cols] - ref).round(2)
# delta.columns = [f"Δ {c}" for c in delta.columns]
# table = pd.concat([mine[metric_cols].round(2), delta], axis=1)
# table.insert(0, "reliable", mine["transit_reliable"].map({True: "✓", False: "✗"}))
# table = table.reindex([c for c in corner_order if c in table.index])
# st.dataframe(table, use_container_width=True, height=560)

if not is_single:
    st.subheader("Per-corner section time")
    sel = (sec_t[(sec_t["session_id"] == sid) & (sec_t["lap"] == lap)]
           .set_index("corner_id")["section_time_s"])
    best_sel = None
    if best_ref is not None:
        best_sel = (sec_t[(sec_t["session_id"] == best_ref[0])
                          & (sec_t["lap"] == best_ref[1])]
                    .set_index("corner_id")["section_time_s"])
    rows = []
    for cid in range_corners:
        s_sel = sel.get(cid)
        s_best = best_sel.get(cid) if best_sel is not None else None
        rows.append({
            "corner": cid,
            "section time (s)": round(s_sel, 3) if s_sel is not None else None,
            "Δ vs best (s)": (round(s_sel - s_best, 3)
                              if s_sel is not None and s_best is not None else None),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Per-corner section time from section_times(); Δ is vs the fastest "
               "lap through the focused section. The continuous Δt curve above "
               "shows where time moved; this table attributes it per corner.")
```

- [ ] **Step 7: Smoke-test the page**

Run from `D:\Projects\lap-analyzer` with bare `python` (the venv has no streamlit):

```powershell
cd app\visualizer
$env:PYTHONPATH = "..\src"; python -c "from streamlit.testing.v1 import AppTest; at = AppTest.from_file('app.py'); at.run(timeout=300); print('SMOKE FAIL' if at.exception else 'SMOKE OK'); [print(e.value) for e in at.exception]"
```

Expected: `SMOKE OK`, no exception. (`use_container_width` deprecation warnings are pre-existing and fine.) The default render is full-lap, so this exercises the full-lap path.

- [ ] **Step 8: Smoke-test a range and a single corner**

Exercise the range and single-corner paths headlessly by setting the From/To pickers:

```powershell
cd app\visualizer
$env:PYTHONPATH = "..\src"; python -c "from streamlit.testing.v1 import AppTest; at = AppTest.from_file('app.py'); at.run(timeout=300); at.session_state['pick_from_sel']='T8'; at.session_state['pick_to_sel']='T10'; at.run(timeout=300); print('RANGE FAIL' if at.exception else 'RANGE OK'); [print(e.value) for e in at.exception]; at.session_state['pick_from_sel']='T8'; at.session_state['pick_to_sel']='T8'; at.run(timeout=300); print('SINGLE FAIL' if at.exception else 'SINGLE OK'); [print(e.value) for e in at.exception]"
```

Expected: `RANGE OK` and `SINGLE OK`, no exceptions. If a picker key differs, inspect `at.session_state` keys after the first run and adjust — the goal is to set From=T8/To=T10 (range) and From=T8/To=T8 (single corner) and confirm both render without exception.

- [ ] **Step 9: Manual verification**

Run the visualizer: `cd app; $env:PYTHONPATH = "src"; python -m streamlit run visualizer/app.py`. Confirm:
- Default full-lap view renders; the per-corner section-time table lists all corners.
- Pick From=T8, To=T8 → single-corner view is unchanged from before (4-metric header, no breakdown table, one apex line).
- Pick From=T8, To=T10 → range view: 3-metric header (section time / entry speed / exit speed), the T8/T9/T10 corners shaded darker, an apex line per corner, the Δt curve spans the range, and the per-corner section-time table shows T8/T9/T10 with deltas.
- Pick From=T1, To=T5 and From=T3, To=T5 → both render.
Stop the server once confirmed.

- [ ] **Step 10: Commit** — SKIP (repo not git-initialized).

---

## Self-Review

**Spec coverage:**
- `section_range_bounds` (`from==to` equals `section_bounds[corner]`) — Task 1. ✓
- Range section time, directly computed, GPS-glitch rejected via `span_time` — Task 2. ✓
- From/To pickers, To constrained ≥ From, From==To = single corner unchanged — Task 3 Step 4. ✓
- `full lap` mode preserved — Task 3 Steps 4–5, Task 4. ✓
- Range-aware `find_best_lap` (clean + all-range-corners `transit_reliable`) — Task 3 Step 3. ✓
- Lap-label shows range section time — Task 3 Step 6. ✓
- Header: single-corner 4-metric unchanged; range 3-metric (section time / entry / exit) — Task 4 Step 1. ✓
- Envelope over the range window, Δt curve, darker shading for in-range corners, apex line per range corner — Task 4 Steps 2–5. ✓
- Per-corner section-time breakdown table (range + full lap; not single corner) — Task 4 Step 6. ✓
- Old transit table commented out, not deleted — Task 4 Step 6. ✓
- Error handling: To<From impossible (picker constrained); uncrossed range → "—" + help text; glitched laps rejected by `span_time`'s OBD check; no eligible lap → existing "No reliable + clean" message — Tasks 3–4. ✓
- Verification — Tasks 1, 2 (`verify_range_section.py`), Task 4 Steps 7–9. ✓

**Placeholder scan:** No TBD/TODO; every code step carries complete code.

**Type consistency:** `section_range_bounds` returns `tuple[float, float]`; `range_section_times` / `_range_section_times` return a DataFrame with columns `session_id, lap, section_time_s`. `find_best_lap(range_corners, is_full_lap, range_sec_t)` and `eligible_section_keys(range_corners)` signatures match every call site (Task 3 Steps 3, 5, 6; Task 4 Step 1). Section-model names (`is_full_lap`, `is_single`, `is_range`, `range_corners`, `section_lo`, `section_hi`, `section_label`, `range_sec_t`, `from_choice`, `to_choice`) are defined once in Task 3 Step 4 and used consistently thereafter.

**Note for the executor:** Task 3 leaves `app.py` referencing names not yet defined until Task 4 completes — do not smoke-test between Task 3 and Task 4. The first smoke test is Task 4 Step 7.
