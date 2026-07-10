"""Lap Analyzer visualizer.

Run from the repo root (PowerShell):
    $env:PYTHONPATH = "."; python -m streamlit run visualizer/app.py

Run from the repo root (bash):
    PYTHONPATH=. python -m streamlit run visualizer/app.py

Docs: docs/VISUALIZER.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from lap_analyzer.analysis import (
    _first_crossing_t,
    lap_summary,
    range_section_times,
    section_bounds,
    section_range_bounds,
    section_times,
    top_decile_laps,
)
from lap_analyzer.fused_axis import compute_fused_dist
from shared import available_tracks, current_track, drop_gps_glitches as _drop_gps_glitches
from shared import corpus as _corpus, laps as _laps, track_def as _track_def
from shared import samples as _samples, session_hhmm, format_lap_time

CHANNELS = {
    "speed_mph": "Speed (OBD)",
    "speed_mph_gps": "Speed (GPS)",
    "throttle_norm": "Throttle",
    "long_g": "Longitudinal G",  # canonical convention (+ = accel, − = brake)
    "lat_g": "Lateral G",
}

st.set_page_config(page_title="Lap Analyzer", layout="wide")


def _fix_dropdown_overflow() -> None:
    """Keep selectbox dropdowns inside the viewport.

    Streamlit's selectbox popover (react-aria, since ~1.59) opens *below* the
    input and does not flip up when the input sits near the bottom of the page —
    so the list renders past the fold and, being ``position: fixed``, the page
    can't scroll to reach it (most visible on the Lap picker at the sidebar's
    bottom). This injects a MutationObserver into the parent document that, when
    a dropdown would overflow, flips it above its input (and caps a very long
    list to the viewport with internal scroll). No-op when the native placement
    already fits. Remove once the upstream placement bug is fixed.
    """
    st.components.v1.html(
        """
<script>
(function () {
  const win = window.parent, doc = win.document;
  const SEL = '[data-testid="stSelectboxVirtualDropdown"]';
  const MARGIN = 8, INPUT_H = 40;
  function fit(pop) {
    const cs = getComputedStyle(pop);
    const m = cs.transform.match(/matrix\\(1, 0, 0, 1, ([-\\d.]+), ([-\\d.]+)\\)/);
    if (!m) return;                                   // not transform-positioned yet
    const tx = parseFloat(m[1]), ty = parseFloat(m[2]);
    const vh = win.innerHeight, avail = vh - 2 * MARGIN;
    const listbox = pop.querySelector('[role="listbox"]');
    let h = pop.getBoundingClientRect().height;
    if (h > avail && listbox) {                       // taller than the viewport: cap + scroll
      listbox.style.setProperty('max-height', avail + 'px', 'important');
      h = pop.getBoundingClientRect().height;
    }
    if (ty + h <= vh - MARGIN && ty >= MARGIN) return; // native placement already fits
    const above = ty - INPUT_H - h - MARGIN;           // flip above the input if it fits
    const newTy = above >= MARGIN ? above : Math.max(MARGIN, vh - MARGIN - h);
    if (Math.abs(newTy - ty) > 1)
      pop.style.setProperty('transform', 'translate(' + tx + 'px, ' + newTy + 'px)', 'important');
  }
  function scan() { doc.querySelectorAll(SEL).forEach(fit); }
  new MutationObserver(scan).observe(doc.body,
    { childList: true, subtree: true, attributes: true, attributeFilter: ['style'] });
  win.addEventListener('resize', scan);
  scan();
})();
</script>
""",
        height=0,
    )


_fix_dropdown_overflow()


# Bump _GLITCH_FILTER_VERSION whenever _drop_gps_glitches logic changes — its
# value is baked into the cached function's source via the keyword default below,
# which forces @st.cache_data to invalidate. Otherwise the helper's source is
# invisible to Streamlit's hash and stale envelopes survive a reload.
_GLITCH_FILTER_VERSION = 6


@st.cache_data(show_spinner="loading top-decile samples (one-time)")
def _top_decile_traces(track: str, _filter_version: int = _GLITCH_FILTER_VERSION) -> pd.DataFrame:
    """Long-form: all top-decile lap samples concatenated, with per-lap GPS-glitch
    samples removed so envelope bins aren't polluted by backward/forward teleports.
    Per-channel envelope is derived on demand so adding a channel doesn't invalidate cache.
    """
    pool = top_decile_laps(_corpus(track))
    frames = []
    for sid, lp in pool:
        try:
            s = _samples(track, sid, lp)
        except Exception:
            continue
        s = _drop_gps_glitches(s)
        frames.append(s.assign(lap_key=f"{sid}-L{lp}"))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def _envelope(track: str, channel: str, n_points: int = 400) -> pd.DataFrame:
    traces = _top_decile_traces(track)
    if traces.empty or channel not in traces.columns:
        return pd.DataFrame(columns=["track_dist_m", "p10", "p50", "p90"])
    t = traces[["track_dist_m", channel]].dropna()
    if t.empty:
        return pd.DataFrame(columns=["track_dist_m", "p10", "p50", "p90"])
    edges = np.linspace(t["track_dist_m"].min(), t["track_dist_m"].max(), n_points + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cuts = pd.cut(t["track_dist_m"], edges, labels=False, include_lowest=True)
    grp = t.groupby(cuts)[channel]
    return pd.DataFrame({
        "track_dist_m": centers,
        "p10": grp.quantile(0.10).reindex(range(n_points)).to_numpy(),
        "p50": grp.median().reindex(range(n_points)).to_numpy(),
        "p90": grp.quantile(0.90).reindex(range(n_points)).to_numpy(),
    }).dropna()


def _insert_gap_breaks(df: pd.DataFrame, x_col: str, gap_threshold: float = 30.0) -> pd.DataFrame:
    """Insert NaN-y rows where x has a gap > threshold, so Plotly breaks the line."""
    df = df.reset_index(drop=True)
    if len(df) < 2:
        return df
    gap_idxs = df.index[df[x_col].diff() > gap_threshold].tolist()
    if not gap_idxs:
        return df
    pieces = []
    last = 0
    for gi in gap_idxs:
        pieces.append(df.iloc[last:gi])
        nan_row = {c: np.nan for c in df.columns}
        nan_row[x_col] = (df[x_col].iloc[gi - 1] + df[x_col].iloc[gi]) / 2
        pieces.append(pd.DataFrame([nan_row]))
        last = gi
    pieces.append(df.iloc[last:])
    return pd.concat(pieces, ignore_index=True)


# Bump _SECTION_TIMES_VERSION when section_times() / span_time() / range
# section-time logic changes — baked into the cached functions' source via the
# default arg below, so @st.cache_data invalidates on reload.
_SECTION_TIMES_VERSION = 5


@st.cache_data(show_spinner="computing per-corner section times (one-time)")
def _section_times(track: str, _version: int = _SECTION_TIMES_VERSION) -> pd.DataFrame:
    return section_times(track, _track_def(track))


@st.cache_data(show_spinner="computing range section times")
def _range_section_times(track: str, from_id: str, to_id: str,
                         _version: int = _SECTION_TIMES_VERSION) -> pd.DataFrame:
    """Per-(session, lap) elapsed time across the from_id..to_id section window.
    Keyed on (track, from_id, to_id); recomputed when the selected range changes."""
    return range_section_times(track, _track_def(track), from_id, to_id)


def pick_with_arrows(label: str, options: list, key: str, format_func=str):
    """Selectbox with prev/next stepper buttons. Selection state IS the widget's own key.

    The widget key is `f"{key}_sel"` and holds the chosen option value (not an index).
    Setting that key programmatically (in a button on_click) is honored on the next
    rerun; this is what makes `+`/`−` and cascade resets visually update the dropdown.
    """
    sel_key = f"{key}_sel"
    # If the stored value isn't in the current options (cascade just changed), reset.
    if sel_key not in st.session_state or st.session_state[sel_key] not in options:
        st.session_state[sel_key] = options[0]

    def step(d: int):
        cur = st.session_state[sel_key]
        i = options.index(cur) if cur in options else 0
        st.session_state[sel_key] = options[max(0, min(len(options) - 1, i + d))]

    c1, c2, c3 = st.columns([1, 6, 1])
    c1.button("<", key=f"{key}_prev", on_click=step, args=(-1,), use_container_width=True)
    c3.button(">", key=f"{key}_next", on_click=step, args=(1,), use_container_width=True)
    return c2.selectbox(label, options, key=sel_key, format_func=format_func,
                        label_visibility="collapsed")


def reset_cascade_to(date_str, sid: str, lap: int):
    st.session_state["pick_date_sel"] = date_str
    st.session_state["pick_time_sel"] = sid
    st.session_state["pick_lap_sel"] = lap


OBD_DISTANCE_TOLERANCE_M = 15.0  # mirror of section_times() default; for tooltip text


def _section_time_help(track: str, sid_: str, lap_: int, corner: str) -> str | None:
    """Diagnostic text explaining why section_time is missing for this (sid, lap, corner)."""
    if corner == "full lap":
        return None
    a, b = sec_bounds_all[corner]
    try:
        s = _samples(track, sid_, lap_).sort_values("t").reset_index(drop=True)
    except Exception:
        return "Samples not available for this lap."
    xs = s["track_dist_m"].to_numpy()
    ts = s["t"].to_numpy()
    ds = s["dist_lap_m"].to_numpy()
    ta = _first_crossing_t(xs, ts, a)
    if ta is None:
        return f"Lap didn't cross the section start ({a:.0f}m) — section_time can't be computed."
    tb = _first_crossing_t(xs, ts, b, after_t=ta)
    if tb is None:
        return f"Lap didn't cross the section end ({b:.0f}m) — section_time can't be computed."
    obd_dist = float(np.interp(tb, ts, ds) - np.interp(ta, ts, ds))
    disc = obd_dist - (b - a)
    direction = "shorter" if disc < 0 else "longer"
    return (
        f"section_time rejected by OBD-distance validation. "
        f"OBD says the car physically traveled {obd_dist:.0f}m, "
        f"but the section nominally spans {b - a:.0f}m "
        f"({disc:+.0f}m, {abs(disc):.0f}m {direction}; threshold ±{OBD_DISTANCE_TOLERANCE_M:.0f}m). "
        f"Likely a GPS glitch in this section."
    )


# --- sidebar: Track picker --------------------------------------------------
# Must run BEFORE any data load that depends on current_track(). Writing to
# st.session_state["track"] via the `key=` arg means current_track() (read just
# below) picks up the user's choice for this render — switching tracks reruns
# the page with the new selection and all the cached loaders below cache-key
# on `track` so the cache doesn't serve stale data from the previous track.

with st.sidebar:
    _tracks = available_tracks()
    if not _tracks:
        st.error("No tracks found in tracks/")
        st.stop()
    _default = current_track()
    if _default not in _tracks:
        _default = _tracks[0]
    st.selectbox("Track", _tracks, index=_tracks.index(_default), key="track")
    st.divider()

track = current_track()


# --- load data --------------------------------------------------------------

laps = _laps(track)
corpus = _corpus(track)
track_def = _track_def(track)
sec_t = _section_times(track)
sec_bounds_all = section_bounds(track_def)
corner_apex = {c["id"]: c for c in track_def["corners"]}
corner_order = [c["id"] for c in sorted(track_def["corners"], key=lambda c: c["start_m"])]


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
    # NOTE: an `obd_discrepancy_m` gate used to live here to reject "GPS-glitched"
    # laps. It was removed 2026-05-19 — it gated on the *magnitude* of the
    # OBD-vs-centerline distance divergence, which cannot distinguish a GPS glitch
    # from a genuinely shorter racing line (both shrink OBD distance below the
    # centerline section length). On corners with a systematic line-vs-centerline
    # offset (e.g. T6, median ~-6m) it rejected ~40% of laps including the real
    # fastest ones. `transit_reliable` (already required via eligible_section_keys)
    # gates on track_dist_offset — the actual glitch signature — so it is the
    # correct filter. Retired gate:
    #   a, b = section_range_bounds(track_def, range_corners[0], range_corners[-1])
    #   ref_obd_bound = max(8.0, 0.02 * (b - a))
    #   eligible = eligible[eligible["obd_discrepancy_m"].abs() <= ref_obd_bound]
    if eligible.empty:
        return None
    r = eligible.sort_values("section_time_s").iloc[0]
    return (r["session_id"], int(r["lap"]), float(r["section_time_s"]))


# --- sidebar: Corner section ------------------------------------------------

with st.sidebar:
    st.header("Corner")
    from_choice = pick_with_arrows("From", ["full lap"] + corner_order, "pick_from")
    if from_choice == "full lap":
        to_choice = "full lap"
    else:
        # To options are constrained to corners >= From. When From moves forward
        # past the stored To, pick_with_arrows resets To to the new From (single
        # corner); moving From backward keeps a still-valid To. Both intentional.
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
                         else f"{from_choice}-{to_choice}")
    is_single = (not is_full_lap) and len(range_corners) == 1
    is_range = (not is_full_lap) and len(range_corners) > 1
    range_sec_t = (
        pd.DataFrame(columns=["session_id", "lap", "section_time_s", "obd_discrepancy_m"])
        if is_full_lap else _range_section_times(track, from_choice, to_choice))

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

    st.divider()

    # --- sidebar: Lap section ----------------------------------------------

    st.header("Lap")
    show_only_clean = st.checkbox("Clean laps only", value=True, key="lap_only_clean")
    show_only_reliable = st.checkbox("Lap reliable only", value=False, key="lap_only_reliable")

    view = laps.copy()
    if show_only_clean:
        view = view[view["is_clean"].fillna(False).astype(bool)]
    if show_only_reliable:
        view = view[view["lap_reliable"].fillna(False).astype(bool)]
    if view.empty:
        st.warning("No laps match filters")
        st.stop()

    dates_avail = sorted(view["date"].dropna().unique(), reverse=True)  # newest first
    picked_date = pick_with_arrows("Date", dates_avail, "pick_date", format_func=str)

    times_in_date = sorted(view[view["date"] == picked_date]["session_id"].unique())
    picked_sid = pick_with_arrows("Time", times_in_date, "pick_time", format_func=session_hhmm)

    laps_in_session = sorted(view[view["session_id"] == picked_sid]["lap"].astype(int).tolist())

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

    picked_lap = pick_with_arrows("Lap", laps_in_session, "pick_lap", format_func=lap_label)

sid = picked_sid
lap = int(picked_lap)


# --- header -----------------------------------------------------------------

summ = lap_summary(corpus, sid, lap)
lap_row = view[(view["session_id"] == sid) & (view["lap"] == lap)].iloc[0]
ok_mark = "✓" if summ["lap_reliable"] else "✗"
st.subheader(
    f"{summ['date']} · {session_hhmm(sid)} · L{lap} · "
    f"{format_lap_time(lap_row['lap_time_s'])}  {ok_mark}"
)

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
            help=(_section_time_help(track, sid, lap, from_choice)
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
        if not entry_row.empty:
            ev = float(entry_row.iloc[0]["entry_speed_mph"])
            ed = (None if is_self or ref_entry is None
                  else f"{ev - float(ref_entry['entry_speed_mph']):+.1f} vs best")
            c2.metric(f"Entry speed ({from_choice})", f"{ev:.1f} mph", delta=ed)
        else:
            c2.metric(f"Entry speed ({from_choice})", "—")
        if not exit_row.empty:
            xv = float(exit_row.iloc[0]["exit_speed_mph"])
            xd = (None if is_self or ref_exit is None
                  else f"{xv - float(ref_exit['exit_speed_mph']):+.1f} vs best")
            c3.metric(f"Exit speed ({to_choice})", f"{xv:.1f} mph", delta=xd)
        else:
            c3.metric(f"Exit speed ({to_choice})", "—")


# --- channel envelope plot --------------------------------------------------

# Three stacked panels with shared x-axis (track_dist_m).
PANELS: list[tuple[str, str]] = [
    ("speed_mph", "Speed (mph)"),
    ("throttle_norm", "Throttle"),
    ("long_g", "Longitudinal G"),
]


def _prepare_lap_trace(sid_: str, lap_: int) -> tuple[pd.DataFrame, int]:
    raw = _samples(track, sid_, lap_)
    clean = _drop_gps_glitches(raw)
    dropped = len(raw) - len(clean)
    out = clean.sort_values("track_dist_m")
    out = _insert_gap_breaks(out, "track_dist_m", gap_threshold=30.0)
    return out, dropped

lap_s, n_dropped = _prepare_lap_trace(sid, lap)

# Best-lap reference trace. Skip if the selected lap IS the best (would just overlay).
best_lap_s = None
best_label = None
if best_ref is not None and (best_ref[0] != sid or best_ref[1] != lap):
    b_sid, b_lap, _ = best_ref
    best_lap_s, _ = _prepare_lap_trace(b_sid, b_lap)
    b_lap_time = float(
        laps[(laps["session_id"] == b_sid) & (laps["lap"] == b_lap)].iloc[0]["lap_time_s"]
    )
    best_label = f"Best: {session_hhmm(b_sid)} L{b_lap} ({format_lap_time(b_lap_time)})"

# Extra context on each side of the section so adjacent brake zones and corner
# transitions are visible (especially for tight complexes like T13-T15).
DISPLAY_BUFFER_M = 200.0

if not is_full_lap:
    display_a = max(0.0, section_lo - DISPLAY_BUFFER_M)
    display_b = section_hi + DISPLAY_BUFFER_M
    lap_s = lap_s[(lap_s["track_dist_m"] >= display_a)
                  & (lap_s["track_dist_m"] <= display_b)]
    if best_lap_s is not None:
        best_lap_s = best_lap_s[(best_lap_s["track_dist_m"] >= display_a)
                                & (best_lap_s["track_dist_m"] <= display_b)]

if n_dropped > 0:
    st.caption(f"⚠ Dropped {n_dropped} GPS-glitched samples on this lap "
               f"(track_dist_m walked backwards). OBD channels were clean; this only "
               f"affects how the lap-line plots against track_dist_m.")

# Selected-lap label: match the reference's units — section time in a section
# view, full lap time in a full-lap view — so the legend compares like with like.
if is_full_lap or section_t_val is None:
    selected_label = f"L{lap} ({format_lap_time(lap_row['lap_time_s'])})"
else:
    selected_label = f"L{lap} ({section_t_val:.2f}s)"

# Cumulative time-delta curve (t_selected - t_best, anchored at left edge of display).
# This is the load-bearing diagnostic per the visualizer design doc — where exactly
# you gained / lost time relative to the best lap, in seconds.
#
# The two laps are aligned on the fused distance axis (fused_axis.py), not raw
# track_dist_m. Raw track_dist_m jitters sample-to-sample; its derivative drives
# the Δt slope, so a few metres of GPS noise fabricated a phantom wobble (~0.1s
# on a clean lap, far more on a glitched one). The fused axis is OBD-smooth and
# monotonic, so the Δt slope tracks true speed-and-line. Note this does NOT flatten
# genuine sign changes: a slower lap on a shorter line really does gain time on
# that stretch, and the fused Δt still shows it.
# delta_y is the total time gap at each track position. delta_speed_y is the
# same gap with line-length removed — the laps aligned by distance travelled
# instead of track position. The band between the two is what the selected lap's
# racing line (shorter or longer than best's) is worth in seconds: it answers
# "am I down because I'm slow, or because my line is longer?"
delta_x = delta_y = delta_speed_y = None
if best_ref is not None and (best_ref[0] != sid or best_ref[1] != lap):
    b_sid, b_lap, _ = best_ref

    def _fused_lap(sid_: str, lap_: int):
        """(fused_distance, t, obd_distance) for one lap, time-sorted. fused is
        monotone, so np.interp maps track position -> elapsed time cleanly."""
        s = _drop_gps_glitches(_samples(track, sid_, lap_)).sort_values("t")
        if len(s) < 2:
            return None
        return (compute_fused_dist(s), s["t"].to_numpy(),
                s["dist_lap_m"].to_numpy())

    sel = _fused_lap(sid, lap)
    bst = _fused_lap(b_sid, b_lap)
    if sel is not None and bst is not None:
        f_sel, t_sel, o_sel = sel
        f_bst, t_bst, o_bst = bst
        if not is_full_lap:
            # Δt curve spans only the section (not the ±200m display buffer): its
            # endpoint then equals the section-time delta in the header. The buffer
            # belongs to the channel traces for visual context, not to this metric.
            grid_a, grid_b = section_lo, section_hi
        else:
            # Full lap: intersect both laps' fused ranges, anchor at lap start.
            grid_a = float(max(f_sel.min(), f_bst.min()))
            grid_b = float(min(f_sel.max(), f_bst.max()))

        if grid_b > grid_a:
            grid = np.arange(grid_a, grid_b + 0.5, 1.0)
            # Elapsed time each lap reached each track position, zeroed to the
            # section start.
            ta = np.interp(grid, f_sel, t_sel)
            tb = np.interp(grid, f_bst, t_bst)
            ta = ta - ta[0]
            tb = tb - tb[0]
            # OBD distance each lap had travelled to reach each track position.
            da = np.interp(grid, f_sel, o_sel)
            db = np.interp(grid, f_bst, o_bst)
            da = da - da[0]
            db = db - db[0]
            delta_x = grid
            delta_y = ta - tb
            # Speed-only Δt: gap against the best lap's time to travel the SAME
            # distance the selected lap travelled. delta_y - delta_speed_y is the
            # line-length contribution, priced at the best lap's pace.
            delta_speed_y = ta - np.interp(da, db, tb)

show_delta_panel = delta_x is not None
n_rows = len(PANELS) + (1 if show_delta_panel else 0)
fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04)

for i, (ch, label) in enumerate(PANELS, start=1):
    env_i = _envelope(track, ch)
    if not is_full_lap:
        env_i = env_i[(env_i["track_dist_m"] >= display_a)
                      & (env_i["track_dist_m"] <= display_b)]
    show_leg = (i == 1)  # only top panel contributes to legend
    fig.add_trace(go.Scatter(x=env_i["track_dist_m"], y=env_i["p90"], mode="lines",
                             line=dict(width=0), showlegend=False),
                  row=i, col=1)
    fig.add_trace(go.Scatter(x=env_i["track_dist_m"], y=env_i["p10"], mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(120,160,255,0.20)",
                             name="top-decile p10–p90", showlegend=show_leg),
                  row=i, col=1)
    fig.add_trace(go.Scatter(x=env_i["track_dist_m"], y=env_i["p50"], mode="lines",
                             line=dict(color="rgba(120,160,255,0.9)", width=2, dash="dot"),
                             name="top-decile median", showlegend=show_leg),
                  row=i, col=1)
    if best_lap_s is not None and not best_lap_s.empty and ch in best_lap_s.columns:
        fig.add_trace(go.Scatter(x=best_lap_s["track_dist_m"], y=best_lap_s[ch],
                                 mode="lines",
                                 line=dict(color="#2e8b57", width=2),
                                 name=best_label, showlegend=show_leg),
                      row=i, col=1)
    fig.add_trace(go.Scatter(x=lap_s["track_dist_m"], y=lap_s[ch], mode="lines",
                             line=dict(color="crimson", width=2.5),
                             name=selected_label, showlegend=show_leg),
                  row=i, col=1)
    fig.update_yaxes(title_text=label, row=i, col=1)

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

# Time-delta panel (the bottom row, if available). Two curves: the total time
# gap, and the speed-only gap with line-length removed. The shaded band between
# them is the line-length effect — explained in the caption under the chart.
if show_delta_panel:
    delta_row = n_rows
    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"), row=delta_row, col=1)
    # Speed-only drawn first so the total trace fills the band between the two.
    fig.add_trace(go.Scatter(
        x=delta_x, y=delta_speed_y, mode="lines",
        line=dict(color="gray", width=1.5, dash="dot"),
        name="Δt — speed only", showlegend=True,
    ), row=delta_row, col=1)
    fig.add_trace(go.Scatter(
        x=delta_x, y=delta_y, mode="lines",
        line=dict(color="#ff8c00", width=2.5),  # dark orange
        name="Δt — total", showlegend=True,
        fill="tonexty", fillcolor="rgba(255,140,0,0.13)",
    ), row=delta_row, col=1)
    fig.update_yaxes(title_text="Δ time vs best (s)  + = slower",
                     row=delta_row, col=1)

fig.update_xaxes(title_text="track_dist_m", row=n_rows, col=1)
fig.update_layout(
    height=720 + (180 if show_delta_panel else 0),
    margin=dict(l=20, r=20, t=30, b=30),
    legend=dict(orientation="h", y=1.05, x=0),
)
st.plotly_chart(fig, use_container_width=True)
if show_delta_panel:
    st.caption(
        "**Δt panel** — *total* (orange) is the time gap at each track position. "
        "*Speed only* (dotted) aligns the laps by distance travelled, so a "
        "racing-line-length difference no longer counts toward it. The shaded "
        "band is that line effect: where total runs **below** speed-only, the "
        "selected lap's **shorter line is buying back time** despite lower speed; "
        "where total runs above, its line is longer than the best lap's."
    )


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
    if best_ref is not None and (best_ref[0], best_ref[1]) != (sid, lap):
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
