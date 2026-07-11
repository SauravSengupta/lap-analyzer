"""Visualizer page: T8-T11 downshift comparison.

Splits laps by whether a downshift was made entering T8, then compares the
T8 -> T11 section. A Ridge-specific worked example; see docs/VISUALIZER.md
(track-specific pages) and docs/NEW-TRACK.md for how to write one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from shared import (current_track, drop_gps_glitches, format_lap_time, laps,
                    samples, session_hhmm, track_def)
from lap_analyzer.analysis import (classify_t8_section, derive_gear, gear_bands,
                                   load_centerline, section_times, span_time)

st.set_page_config(page_title="Ridge — T8-T11 Downshift", layout="wide")
st.title("Ridge — T8-T11 Downshift Comparison")

# Page reads st.session_state['track'] (set by the main app.py picker) via
# current_track(); falls back to LAP_ANALYZER_TRACK env var -> 'ridge' if the
# user came straight to this page without visiting the main page first.
track = current_track()

# This page is a Ridge-specific worked example, not core visualizer surface.
# Corner IDs (T8/T10/T11) and gear-band thresholds are tuned to Ridge data.
# When a different track is selected, render an explainer instead of crashing
# on missing corners or producing nonsense gear classifications.
if track != "ridge":
    st.info(
        "**This is a Ridge-only worked example.** It analyzes whether "
        "downshifting entering T8 at Ridge Motorsports Park pays off across "
        "the T8 → T11 section. The corner IDs and gear-band thresholds are "
        "Ridge-specific, so it can't run against the currently selected "
        f"track (`{track}`).\n\n"
        "Switch to Ridge from the sidebar Track picker on the main page to "
        "see this page in action. See `docs/NEW-TRACK.md` for how to write a "
        "similar analysis page tailored to your own track."
    )
    st.stop()

_td = track_def(track)
_corners = {c["id"]: c for c in _td["corners"]}
T8, T10, T11 = _corners["T8"], _corners["T10"], _corners["T11"]
SPAN_A = T8["start_m"]


@st.cache_data(show_spinner="deriving gear bands (one-time)")
def _bands(track: str) -> list[float]:
    return gear_bands(track).tolist()


bands = np.array(_bands(track))
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
def _classify_all(track: str, endpoint: float, bands_key: tuple) -> pd.DataFrame:
    b = np.array(bands_key)
    td = track_def(track)
    _centerline = load_centerline(track)
    rows = []
    for r in laps(track).itertuples(index=False):
        try:
            s = samples(track, r.session_id, int(r.lap))
        except Exception:
            continue
        res = classify_t8_section(s, td, b)
        rows.append({
            "session_id": r.session_id,
            "lap": int(r.lap),
            "date": r.date,
            "lap_pace_decile": r.lap_pace_decile,
            "is_clean": getattr(r, "is_clean", None),
            "lap_reliable": bool(getattr(r, "lap_reliable", False)),
            "label": res["label"],
            "excluded_reason": res["excluded_reason"],
            "downshift_dist_m": res["downshift_dist_m"],
            "t8_apex_gear": res["t8_apex_gear"],
            "section_time_s": (lambda r: r[0] if r else None)(
                span_time(s, SPAN_A, endpoint, _centerline)),
        })
    return pd.DataFrame(rows)


cls = _classify_all(track, endpoint_m, tuple(bands.tolist()))

# Drop GPS-unreliable laps. This page positions everything by track_dist_m, and
# a lap_reliable=False lap has track_dist_m off by tens of metres (or more) — its
# kinematic data is fine but can't be placed on the track. See README: filter
# spatial analysis by reliability.
n_glitched = int((~cls["lap_reliable"]).sum())
cls = cls[cls["lap_reliable"]].reset_index(drop=True)

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

if n_glitched:
    st.caption(f"{n_glitched} GPS-unreliable laps (lap_reliable=False) dropped before "
               "classification — unreliable track_dist_m makes them unsafe for a "
               "distance-based comparison.")

st.markdown("**Pace mix** — lap count per pace decile (0 = fastest). "
            "Check the two groups are pace-matched before trusting the comparison.")
pace_mix = (filt[filt["label"].isin(["downshift", "no-downshift"])]
            .pivot_table(index="label", columns="lap_pace_decile",
                         values="lap", aggfunc="count", fill_value=0))
st.dataframe(pace_mix, use_container_width=True)

if len(ds) < 5 or len(nd) < 5:
    st.warning("One or both groups have fewer than 5 laps after filtering — "
               "loosen the filters; bands below may be unreliable.")

# --- grouped channel envelopes ---------------------------------------------

DISPLAY_LO = SPAN_A - 200.0
DISPLAY_HI = endpoint_m + 200.0
DS_COLOR = "crimson"
ND_COLOR = "#2e8b57"


@st.cache_data(show_spinner="pooling group traces")
def _group_traces(track: str, keys: tuple, bands_key: tuple,
                  lo: float, hi: float) -> pd.DataFrame:
    """Pool samples for a set of (session_id, lap) keys across [lo, hi], with a
    derived `gear` column and GPS glitches removed."""
    b = np.array(bands_key)
    frames = []
    for sid, lap in keys:
        try:
            s = drop_gps_glitches(samples(track, sid, int(lap)))
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


def _median_time_curve(track: str, keys: tuple, grid: np.ndarray) -> np.ndarray | None:
    """Median (over laps) of cumulative time vs track_dist_m, anchored at grid[0]."""
    rows = []
    for sid, lap in keys:
        try:
            s = drop_gps_glitches(samples(track, sid, int(lap)))
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
ds_traces = _group_traces(track, ds_keys, tuple(bands.tolist()), DISPLAY_LO, DISPLAY_HI)
nd_traces = _group_traces(track, nd_keys, tuple(bands.tolist()), DISPLAY_LO, DISPLAY_HI)

PANELS = [("speed_mph", "Speed (mph)"),
          ("throttle_norm", "Throttle"),
          ("gear", "Gear (0=low)")]

# Time-delta: no-downshift minus downshift. Positive = no-downshift slower.
grid = np.arange(SPAN_A, endpoint_m + 0.5, 2.0)
ds_curve = _median_time_curve(track, ds_keys, grid) if ds_keys else None
nd_curve = _median_time_curve(track, nd_keys, grid) if nd_keys else None
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
def _per_corner_times(track: str) -> pd.DataFrame:
    return section_times(track, track_def(track))


per_corner = _per_corner_times(track)
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
            s = drop_gps_glitches(samples(track, sid, int(lap)))
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
