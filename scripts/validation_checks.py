"""Independent trust-validation checks (W0) for the GPS-trust layer.

Every check scores the pipeline against something the pipeline did not produce:
V1 an Accel-Z road-profile ruler (PIR), V2 planted faults on clean laps, V3 a
GPS-only emulation on OBD laps, V4 an odometer-only section time. Reads data/ only;
writes nothing. Spec: docs/superpowers/specs/2026-10-10-sensor-fusion-rethink-design.md §W0.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lap_analyzer.analysis import (
    _load_or_build_corridor,
    _read_traj_samples,
    load_centerline,
    section_bounds,
)
from lap_analyzer.config import sessions_dir, tracks_dir
from lap_analyzer.gates import TrackFrame
from lap_analyzer.trajectory import MPH_TO_MPS

CAL_FRACTION = 0.6          # 2026-10-10, spec §W0 calibration share per (track, has_obd) stratum
MIN_FIX_RATE_HZ = 3.0       # 2026-10-10, below this a session is "1 Hz" (4 Ridge sessions at ~1 Hz, rest ~6-8 Hz fresh)
MIN_LAP_ROWS = 200


def _hash(session_id: str) -> int:
    return int(hashlib.sha1(session_id.encode()).hexdigest(), 16)


def session_split(sessions: pd.DataFrame, cal_fraction: float = CAL_FRACTION) -> dict:
    """Deterministic calibration/held-out split, stratified by (track, has_obd).

    Within each stratum sessions are ordered by sha1(session_id); the first
    round(cal_fraction·n) are 'cal', the rest 'held'. Order of the input is irrelevant."""
    out: dict[tuple[str, str], str] = {}
    for _, grp in sessions.groupby(["track", "has_obd"]):
        ids = sorted(grp["session_id"], key=_hash)
        n_cal = int(round(cal_fraction * len(ids)))
        track = grp["track"].iloc[0]
        for i, sid in enumerate(ids):
            out[(track, sid)] = "cal" if i < n_cal else "held"
    return out


def fix_rows(lat, lon) -> np.ndarray:
    """Row indices where lat or lon changed from the previous row (first row counts)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ch = np.ones(len(lat), dtype=bool)
    ch[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
    return np.where(ch & np.isfinite(lat) & np.isfinite(lon))[0]


def list_sessions(track: str) -> pd.DataFrame:
    """Every real session of a track: has_obd from meta.json, fresh-fix rate from samples."""
    rows = []
    for sd in sorted(p for p in sessions_dir(track).iterdir() if p.is_dir() and not p.name.startswith("_")):
        sp, mp = sd / "samples.parquet", sd / "meta.json"
        if not (sp.exists() and mp.exists()):
            continue
        meta = json.loads(mp.read_text())
        s = pd.read_parquet(sp, columns=["t", "lat", "long"])
        dur = float(s["t"].iloc[-1] - s["t"].iloc[0]) if len(s) > 1 else 0.0
        rate = len(fix_rows(s["lat"], s["long"])) / dur if dur > 0 else 0.0
        rows.append(dict(track=track, session_id=sd.name, has_obd=bool(meta.get("has_obd", True)),
                         fix_rate_hz=rate))
    return pd.DataFrame(rows, columns=["track", "session_id", "has_obd", "fix_rate_hz"])


def odometer_section_time(g: pd.DataFrame, a: float, b: float) -> float:
    """GPS-free section time: when the (monotone) odometer dist_lap_m crosses a and b."""
    t = g["t"].to_numpy(dtype=float)
    dl = np.maximum.accumulate(g["dist_lap_m"].to_numpy(dtype=float))
    if not (dl[0] <= a < b <= dl[-1]):
        return float("nan")
    return float(np.interp(b, dl, t) - np.interp(a, dl, t))


def gps_only_emulation(g: pd.DataFrame) -> pd.DataFrame:
    """The same lap as normalize would ingest it without OBD: speed_mph all-NaN and
    dist_lap_m re-integrated from (lag-corrected) speed_mph_gps, rescaled to the lap's
    own dist_lap_m length (normalize rescales every lap to the track length)."""
    h = g.copy()
    h["speed_mph"] = np.nan
    t = h["t"].to_numpy(dtype=float)
    v = np.nan_to_num(h["speed_mph_gps"].to_numpy(dtype=float)) * MPH_TO_MPS
    d = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])
    length = float(g["dist_lap_m"].max() - g["dist_lap_m"].min())
    h["dist_lap_m"] = d * (length / d[-1]) if d[-1] > 0 else d
    return h


@dataclass
class TrackCtx:
    track: str
    corridor: object
    frame: TrackFrame
    bounds: dict

    @classmethod
    def load(cls, track: str) -> TrackCtx:
        tdef = json.loads((tracks_dir() / f"{track}.json").read_text())
        return cls(track=track, corridor=_load_or_build_corridor(track),
                   frame=TrackFrame.from_centerline(load_centerline(track)),
                   bounds=section_bounds(tdef))


def clean_laps(ctx: TrackCtx, session_ids: list[str], max_laps: int | None = None
               ) -> Iterator[tuple[str, int, pd.DataFrame]]:
    """(session_id, lap, time-sorted lap samples) for OBD laps at ≥ MIN_FIX_RATE_HZ whose
    corners.parquet marks them lap_reliable_traj. Deterministic order; stops at max_laps."""
    n = 0
    for sid in sorted(session_ids):
        sd = sessions_dir(ctx.track) / sid
        cp = sd / "corners.parquet"
        if not cp.exists():
            continue
        c = pd.read_parquet(cp)
        if "lap_reliable_traj" not in c or not c["obd_present"].any():
            continue
        good = sorted(set(c.loc[c["lap_reliable_traj"].fillna(False).astype(bool) & c["obd_present"], "lap"]))
        s = _read_traj_samples(sd / "samples.parquet")
        if s is None:
            continue
        for lap in good:
            g = s[s["lap"] == lap].sort_values("t", kind="stable").reset_index(drop=True)
            if len(g) < MIN_LAP_ROWS:
                continue
            dur = float(g["t"].iloc[-1] - g["t"].iloc[0])
            if dur <= 0 or len(fix_rows(g["lat"], g["long"])) / dur < MIN_FIX_RATE_HZ:
                continue
            yield sid, int(lap), g
            n += 1
            if max_laps is not None and n >= max_laps:
                return


# --- V2 fault planting ------------------------------------------------------

from lap_analyzer.centerline import M_PER_DEG_LAT  # noqa: E402


def plant_offset(g: pd.DataFrame, rows: np.ndarray, dx_m: np.ndarray, dy_m: np.ndarray,
                 frame: TrackFrame, ref) -> pd.DataFrame:
    """Copy of g with the `rows` positions moved by (dx_m, dy_m) metres (east, north) and
    their track_dist_m re-projected through `ref.lookup` on drift-corrected coordinates,
    exactly as the labeler would have projected those fixes."""
    h = g.copy()
    rows = np.asarray(rows, dtype=bool)
    lat = h["lat"].to_numpy(dtype=float).copy()
    lon = h["long"].to_numpy(dtype=float).copy()
    lat[rows] += np.asarray(dy_m, float)[rows] / M_PER_DEG_LAT
    lon[rows] += np.asarray(dx_m, float)[rows] / frame.m_per_deg_lon
    dlat = h["gps_drift_lat_m"].to_numpy(dtype=float) / M_PER_DEG_LAT
    dlon = h["gps_drift_lon_m"].to_numpy(dtype=float) / frame.m_per_deg_lon
    td = h["track_dist_m"].to_numpy(dtype=float).copy()
    if rows.any():
        td[rows] = ref.lookup(lat[rows] - dlat[rows], lon[rows] - dlon[rows])[0]
    h["lat"], h["long"], h["track_dist_m"] = lat, lon, td
    return h


def burst_rows(g: pd.DataFrame, start_fix: int, n_fix: int) -> np.ndarray:
    """Bool mask of every row holding fixes start_fix .. start_fix+n_fix-1 (a fix's rows
    run until the next fresh fix)."""
    fr = fix_rows(g["lat"], g["long"])
    ends = np.append(fr[1:], len(g))
    m = np.zeros(len(g), dtype=bool)
    for j in range(start_fix, min(start_fix + n_fix, len(fr))):
        m[fr[j]:ends[j]] = True
    return m


def slow_drift_offsets(g: pd.DataFrame, center_m: float, amp_m: float, corridor,
                       half_len_m: float = 75.0):
    """A T8-style slow lateral excursion: amp_m·sin²(π·u) along the corridor's left normal
    for rows within ±half_len_m of center_m (u = 0..1 across that window). Returns
    (rows mask, dx_m, dy_m) arrays aligned to g."""
    s = g["track_dist_m"].to_numpy(dtype=float)
    u = (s - (center_m - half_len_m)) / (2.0 * half_len_m)
    rows = np.isfinite(u) & (u >= 0.0) & (u <= 1.0)
    bump = np.where(rows, np.sin(np.pi * np.clip(u, 0.0, 1.0)) ** 2, 0.0) * amp_m
    b = corridor.bin_index(s)
    nx, ny = -corridor.ty[b], corridor.tx[b]            # unit left normal of travel
    return rows, bump * nx, bump * ny
