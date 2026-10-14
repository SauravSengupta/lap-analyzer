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


# --- V1 Accel-Z road-profile ruler -------------------------------------------

from lap_analyzer.config import APP_ROOT  # noqa: E402

Z_HIGHPASS_M = 25           # 2026-10-01, running-mean high-pass on the 1 m grid (prototype m2b_zreg)
V1_PLANT_M = 7.0            # 2026-10-01, self-test shift; prototype recovered 7.0 m in 4/4 sessions
V1_PLANT_TOL_M = 1.0        # 2026-10-10, spec §W0 V1: planted shift must come back within ±1 m


def register_profiles(P: np.ndarray, win: int = 120, max_shift: int = 25, step: int = 40,
                      min_corr: float = 0.6, plant_m: float = 0.0) -> pd.DataFrame:
    """Register each lap's profile against the leave-one-out median of the others, in
    `win`-metre windows every `step` m, over integer shifts ±max_shift. Returns one row
    per (window, lap) with the best correlation and shift (metres; + = the lap's profile
    sits later along track). `plant_m` shifts the lap under test only (self-test: the
    recovered shift should equal plant_m). Only rows with corr > min_corr are returned."""
    P = np.asarray(P, dtype=float)
    idx = np.arange(P.shape[1])
    out = []
    for c0 in range(win // 2, P.shape[1] - win // 2, step):
        a, b = c0 - win // 2, c0 + win // 2
        for i in range(len(P)):
            ref = np.nanmedian(np.delete(P, i, axis=0), axis=0)[a:b]
            if np.isnan(ref).any() or ref.std() < 0.03:
                continue
            x0 = (np.interp(idx - plant_m, idx, P[i], left=np.nan, right=np.nan)
                  if plant_m else P[i])
            best = (-1.0, 0)
            for sh in range(-max_shift, max_shift + 1):
                if a + sh < 0 or b + sh > P.shape[1]:
                    continue
                x = x0[a + sh:b + sh]
                if np.isnan(x).any() or x.std() == 0:
                    continue
                cc = float(np.corrcoef(x, ref)[0, 1])
                if cc > best[0]:
                    best = (cc, sh)
            out.append((c0, i, best[0], best[1]))
    R = pd.DataFrame(out, columns=["win", "lap", "corr", "shift"])
    return R[R["corr"] > min_corr].reset_index(drop=True)


def zprofiles(track: str, session_id: str, laps: list[int], grid_len_m: float) -> np.ndarray | None:
    """High-passed Accel-Z per lap on a 1 m track_dist_m grid, using only rows where the
    raw Accel Z value changed (fresh samples). None if the raw CSV is missing or its row
    count does not match samples.parquet (rows are joined positionally)."""
    raw = sorted((APP_ROOT / "data" / "raw" / track).glob(f"Log-{session_id}*.csv"))
    if not raw:
        return None
    r = pd.read_csv(raw[0], comment="#", usecols=["Accel Z"])
    s = pd.read_parquet(sessions_dir(track) / session_id / "samples.parquet",
                        columns=["t", "lap", "track_dist_m"])
    if len(r) != len(s):
        return None
    s["az"] = r["Accel Z"].to_numpy(dtype=float)
    s["fresh"] = r["Accel Z"].diff().ne(0).to_numpy()
    grid = np.arange(0.0, grid_len_m, 1.0)
    P = []
    for lap in laps:
        g = s[(s["lap"] == lap) & s["fresh"] & s["track_dist_m"].notna()].sort_values("t")
        sd = g["track_dist_m"].to_numpy()
        ok = np.concatenate([[True], np.diff(sd) > 0])
        if ok.sum() < 2:
            P.append(np.full(len(grid), np.nan))
            continue
        z = pd.Series(np.interp(grid, sd[ok], g["az"].to_numpy()[ok], left=np.nan, right=np.nan))
        P.append((z - z.rolling(Z_HIGHPASS_M, center=True, min_periods=5).mean()).to_numpy())
    return np.array(P)


def run_v1(track: str, session_ids: list[str], grid_len_m: float = 4200.0) -> pd.DataFrame:
    """Per session: cross-lap shift std at well-registered windows + the planted self-test."""
    rows = []
    for sid in sorted(session_ids):
        cp = sessions_dir(track) / sid / "corners.parquet"
        if not cp.exists():
            continue
        c = pd.read_parquet(cp)
        laps = sorted(set(c.loc[c["lap_reliable_traj"].fillna(False).astype(bool), "lap"]))
        if len(laps) < 4:
            continue
        P = zprofiles(track, sid, laps, grid_len_m)
        if P is None:
            continue
        R = register_profiles(P)
        w = R.groupby("win")["shift"].agg(["size", "std"])
        w = w[w["size"] >= max(4, len(laps) // 2)]
        Rp = register_profiles(P, plant_m=V1_PLANT_M)
        rec = float(Rp["shift"].median()) if len(Rp) else float("nan")
        rows.append(dict(session_id=sid, n_laps=len(laps), n_windows=len(w),
                         shift_std_median_m=float(w["std"].median()) if len(w) else float("nan"),
                         planted_recovered_m=rec,
                         valid=bool(np.isfinite(rec) and abs(rec - V1_PLANT_M) <= V1_PLANT_TOL_M)))
    return pd.DataFrame(rows)


# --- V2 / V3 / V4 runners ------------------------------------------------------

from lap_analyzer.trajectory import estimate_trajectory, section_timing  # noqa: E402

V2_KINDS = ("spike", "burst", "slow_drift")
V2_MAGS_M = (5.0, 10.0, 15.0, 20.0, 30.0)
V2_GATE_WINDOW_M = 25.0     # 2026-10-10, gate placement: first faulted fix within ±25 m of a gate (jd_plant_gate)
V2_BAD_ERR_S = 0.05         # 2026-10-10, spec §W0 V2: "bad-but-rankable" = |error| > 0.05 s and tier A
V2_DRIFT_AMPS_M = (20.0, 30.0, 40.0)   # 2026-10-10, spec §W0 V2 slow drift 20-40 m over ~150 m


def _timing(g, ctx, a, b):
    return section_timing(estimate_trajectory(g, ctx.corridor, ctx.frame), a, b, ctx.corridor)


def run_v2(ctx, laps, ref, rng: np.random.Generator, trials_per_lap: int = 6) -> pd.DataFrame:
    """Plant one fault per trial on a clean lap inside a section that is rankable on the
    unfaulted lap; truth = that lap's unfaulted section time."""
    out = []
    cids = list(ctx.bounds)
    for sid, lap, g in laps:
        base = estimate_trajectory(g, ctx.corridor, ctx.frame)
        fr = fix_rows(g["lat"], g["long"])
        s_fix = g["track_dist_m"].to_numpy()[fr]
        for _ in range(trials_per_lap):
            cid = cids[int(rng.integers(len(cids)))]
            a, b = ctx.bounds[cid]
            st0 = section_timing(base, a, b, ctx.corridor)
            if not st0.rank_eligible:
                continue
            kind = V2_KINDS[int(rng.integers(len(V2_KINDS)))]
            where = "gate" if rng.random() < 0.5 else "mid"
            if kind == "slow_drift":
                mag = float(V2_DRIFT_AMPS_M[int(rng.integers(len(V2_DRIFT_AMPS_M)))])
                center = 0.5 * (a + b) if where == "mid" else (a if rng.random() < 0.5 else b)
                rows, dx, dy = slow_drift_offsets(g, center, mag * (1 if rng.random() < 0.5 else -1),
                                                  ctx.corridor)
                n_fix = int(np.isin(fr, np.where(rows)[0]).sum())
            else:
                mag = float(V2_MAGS_M[int(rng.integers(len(V2_MAGS_M)))])
                n_fix = 1 if kind == "spike" else int(rng.integers(3, 17))   # 0.4–2 s at 8 Hz
                if where == "gate":
                    gate = a if rng.random() < 0.5 else b
                    cand = np.where(np.abs(s_fix - gate) < V2_GATE_WINDOW_M)[0]
                else:
                    cand = np.where(np.abs(s_fix - 0.5 * (a + b)) < 0.3 * (b - a))[0]
                cand = cand[(cand >= 3) & (cand + n_fix < len(fr) - 3)]
                if not len(cand):
                    continue
                rows = burst_rows(g, int(cand[int(rng.integers(len(cand)))]), n_fix)
                ang = rng.uniform(0.0, 2.0 * np.pi)
                dx = np.full(len(g), mag * np.cos(ang))
                dy = np.full(len(g), mag * np.sin(ang))
            h = plant_offset(g, rows, dx, dy, ctx.frame, ref)
            traj = estimate_trajectory(h, ctx.corridor, ctx.frame)
            st = section_timing(traj, a, b, ctx.corridor)
            caught = (float(not traj.evidence[rows].any()) if kind != "slow_drift" else float("nan"))
            out.append(dict(track=ctx.track, session_id=sid, lap=lap, corner_id=cid, kind=kind,
                            where=where, mag_m=mag, n_fix=n_fix, err_s=st.time_s - st0.time_s,
                            rank=bool(st.rank_eligible), caught=caught))
    return pd.DataFrame(out, columns=["track", "session_id", "lap", "corner_id", "kind", "where",
                                      "mag_m", "n_fix", "err_s", "rank", "caught"])


def run_v3(ctx, laps) -> pd.DataFrame:
    out = []
    for sid, lap, g in laps:
        t_o = estimate_trajectory(g, ctx.corridor, ctx.frame)
        t_g = estimate_trajectory(gps_only_emulation(g), ctx.corridor, ctx.frame)
        for cid, (a, b) in ctx.bounds.items():
            so = section_timing(t_o, a, b, ctx.corridor)
            sg = section_timing(t_g, a, b, ctx.corridor)
            out.append(dict(track=ctx.track, session_id=sid, lap=lap, corner_id=cid,
                            t_obd=so.time_s, rank_obd=bool(so.rank_eligible),
                            t_gps=sg.time_s, sigma_gps=sg.sigma_s, rank_gps=bool(sg.rank_eligible)))
    return pd.DataFrame(out)


def run_v4(ctx, laps) -> pd.DataFrame:
    out = []
    for sid, lap, g in laps:
        traj = estimate_trajectory(g, ctx.corridor, ctx.frame)
        for cid, (a, b) in ctx.bounds.items():
            st = section_timing(traj, a, b, ctx.corridor)
            out.append(dict(track=ctx.track, session_id=sid, lap=lap, corner_id=cid,
                            t_prod=st.time_s, rank=bool(st.rank_eligible),
                            t_odo=odometer_section_time(g, a, b)))
    return pd.DataFrame(out)
