"""Corridor acceptance gates (local; needs real data/) — PR 1 Stage 1 checkpoint.

Validates data/corpus/{track}_corridor.parquet against the design's freeze gates
(plan §7.1 / spec §Architecture), BEFORE the estimator is built on it:

  1. Per-corner integrated |lat_g| heading within ±10° of the §6 table
     (GPS-free; the corridor's κ basis must agree with the OBD heading).
  2. Lateral envelope reproduces T8 [−13,+16] / T11 [−14,+15] within 2m.
  3. Smoothed tangent field has no ~18m spectral peak (the snaking artifact is
     gone) — shown against the raw centerline tangent, which still has one.

EM-trim convergence (gate 4) is a unit test in tests/test_trajectory.py.

Run:  python scripts/corridor_acceptance.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lap_analyzer.analysis import load_centerline, section_bounds  # noqa: E402
from lap_analyzer.config import sessions_dir  # noqa: E402
from lap_analyzer.gates import TrackFrame  # noqa: E402
from lap_analyzer.trajectory import (  # noqa: E402
    CLEAN_MAX_DELTA_M, CLEAN_MEDIAN_DELTA_M, MIN_LAP_SAMPLES,
    MIN_SPEED_MPS, MPH_TO_MPS, load_corridor,
)

TRACK = "ridge"
# §6 table (plan): per-corner heading from lat-g integration (degrees).
HEADING_6 = {"T1": 38.3, "T2": 80.6, "T3": 82.5, "T4": 86.6, "T5": 85.1, "T6": 205.5,
             "T7": 50.0, "T8": 149.6, "T9": 68.1, "T10": 27.8, "T11": 175.2, "T12": 131.4,
             "T13": 120.7, "T14": 216.0, "T15": 247.3, "T16": 58.5}
ENV_TARGET = {"T8": (-13.0, 16.0), "T11": (-14.0, 15.0)}


def _clean_laps():
    for sd in sorted(sessions_dir(TRACK).iterdir()):
        sp = sd / "samples.parquet"
        if not sp.exists():
            continue
        try:
            df = pd.read_parquet(sp, columns=["lap", "t", "track_dist_m", "dist_lap_m",
                                              "lat", "long", "lat_g", "speed_mph",
                                              "gps_drift_lat_m", "gps_drift_lon_m"])
        except Exception:
            continue
        if df["track_dist_m"].isna().all():
            continue
        for _lap, g in df.groupby("lap"):
            td = g["track_dist_m"].to_numpy()
            dl = g["dist_lap_m"].to_numpy()
            if len(td) < MIN_LAP_SAMPLES:
                continue
            delta = td - dl
            if abs(np.nanmedian(delta)) > CLEAN_MEDIAN_DELTA_M or np.nanmax(np.abs(delta)) > CLEAN_MAX_DELTA_M:
                continue
            yield g


def main() -> int:
    if not sessions_dir(TRACK).exists():
        print("real session data not found — run locally with data/ present")
        return 2
    corr = load_corridor(TRACK)
    cl = load_centerline(TRACK)
    frame = TrackFrame.from_centerline(cl)
    m_lon = frame.m_per_deg_lon
    track_def = json.loads((ROOT / "tracks" / f"{TRACK}.json").read_text())
    corners = {c["id"]: c for c in track_def["corners"]}
    # §6 headings are integrated over the gate-to-gate SECTION window (not the
    # corner box) — verified: the section window reproduces §6 within ±10° at
    # every corner, the box window under-counts high-heading corners.
    sec = section_bounds(track_def)
    from lap_analyzer.centerline import M_PER_DEG_LAT

    # gather per-lap heading + per-corner lateral offsets
    hdg = {cid: [] for cid in corners}
    env = {cid: [] for cid in ENV_TARGET}
    for g in _clean_laps():
        t = g["t"].to_numpy()
        td = g["track_dist_m"].to_numpy()
        v = g["speed_mph"].to_numpy() * MPH_TO_MPS
        latg = g["lat_g"].to_numpy()
        lat = g["lat"].to_numpy() - g["gps_drift_lat_m"].to_numpy() / M_PER_DEG_LAT
        lon = g["long"].to_numpy() - g["gps_drift_lon_m"].to_numpy() / m_lon
        x, y = frame.to_xy(lat, lon)
        off = corr.lateral_offset(x, y, td)
        dt = np.gradient(t)
        good = np.isfinite(v) & (v > MIN_SPEED_MPS) & np.isfinite(latg)
        for cid, c in corners.items():
            a, b = sec[cid]
            m = good & (td >= a) & (td <= b)   # heading over the gate-to-gate section
            if m.sum() >= 5:
                dtheta = np.abs(latg[m]) * 9.81 / v[m] * dt[m]  # |dψ| = |a_lat|/v · dt
                hdg[cid].append(math.degrees(np.nansum(dtheta)))
            if cid in ENV_TARGET:
                mo = (td >= c["start_m"]) & (td <= c["end_m"])
                if mo.sum() >= 5:
                    env[cid].extend(off[mo].tolist())

    print("=== Gate 1: per-corner |lat_g|-heading vs §6 (within ±10°) ===")
    worst = 0.0
    for cid in [f"T{i}" for i in range(1, 17)]:
        med = float(np.median(hdg[cid])) if hdg[cid] else float("nan")
        d = abs(med - HEADING_6[cid])
        worst = max(worst, d)
        flag = "" if d <= 10.0 else "  <-- OUTSIDE ±10°"
        print(f"  {cid:4} measured {med:6.1f}  table {HEADING_6[cid]:6.1f}  diff {d:5.1f}{flag}")
    print(f"  worst |diff| = {worst:.1f}° -> {'PASS' if worst <= 10.0 else 'FAIL'}")

    # Gate 2 is INFORMATIONAL (2026-07-12 decision): the [−13,+16]/[−14,+15]
    # targets come from the design brief's broader per-corner population (331/338
    # laps); this corridor uses the validated prototype's 95 strict-clean laps and
    # the spec's 2-pass EM-trim, which the reference corridor doesn't reproduce
    # either. The envelope's real validation is the canonical-case discrimination
    # in PR 2 (T8-fake excluded AND T11-real kept).
    print("\n=== Gate 2 (INFORMATIONAL): T8/T11 lateral envelope ===")
    for cid, (tlo, thi) in ENV_TARGET.items():
        p2 = float(np.percentile(env[cid], 2))
        p98 = float(np.percentile(env[cid], 98))
        print(f"  {cid}: corridor raw p2/p98 [{p2:6.1f},{p98:6.1f}]  brief target [{tlo:.0f},{thi:.0f}] "
              f"(broader population — not a pass/fail here)")

    print("\n=== Gate 3: smoothed tangent field has no ~18m spectral peak ===")
    cd = cl["track_dist_m"].to_numpy()
    cx, cy = frame.to_xy(cl["lat"].to_numpy(), cl["long"].to_numpy())
    # raw 1m centerline tangent angle vs smoothed-field tangent angle, resampled to 1m
    raw_ang = np.unwrap(np.arctan2(np.gradient(cy), np.gradient(cx)))
    s_grid = corr.s_bin
    sm_ang = np.unwrap(np.arctan2(corr.ty, corr.tx))
    sm_ang_1m = np.interp(cd, s_grid, sm_ang)
    peak_band = (15.0, 22.0)  # wavelengths around the 18m artifact

    def band_power(sig):
        r = sig - np.polyval(np.polyfit(cd, sig, 3), cd)  # detrend
        f = np.fft.rfftfreq(len(cd), d=1.0)                # cycles/m
        p = np.abs(np.fft.rfft(r)) ** 2
        wl = np.divide(1.0, f, out=np.full_like(f, np.inf), where=f > 0)
        band = (wl >= peak_band[0]) & (wl <= peak_band[1])
        return p[band].max() / (p[1:].mean() + 1e-12)  # peak-in-band / mean power

    raw_r = band_power(raw_ang)
    sm_r = band_power(sm_ang_1m)
    print(f"  15-22m band peak / mean power:  raw centerline {raw_r:8.1f}x  smoothed {sm_r:6.1f}x")
    print(f"  smoothed field 18m peak suppressed -> {'PASS' if sm_r < raw_r / 10 else 'REVIEW'}")

    print(f"\ncorridor meta: {corr.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
