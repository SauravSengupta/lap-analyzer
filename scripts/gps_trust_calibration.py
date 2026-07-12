"""Per-mode σ-calibration battery for the trajectory layer (design R7 hard gate).

Synthesises randomised laps through a real corner on the real ridge corridor, one
per GPS failure mode, with a KNOWN ground-truth section time, and checks that the
emitted 1σ honestly covers the error: |section-time error| < 2·σ̂ in ≥95% of draws,
asserted SEPARATELY for each mode (a global scale can hide Mode-4 under-coverage
behind Mode-3 over-coverage — judge finding).

The ground truth is always the time for the car's TRUE track position `s_true` to
cross the gates. Each knob corrupts only what its failure mode corrupts:
- clean       : accurate GPS on the centerline (baseline).
- teleport    : a short along-track GPS spike (Mode 2) — Hampel must reject it.
- drift       : smooth OFF-ribbon lateral walk + opposite-signed along error at the
                gates (Mode 4) — the corridor must kill the evidence and revert to OBD.
- line_offset : a genuine IN-corridor, odometer-consistent tight line — the estimator
                must FOLLOW the real δ and report the true (fast) time (rankable).
- gps_hold    : frozen GPS for runs (Mode 3, coarse fixes).
- lockup      : odometer under-read near braking — accurate GPS must correct it.
- gps_only    : no OBD speed; backbone integrated from GPS (status gps_backbone).

Run (full R7 gate, 500 draws/mode):  python scripts/gps_trust_calibration.py
The pytest imports `run_calibration` for a lighter CI-enforced guard.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

from lap_analyzer.analysis import load_centerline, section_bounds
from lap_analyzer.gates import TrackFrame
from lap_analyzer.trajectory import (
    MPH_TO_MPS, estimate_trajectory, load_corridor, section_timing,
)

MODES = ["clean", "teleport", "drift", "line_offset", "gps_hold", "lockup", "gps_only"]
# corners spanning geometry: a long sweeper, a tight hairpin, a fast kink, a medium.
CORNERS = ["T8", "T14", "T10", "T5"]
WINDOW_M = 180.0            # context added on each side of the section
G = 9.81


class _Ctx:
    """Calibration context: a corridor + centerline field + the corners to draw from.
    Built either from a real track (the authoritative local battery) or from an
    analytic straight-arc-straight synthetic track (the self-contained CI guard)."""

    def __init__(self, corr, frame, cd, cx, cy, bounds, corners):
        self.corr, self.frame = corr, frame
        self.cd, self.cx, self.cy = cd, cx, cy
        self.bounds, self.corners = bounds, corners

    @classmethod
    def from_track(cls, track: str = "ridge"):
        corr = load_corridor(track)
        cl = load_centerline(track)
        frame = TrackFrame.from_centerline(cl)
        cx, cy = frame.to_xy(cl["lat"].to_numpy(), cl["long"].to_numpy())
        return cls(corr, frame, cl["track_dist_m"].to_numpy(), cx, cy,
                   section_bounds(json.load(open("tracks/ridge.json"))), CORNERS)

    @classmethod
    def synthetic(cls):
        """A self-contained straight-arc-straight track + corridor (no real data):
        analytic κ (=1/R in the arc), a ±12m envelope, one section over the arc."""
        from lap_analyzer.trajectory import CALIB_VERSION, Corridor
        ds, L, R = 2.0, 1500.0, 200.0
        s = np.arange(0.0, L, ds)
        kap = np.where((s >= 500.0) & (s < 800.0), 1.0 / R, 0.0)   # left arc
        theta = np.cumsum(kap) * ds
        x = np.cumsum(np.cos(theta)) * ds
        y = np.cumsum(np.sin(theta)) * ds
        lat0, lon0 = 45.0, -122.0
        from lap_analyzer.centerline import M_PER_DEG_LAT
        m_lat = M_PER_DEG_LAT
        m_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))
        cl = pd.DataFrame({"track_dist_m": s, "lat": lat0 + y / m_lat, "long": lon0 + x / m_lon})
        frame = TrackFrame.from_centerline(cl)
        cx, cy = frame.to_xy(cl["lat"].to_numpy(), cl["long"].to_numpy())
        sb = np.arange(0.0, L, 10.0)
        b_theta = np.interp(sb, s, theta)
        corr = Corridor(
            s_bin=sb, e_lo=np.full(len(sb), -12.0), e_hi=np.full(len(sb), 12.0),
            tx=np.cos(b_theta), ty=np.sin(b_theta),
            gx=np.interp(sb, s, cx), gy=np.interp(sb, s, cy),
            kappa_signed=np.interp(sb, s, kap),
            meta={"calib_version": CALIB_VERSION})
        return cls(corr, frame, s, cx, cy, {"SYN": (450.0, 850.0)}, ["SYN"])

    def xy_at(self, s):
        return np.interp(s, self.cd, self.cx), np.interp(s, self.cd, self.cy)

    def tangent_at(self, s):
        b = self.corr.bin_index(s)
        return self.corr.tx[b], self.corr.ty[b]

    def kappa_at(self, s):
        b = self.corr.bin_index(s)
        return self.corr.kappa_signed[b]

    def latlon_from_xy(self, x, y):
        lon = self.frame.lon0 + x / self.frame.m_per_deg_lon
        lat = self.frame.lat0 + y / self.frame.m_per_deg_lat
        return lat, lon


def _bump(u):  # 0 at u=0,1; 1 at u=0.5 (u in [0,1], else 0)
    return np.where((u >= 0) & (u <= 1), np.sin(np.clip(u, 0, 1) * math.pi) ** 2, 0.0)


def make_synth_lap(mode: str, rng: np.random.Generator, ctx: _Ctx):
    """Return (lap_df, a, b, truth_s) for one randomised draw of `mode`."""
    corner = ctx.corners[rng.integers(len(ctx.corners))]
    a, b = ctx.bounds[corner]
    w0, w1 = max(a - WINDOW_M, ctx.cd[0]), min(b + WINDOW_M, ctx.cd[-1])
    n = int(rng.integers(120, 200))
    s_true = np.linspace(w0, w1, n)

    vbar = rng.uniform(22.0, 42.0)
    v = vbar * (1.0 + 0.15 * np.sin(rng.uniform(0, 2 * math.pi) + s_true / rng.uniform(150, 400)))
    v = np.maximum(v, 9.0)
    dt = np.gradient(s_true) / v
    t = np.cumsum(dt) - dt[0]

    cx_s, cy_s = ctx.xy_at(s_true)
    tx_s, ty_s = ctx.tangent_at(s_true)
    nx, ny = -ty_s, tx_s                       # unit left normal
    kap = ctx.kappa_at(s_true)
    u = (s_true - a) / (b - a)

    # defaults: accurate GPS on the centerline, odometer == true track position
    e = np.zeros(n)                            # true lateral offset of the car (m)
    dl = s_true.copy()                         # odometer
    td = s_true.copy()                         # GPS along-track reading
    gps_x, gps_y = cx_s.copy(), cy_s.copy()    # GPS position
    has_obd = True
    truth = float(np.interp(b, s_true, t) - np.interp(a, s_true, t))

    if mode == "teleport":
        i0 = int(rng.integers(np.searchsorted(s_true, a), np.searchsorted(s_true, b) - 8))
        span = int(rng.integers(3, 9))
        mag = rng.uniform(50.0, 200.0) * rng.choice([-1.0, 1.0])
        td[i0:i0 + span] += mag
    elif mode == "drift":
        # A faithful Mode-4 excursion is OFF-ribbon across (most of) the section — the
        # T8 fake ran "18m outside p2 for 150m" of its 242m. Place the drift a clear
        # margin OUTSIDE the local envelope edge for the whole section (on-ribbon only
        # in the flanks), with an opposite-signed along-track error at the two gates.
        # The corridor must down-weight this whole-section evidence so δ̂ bridges from
        # the on-ribbon flanks back to the OBD backbone.
        sign = rng.choice([-1.0, 1.0])
        bidx = ctx.corr.bin_index(s_true)
        edge = np.where(sign < 0, ctx.corr.e_lo[bidx], ctx.corr.e_hi[bidx])
        margin = rng.uniform(12.0, 22.0)
        in_sec = (s_true >= a) & (s_true <= b)
        e = np.where(in_sec, edge + sign * margin, 0.0)     # off-ribbon across the section
        along = np.where(in_sec, rng.uniform(6.0, 16.0) * np.cos(np.pi * np.clip(u, 0, 1)), 0.0)
        td = s_true + along                     # opposite-signed at the two gates
        gps_x = cx_s + e * nx
        gps_y = cy_s + e * ny
    elif mode == "line_offset":
        amp = -rng.uniform(4.0, 11.0)           # IN-corridor tight line (inside)
        e = amp * _bump(u)
        short = np.concatenate([[0.0], np.cumsum(kap[1:] * e[1:] * np.diff(s_true))])
        dl = s_true + short                     # odometer-consistent shortening (kap*e<0)
        gps_x = cx_s + e * nx                   # accurate GPS of the real (offset) car
        gps_y = cy_s + e * ny
        td = s_true.copy()                       # GPS projects back to the centerline
    elif mode == "gps_hold":
        hold = int(rng.integers(6, 14))
        held = (np.arange(n) // hold) * hold
        td = s_true[held]
        gps_x, gps_y = cx_s[held], cy_s[held]
    elif mode == "lockup":
        i0 = int(rng.integers(np.searchsorted(s_true, a), np.searchsorted(s_true, b) - 10))
        span = int(rng.integers(8, 20))
        mag = rng.uniform(6.0, 20.0)
        lb = np.zeros(n)
        lb[i0:] = np.linspace(0, mag, n - i0)   # odometer permanently under-reads after
        dl = s_true - lb                         # GPS (td=s_true) stays accurate
    elif mode == "gps_only":
        has_obd = False

    lat, lon = ctx.latlon_from_xy(gps_x, gps_y)
    lat_g = kap * v ** 2 / G                    # consistent with κ (positive=right)
    spd = v / MPH_TO_MPS
    lap = pd.DataFrame(dict(
        t=t, lap=1, dist_lap_m=dl, track_dist_m=td, lat=lat, long=lon, lat_g=lat_g,
        speed_mph=(np.full(n, np.nan) if not has_obd else spd),
        speed_mph_gps=spd, gps_drift_lat_m=np.zeros(n), gps_drift_lon_m=np.zeros(n)))
    return lap, a, b, truth


def run_calibration(n_per_mode: int = 500, seed: int = 12345, ctx: _Ctx | None = None):
    """Return {mode: dict(n, coverage, under, mean_abs_err, median_sigma)}."""
    ctx = ctx or _Ctx.from_track()
    rng = np.random.default_rng(seed)
    out = {}
    for mode in MODES:
        errs, sigs, within = [], [], 0
        n_valid = 0
        for _ in range(n_per_mode):
            lap, a, b, truth = make_synth_lap(mode, rng, ctx)
            traj = estimate_trajectory(lap, ctx.corr, ctx.frame)
            st = section_timing(traj, a, b, ctx.corr)
            if st.status == "no_coverage" or not np.isfinite(st.time_s) or not np.isfinite(st.sigma_s):
                continue
            n_valid += 1
            err = abs(st.time_s - truth)
            errs.append(err)
            sigs.append(st.sigma_s)
            if err < 2.0 * max(st.sigma_s, 1e-9):
                within += 1
        cov = within / n_valid if n_valid else float("nan")
        out[mode] = dict(n=n_valid, coverage=cov, mean_abs_err=float(np.mean(errs)) if errs else float("nan"),
                         median_sigma=float(np.median(sigs)) if sigs else float("nan"))
    return out


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(f"=== Per-mode sigma calibration (R7): {n} draws/mode, |err| < 2*sigma target >=95% ===")
    res = run_calibration(n_per_mode=n)
    ok = True
    print(f"{'mode':12} {'n':>5} {'coverage':>9} {'mean|err|s':>11} {'med_sigma_s':>12}  gate")
    for mode in MODES:
        r = res[mode]
        passed = r["coverage"] >= 0.95
        ok &= passed
        print(f"{mode:12} {r['n']:>5} {r['coverage']:>9.3f} {r['mean_abs_err']:>11.4f} "
              f"{r['median_sigma']:>12.4f}  {'PASS' if passed else 'FAIL'}")
    print(f"\n{'PASS' if ok else 'FAIL'}: all modes >=95% within 2 sigma")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
