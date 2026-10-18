"""Short-horizon GPS consistency: does each GPS fix agree with the car's own motion?

The path SHAPE over ±JUMP_WIN_S comes from GPS-independent sensors: distance from the
odometer speed, heading from the phone accelerometer's lateral g (yaw rate =
−lat_g·g/v; canonical lat_g is + = right turn, frame angles are CCW). That shape is
accurate to ~1 m over ~1 s but drifts 6–7 m by 5 s (no gyro is logged; lat_g carries
bank/roll — Ridge per-lap turning ratio 0.57–0.95 vs GPS course), so it is only ever
compared LOCALLY: a robust rigid (rotation + translation) fit to the neighbouring
fixes, then the excluded fix's distance from the fitted path is its residual.

Pure, no I/O. Computes numbers only — the threshold DECISION lives in trajectory.py,
the single GPS-trust decision point. Spec: docs/superpowers/specs/
2026-10-10-sensor-fusion-rethink-design.md §W1 (facts F7–F9).
"""
from __future__ import annotations

import numpy as np

G = 9.80665
JUMP_WIN_S = 1.5            # 2026-10-10, ±neighbour window; DR error ~1 m at 1 s vs 6-7 m at 5 s (m6_horizon, 486 windows)
JUMP_MIN_NB = 6             # 2026-10-10, min neighbour fixes for a fit (~0.4 s of 8 Hz each side); 1 Hz sessions never qualify
JUMP_HUBER_M = 3.0          # 2026-10-10, Huber scale of the IRLS fit = clean residual p99 (198k fixes, 207 laps)
JUMP_IRLS_ITERS = 3         # 2026-10-10, prototype jd.py
JUMP_MIN_SPEED_MPS = 10.0   # 2026-10-10, below this yaw = lat_g·g/v is ill-conditioned
YAW_V_FLOOR_MPS = 5.0       # 2026-10-10, v floor inside the yaw division


def fresh_fix_rows(lat, lon) -> np.ndarray:
    """Row indices of fresh GPS fixes: finite and lat or lon changed from the previous row."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    ch = np.ones(len(lat), dtype=bool)
    ch[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
    return np.where(ch & np.isfinite(lat) & np.isfinite(lon))[0]


def path_shape(t, v_mps, lat_g) -> np.ndarray:
    """(n, 2) dead-reckoned path in an arbitrary frame (starts at origin heading +x)."""
    t = np.asarray(t, dtype=float)
    v = np.nan_to_num(np.asarray(v_mps, dtype=float))
    lg = np.nan_to_num(np.asarray(lat_g, dtype=float))
    dt = np.diff(t)
    yaw = -lg[:-1] * G / np.maximum(v[:-1], YAW_V_FLOOR_MPS)
    psi = np.concatenate([[0.0], np.cumsum(yaw * dt)])
    px = np.concatenate([[0.0], np.cumsum(v[:-1] * np.cos(psi[:-1]) * dt)])
    py = np.concatenate([[0.0], np.cumsum(v[:-1] * np.sin(psi[:-1]) * dt)])
    return np.column_stack([px, py])


def _rigid_fit(A: np.ndarray, B: np.ndarray, w: np.ndarray):
    """Weighted 2-D rigid fit (closed form): rotation R and translation T minimising
    Σw|R·a + T − b|²."""
    w = w / w.sum()
    ma, mb = w @ A, w @ B
    a, b = A - ma, B - mb
    th = np.arctan2(np.sum(w * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])),
                    np.sum(w * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    return R, mb - R @ ma


def fix_residuals(t, v_mps, lat_g, x, y, fix_idx) -> np.ndarray:
    """Per fix in `fix_idx` (row indices into t/x/y, time-ordered): distance in metres
    from the sensor path rigidly fitted (Huber IRLS) to the OTHER fixes within
    ±JUMP_WIN_S. NaN where not evaluable: lat_g missing or all zero, v < JUMP_MIN_SPEED_MPS
    at the fix, non-finite position, fewer than JUMP_MIN_NB neighbours, or any non-finite
    lat_g or speed sample within ±JUMP_WIN_S of the fix (never judged on a zero-filled path)."""
    fix_idx = np.asarray(fix_idx, dtype=int)
    res = np.full(len(fix_idx), np.nan)
    lg_raw = np.asarray(lat_g, dtype=float)
    if len(fix_idx) == 0 or not np.isfinite(lg_raw).any() or np.nanmax(np.abs(lg_raw)) == 0.0:
        return res
    t = np.asarray(t, dtype=float)
    bad = ~np.isfinite(lg_raw) | ~np.isfinite(np.asarray(v_mps, dtype=float))
    lg = np.nan_to_num(lg_raw)
    v = np.nan_to_num(np.asarray(v_mps, dtype=float))
    P = path_shape(t, v, lg)
    F = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    good = np.isfinite(F[fix_idx]).all(axis=1)
    tf = t[fix_idx]
    for n, i in enumerate(fix_idx):
        if v[i] < JUMP_MIN_SPEED_MPS or not good[n]:
            continue
        rlo = np.searchsorted(t, tf[n] - JUMP_WIN_S, side="left")
        rhi = np.searchsorted(t, tf[n] + JUMP_WIN_S, side="right")
        if bad[rlo:rhi].any():
            continue
        lo = np.searchsorted(tf, tf[n] - JUMP_WIN_S, side="left")
        hi = np.searchsorted(tf, tf[n] + JUMP_WIN_S, side="right")
        nb = np.concatenate([np.arange(lo, n), np.arange(n + 1, hi)])
        nb = nb[good[nb]]
        if len(nb) < JUMP_MIN_NB:
            continue
        A, B = P[fix_idx[nb]], F[fix_idx[nb]]
        w = np.ones(len(nb))
        for _ in range(JUMP_IRLS_ITERS):
            R, T = _rigid_fit(A, B, w)
            r = np.hypot(*((A @ R.T + T) - B).T)
            if np.all(r <= JUMP_HUBER_M):   # next weights all 1 -> identical refit
                break
            w = np.where(r <= JUMP_HUBER_M, 1.0, JUMP_HUBER_M / np.maximum(r, 1e-9))
        res[n] = float(np.hypot(*((R @ P[i] + T) - F[i])))
    return res
