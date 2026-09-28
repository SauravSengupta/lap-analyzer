"""GPS lag correction — re-time GPS-derived channels onto the row clock at ingest.

The Garmin GLO 2 feed TrackAddict logs is late relative to the row clock (median 0.45 s,
mostly per-session, ±0.07 s lap-to-lap). TrackAddict's GPS_Delay column is ~0 — it only
measures Bluetooth transport. This module estimates the lag per lap by cross-correlating
GPS speed against OBD speed (or d(GPS speed)/dt against the accelerometer when there is
no OBD) and re-attaches GPS channels to the row clock by sample-and-hold.

Design: docs/superpowers/specs/2026-09-18-gps-lag-correction-design.md. Pure, no I/O.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

LAG_MIN_SPEED_MPH = 20.0    # 2026-09-18, below this there is no lag signal and pit-lane holds corrupt corr
LAG_GRID_S = 0.05           # 2026-09-18, uniform resample step and τ search step
LAG_SEARCH_MIN_S = -0.3     # 2026-09-18, τ search floor (allows a small lead)
LAG_SEARCH_MAX_S = 1.5      # 2026-09-18, τ search ceiling (session medians seen up to 1.10 s)
OBD_HOLD_LEAD_S = 0.075     # 2026-09-18, OBD sample-and-hold leads the accel by 0.05–0.10 s (35 laps / 4 sessions)
LAG_MIN_CORR = 0.97         # 2026-09-18, min peak corr to accept a per-lap OBD estimate (corpus 0.99–1.00)
LAG_MIN_SPAN_S = 30.0       # 2026-09-18, min qualifying (>20 mph) data span per lap
LAG_MIN_CORR_ACCEL = 0.6    # 2026-09-18, min peak corr for the no-OBD accelerometer path
LAG_DEFAULT_S = 0.45        # 2026-09-18, corpus median GPS lag (618 clean laps, PIR + Ridge)
ACCEL_SMOOTH_S = 0.5        # 2026-09-27, smoothing window for dv/dt and long_g on the accel path
MPH_TO_MPS = 0.44704        # exact unit conversion
G = 9.80665                 # standard gravity, m/s²
METHOD_VERSION = "gps-lag-v1"

# Every GPS-derived column TrackAddict logs; re-timed together when present.
GPS_COLUMNS = ["lat", "long", "altitude_m", "gps_accuracy_m", "speed_mph_gps",
               "heading", "sector", "lap"]


@dataclass(frozen=True)
class LagEstimate:
    tau_s: float   # lag to apply (already net of OBD_HOLD_LEAD_S on the OBD path)
    corr: float    # peak correlation that produced it (NaN for session/default/disabled)
    source: str    # "lap" | "session" | "accel" | "default" | "disabled" ("none" = unresolved)


_NONE = LagEstimate(float("nan"), float("nan"), "none")


def xcorr_lag(t, ref, sig, mask=None) -> tuple[float, float]:
    """τ maximising Pearson corr(ref(t), sig(t+τ)) over [LAG_SEARCH_MIN_S, LAG_SEARCH_MAX_S].

    Both series are linearly resampled onto a LAG_GRID_S grid; grid points where `mask`
    (per input row, nearest) is False do not participate. The peak is refined with a
    parabola through it and its two neighbours. Returns (nan, nan) if nothing is usable.
    """
    t = np.asarray(t, dtype=float)
    ref = np.asarray(ref, dtype=float)
    sig = np.asarray(sig, dtype=float)
    ok = np.isfinite(t) & np.isfinite(ref) & np.isfinite(sig)
    if ok.sum() < 3:
        return float("nan"), float("nan")
    t, ref, sig = t[ok], ref[ok], sig[ok]
    m = np.ones(len(t), bool) if mask is None else np.asarray(mask, bool)[ok]
    grid = np.arange(t[0], t[-1], LAG_GRID_S)
    if len(grid) < 3:
        return float("nan"), float("nan")
    r = np.interp(grid, t, ref)
    s = np.interp(grid, t, sig)
    gm = np.interp(grid, t, m.astype(float)) > 0.5
    n = len(grid)
    ks = np.arange(round(LAG_SEARCH_MIN_S / LAG_GRID_S), round(LAG_SEARCH_MAX_S / LAG_GRID_S) + 1)
    corrs = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        if k >= 0:
            a, b, va = r[:n - k], s[k:], gm[:n - k] & gm[k:]
        else:
            a, b, va = r[-k:], s[:n + k], gm[-k:] & gm[:n + k]
        if va.sum() < 3:
            continue
        a, b = a[va], b[va]
        if a.std() == 0 or b.std() == 0:
            continue
        corrs[i] = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(corrs).any():
        return float("nan"), float("nan")
    i = int(np.nanargmax(corrs))
    tau = ks[i] * LAG_GRID_S
    if 0 < i < len(ks) - 1 and np.isfinite(corrs[i - 1]) and np.isfinite(corrs[i + 1]):
        y0, y1, y2 = corrs[i - 1], corrs[i], corrs[i + 1]
        denom = y0 - 2 * y1 + y2
        if denom < 0:
            tau += 0.5 * (y0 - y2) / denom * LAG_GRID_S
    return float(tau), float(corrs[i])


def _smooth(t, x):
    """Centred rolling mean over ~ACCEL_SMOOTH_S (window in samples from the median dt)."""
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    w = max(1, int(round(ACCEL_SMOOTH_S / dt)))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


def estimate_gps_lag(t, v_obd, v_gps, long_g) -> LagEstimate:
    """One lap's GPS lag. OBD path when OBD speed exists, else the accelerometer path.

    Returns source "lap"/"accel" when accepted, else source "none" (tau NaN) — the
    session resolver (resolve_session_lags) applies the fallback ladder.
    """
    t = np.asarray(t, dtype=float)
    v_obd = np.asarray(v_obd, dtype=float)
    v_gps = np.asarray(v_gps, dtype=float)
    long_g = np.asarray(long_g, dtype=float)

    if np.isfinite(v_obd).any():
        fast = np.nan_to_num(v_obd, nan=0.0) > LAG_MIN_SPEED_MPH
        span = fast.sum() * (float(np.median(np.diff(t))) if len(t) > 1 else 0.0)
        if span < LAG_MIN_SPAN_S:
            return _NONE
        tau_peak, corr = xcorr_lag(t, v_obd, v_gps, mask=fast)
        if not np.isfinite(corr) or corr < LAG_MIN_CORR:
            return _NONE
        return LagEstimate(tau_peak - OBD_HOLD_LEAD_S, corr, "lap")

    # No OBD: d(GPS speed)/dt against long_g (canonical + = accel), both smoothed.
    fast = np.nan_to_num(v_gps, nan=0.0) > LAG_MIN_SPEED_MPH
    if len(t) < 3 or fast.sum() < 3:
        return _NONE
    dvdt = np.gradient(np.nan_to_num(v_gps) * MPH_TO_MPS, t) / G
    tau, corr = xcorr_lag(t, _smooth(t, long_g), _smooth(t, dvdt), mask=fast)
    if not np.isfinite(corr) or corr < LAG_MIN_CORR_ACCEL:
        return _NONE
    return LagEstimate(tau, corr, "accel")


def resolve_session_lags(per_lap: dict[int, LagEstimate],
                         edge_laps: set[int]) -> dict[int, LagEstimate]:
    """Apply the fallback ladder: accepted non-edge laps keep their own τ; every other lap
    gets the session median of those; with none accepted, LAG_DEFAULT_S (flagged)."""
    accepted = {k: e for k, e in per_lap.items()
                if e.source in ("lap", "accel") and k not in edge_laps}
    if not accepted:
        return {k: LagEstimate(LAG_DEFAULT_S, float("nan"), "default") for k in per_lap}
    median = float(np.median([e.tau_s for e in accepted.values()]))
    return {k: accepted.get(k, LagEstimate(median, float("nan"), "session")) for k in per_lap}


def estimate_session_gps_lag(df: pd.DataFrame, enabled: bool = True) -> dict[int, LagEstimate]:
    """Per-lap τ for one session, keyed by the lap number as logged (pre-shift)."""
    laps = sorted(int(k) for k in df["lap"].unique())
    if not enabled:
        return {k: LagEstimate(0.0, float("nan"), "disabled") for k in laps}
    per_lap = {}
    for k, g in df.groupby("lap", sort=True):
        per_lap[int(k)] = estimate_gps_lag(g["t"], g["speed_mph"], g["speed_mph_gps"], g["long_g"])
    # Warmup/cooldown (pit-lane crawl, partial laps) never set their own τ.
    edge = {laps[0], laps[-1]} if len(laps) >= 3 else set()
    return resolve_session_lags(per_lap, edge)


def apply_gps_lag(df: pd.DataFrame, tau_by_lap: dict[int, float]) -> pd.DataFrame:
    """Re-attach GPS-derived columns to the row clock by SAMPLE-AND-HOLD (never interpolate:
    the trajectory layer detects fresh fixes by |Δtrack_dist| > FRESH_FIX_M).

    τ is not applied as a step function of lap: at a lap boundary, a straight per-lap τ
    step can make the source index j go BACKWARD (e.g. lap k+1 has a smaller τ than lap
    k, so the first rows of lap k+1 would re-use fixes already consumed at the end of
    lap k), replaying position/altitude/speed backward across the seam. Instead τ(t) is
    a piecewise-linear function through one knot per original lap at that lap's time
    midpoint t_mid_k = (first t of lap k + last t of lap k) / 2, value tau_by_lap[k]
    (laps missing from the dict, or with a NaN τ, contribute no knot; with no knots at
    all, τ = 0 everywhere). At each lap's own midpoint the applied τ equals exactly that
    lap's estimate; between midpoints τ blends linearly, and outside the first/last
    knot it holds constant (np.interp's default). This keeps the source index j
    monotone non-decreasing as long as dτ/dt > −1, which is guaranteed here because
    |Δτ| between adjacent lap estimates is at most ~1.8 s spread over at least half a
    lap of t on each side (dτ/dt therefore stays close to 0, never near −1).

    Row i takes col[j], j = last index with t[j] <= t[i] + τ(t[i]) (never interpolated).
    Rows within τ of the log end hold the last fix.
    """
    out = df.copy()
    t = df["t"].to_numpy(dtype=float)
    lap = df["lap"].to_numpy()

    knot_t, knot_tau = [], []
    for k in sorted(pd.unique(lap)):
        tau_k = tau_by_lap.get(k)
        if tau_k is None or not np.isfinite(tau_k):
            continue
        m = lap == k
        knot_t.append((t[m][0] + t[m][-1]) / 2.0)
        knot_tau.append(float(tau_k))

    if knot_t:
        order = np.argsort(knot_t)
        knot_t = np.asarray(knot_t, dtype=float)[order]
        knot_tau = np.asarray(knot_tau, dtype=float)[order]
        tau = np.interp(t, knot_t, knot_tau)
    else:
        tau = np.zeros(len(t), dtype=float)

    j = np.searchsorted(t, t + tau, side="right") - 1
    j = np.clip(j, 0, len(t) - 1)
    for col in GPS_COLUMNS:
        if col in df.columns:
            out[col] = df[col].to_numpy()[j]
    out["gps_lag_s"] = tau.astype(np.float32)
    return out


def summarize_lags(lags: dict[int, LagEstimate]) -> dict:
    """The meta.json `gps_lag` block (method_version is the staleness marker)."""
    counts = Counter(e.source for e in lags.values())
    taus = [e.tau_s for e in lags.values() if np.isfinite(e.tau_s)]
    block = {
        "method_version": METHOD_VERSION,
        "session_median_s": round(float(np.median(taus)), 3) if taus else None,
        "n_lap": counts["lap"], "n_session": counts["session"],
        "n_accel": counts["accel"], "n_default": counts["default"],
    }
    if counts["disabled"]:
        block["n_disabled"] = counts["disabled"]
    return block
