"""GPS lag correction — re-time GPS speed AND position onto the row clock at ingest.

The Garmin GLO 2's speed output is late relative to the row clock (median 0.45 s,
mostly per-session, ±0.07 s lap-to-lap). TrackAddict's GPS_Delay column is ~0 — it only
measures Bluetooth transport. This module estimates the SPEED lag per lap by
cross-correlating GPS speed against OBD speed (or d(GPS speed)/dt against the
accelerometer when there is no OBD) and re-attaches GPS speed to the row clock by
sample-and-hold.

Position lags too, but less. Originally (2026-09-27) this was measured indirectly as
speed-minus-offset: τ_pos = τ_speed − POS_SPEED_OFFSET_S (0.12 s, cumulative GPS path
length vs OBD odometer, speed channel as a control through the same fit; PIR 135 /
Ridge 196 laps). That offset mixed two different estimation methods (xcorr for
τ_speed, path-length for the offset) and over-corrected PIR. As of 2026-09-28, τ_pos
is measured DIRECTLY per session with the path-length method (`estimate_position_lag`
/ `estimate_session_position_lag`) — one τ_pos for the whole session (per-lap position
estimates are too noisy: p10-p90 ≈ 0.04-0.45 s). The speed-offset method survives only
as the fallback for GPS-only sessions (no OBD odometer to fit against) or sessions
with too few eligible laps. An earlier attempt that shifted position by the full speed
lag overshot by ~0.2 s and regressed PIR T1 — that is why position is never just given
τ_speed. τ_pos is not clamped: it may be negative.

The TrackAddict Lap counter is GPS-position-timed, so it moves WITH position (same
τ_pos, same source index) — lap boundaries move earlier by τ_pos·v (~8 m at S/F speed).
`sector` moves with position too. `heading` is left exactly as logged: it lags ~0.08 s
more than position-derived course, but nothing downstream reads it.

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
METHOD_VERSION = "gps-lag-v3"  # v1: speed-only. v2: τ_pos = τ_speed - offset. v3: τ_pos
                                # measured directly per session (path-length method, 2026-09-28)
POS_SPEED_OFFSET_S = 0.12   # 2026-09-27, speed-channel minus position lag (cumulative GPS
                            # path length vs OBD odometer, PIR 135 / Ridge 196 laps).
                            # Fallback only as of gps-lag-v3: τ_pos = τ_speed − 0.12 when the
                            # direct path-length estimate is unusable (GPS-only session, or
                            # too few eligible laps).
POS_LAG_GRID_S = 0.01       # 2026-09-27, path-length vs OBD odometer, PIR 135 / Ridge 196 laps: τ search step
POS_SMOOTH_FIXES = 7        # 2026-09-27, path-length vs OBD odometer, PIR 135 / Ridge 196 laps: centred rolling mean (in fresh fixes)
POS_TELEPORT_M = 30.0       # 2026-09-27, path-length vs OBD odometer, PIR 135 / Ridge 196 laps: reject a lap with a fresh-fix jump above this
POS_TRIM_ROWS = 50          # 2026-09-27, path-length vs OBD odometer, PIR 135 / Ridge 196 laps: rows trimmed off each end before fitting
POS_MIN_ROWS = 1500         # 2026-09-27, path-length vs OBD odometer, PIR 135 / Ridge 196 laps: per-lap eligibility floor
POS_MAX_SLOW_FRAC = 0.05    # 2026-09-27, path-length vs OBD odometer, PIR 135 / Ridge 196 laps: per-lap eligibility ceiling (rows < LAG_MIN_SPEED_MPH)
POS_MIN_LAPS = 2            # 2026-09-28, minimum eligible non-edge laps to trust the session path-length median over the speed-offset fallback

# speed_mph_gps is re-timed by τ_speed. Position, sector, and the TrackAddict lap
# counter are GPS-position-timed and are re-timed by a single per-session τ_pos
# (measured directly by estimate_session_position_lag). heading lags position by a
# further ~0.08 s but no downstream consumer reads it, so it is left exactly as logged.
SPEED_COLUMNS = ["speed_mph_gps"]
POSITION_COLUMNS = ["lat", "long", "altitude_m", "gps_accuracy_m", "sector", "lap"]


@dataclass(frozen=True)
class LagEstimate:
    tau_s: float   # lag to apply (already net of OBD_HOLD_LEAD_S on the OBD path)
    corr: float    # peak correlation that produced it (NaN for session/default/disabled)
    source: str    # "lap" | "session" | "accel" | "default" | "disabled" ("none" = unresolved)


_NONE = LagEstimate(float("nan"), float("nan"), "none")


def xcorr_lag(t, ref, sig, mask=None) -> tuple[float, float]:
    """τ maximising Pearson corr(ref(t), sig(t+τ)) over [LAG_SEARCH_MIN_S, LAG_SEARCH_MAX_S].

    Both series are linearly resampled onto a LAG_GRID_S grid; `mask` is likewise
    linearly interpolated onto the grid and thresholded at 0.5, and grid points where
    that falls False do not participate. The peak is refined with a parabola through it
    and its two neighbours. Returns (nan, nan) if nothing is usable.
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
    # Duplicate timestamps give np.gradient a zero denominator (inf/NaN); xcorr_lag's
    # isfinite filter drops those grid points, so the warning is noise here.
    with np.errstate(divide="ignore", invalid="ignore"):
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
    """Per-lap τ for one session, keyed by the lap number as logged."""
    laps = sorted(int(k) for k in df["lap"].unique())
    if not enabled:
        return {k: LagEstimate(0.0, float("nan"), "disabled") for k in laps}
    per_lap = {}
    for k, g in df.groupby("lap", sort=True):
        per_lap[int(k)] = estimate_gps_lag(g["t"], g["speed_mph"], g["speed_mph_gps"], g["long_g"])
    # Warmup/cooldown (pit-lane crawl, partial laps) never set their own τ.
    edge = {laps[0], laps[-1]} if len(laps) >= 3 else set()
    return resolve_session_lags(per_lap, edge)


def estimate_position_lag(t, lat, lon, speed_mph) -> float:
    """One lap's POSITION lag by the path-length method (2026-09-27, PIR 135 / Ridge
    196 laps): cross-correlate a smoothed cumulative GPS path length against OBD
    cumulative distance (trapezoidal). Pure per-lap estimator — session-level
    eligibility (min rows, max slow fraction) and the median/fallback ladder live in
    `estimate_session_position_lag`, not here. Returns NaN when the lap is too short,
    has too few fresh GPS fixes, or has a teleport glitch (a fresh-fix step > 30 m).

    Method: D(t) is the trapezoidal cumulative OBD distance. Fresh fixes are rows
    where lat or lon changed from the previous row (first row counts). Positions are
    projected to local (x, y) metres with ONE fixed reference (mean lat/lon over the
    lap — not a per-point cos(lat), which adds a spurious term and shortens the path),
    smoothed with a 7-fix centred rolling mean, then their cumulative path length S is
    interpolated onto the row clock. Both S and D are trimmed 50 rows off each end
    before fitting S(t) ≈ a + k·D(t − τ) + c·(t − t0) by least squares over a
    τ grid; τ is the value minimising the residual std. The returned lag is that τ,
    net of OBD_HOLD_LEAD_S (the OBD sample-and-hold's lead on the row clock).
    """
    t = np.asarray(t, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    speed_mph = np.asarray(speed_mph, dtype=float)
    n = len(t)
    if n - 2 * POS_TRIM_ROWS < 50:
        return float("nan")

    # D(t): trapezoidal cumulative OBD distance.
    v_mps = speed_mph * MPH_TO_MPS
    D = np.concatenate([[0.0], np.cumsum((v_mps[:-1] + v_mps[1:]) / 2.0 * np.diff(t))])

    # Fresh fixes: lat or lon changed from the previous row (first row counts).
    fresh = np.ones(n, dtype=bool)
    fresh[1:] = (lat[1:] != lat[:-1]) | (lon[1:] != lon[:-1])
    if fresh.sum() < 10:
        return float("nan")

    # Local projection with ONE fixed reference (not a per-point cos(lat), which
    # adds a spurious term and shortens the path ~9%).
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    lat0, lon0 = float(np.mean(lat_r)), float(np.mean(lon_r))
    c0 = np.cos(lat0)
    R = 6371000.0
    x = R * c0 * (lon_r - lon0)
    y = R * (lat_r - lat0)

    t_fresh = t[fresh]
    x_fresh = pd.Series(x[fresh]).rolling(POS_SMOOTH_FIXES, center=True, min_periods=1).mean().to_numpy()
    y_fresh = pd.Series(y[fresh]).rolling(POS_SMOOTH_FIXES, center=True, min_periods=1).mean().to_numpy()

    steps = np.hypot(np.diff(x_fresh), np.diff(y_fresh))
    if len(steps) and steps.max() > POS_TELEPORT_M:
        return float("nan")  # teleport glitch

    S_fresh = np.concatenate([[0.0], np.cumsum(steps)])
    S = np.interp(t, t_fresh, S_fresh)

    lo, hi = POS_TRIM_ROWS, n - POS_TRIM_ROWS
    t_trim, S_trim = t[lo:hi], S[lo:hi]
    if len(t_trim) < 50:
        return float("nan")

    t0 = t_trim[0]
    best_tau, best_std = 0.0, np.inf
    for tau in np.arange(LAG_SEARCH_MIN_S, LAG_SEARCH_MAX_S + POS_LAG_GRID_S, POS_LAG_GRID_S):
        D_shift = np.interp(t_trim - tau, t, D)
        A = np.column_stack([np.ones_like(t_trim), D_shift, t_trim - t0])
        coef, *_ = np.linalg.lstsq(A, S_trim, rcond=None)
        resid = S_trim - A @ coef
        std = float(resid.std())
        if std < best_std:
            best_std, best_tau = std, float(tau)

    return best_tau - OBD_HOLD_LEAD_S


def estimate_session_position_lag(df: pd.DataFrame, speed_lags: dict[int, "LagEstimate"],
                                  enabled: bool = True) -> tuple[float, str, int]:
    """One τ_pos for the whole session (2026-09-28: per-lap position estimates are too
    noisy, p10-p90 ~= 0.04-0.45 s, to apply lap-by-lap).

    Tries the path-length method (`estimate_position_lag`) over each non-edge lap
    (edge = first/last lap when >= 3 laps) that meets the per-lap eligibility rule
    (>= POS_MIN_ROWS rows, <= POS_MAX_SLOW_FRAC of rows under LAG_MIN_SPEED_MPH) and
    returns a finite value. With >= POS_MIN_LAPS such laps, returns
    (median, "path", n_laps). Otherwise — a GPS-only session (no OBD speed to fit
    against) or an OBD session with too few eligible laps — falls back to the median
    over laps of (τ_speed - POS_SPEED_OFFSET_S), source "offset". `enabled=False`
    always returns (0.0, "disabled", 0). Not clamped.
    """
    if not enabled:
        return 0.0, "disabled", 0

    laps = sorted(int(k) for k in df["lap"].unique())
    edge = {laps[0], laps[-1]} if len(laps) >= 3 else set()

    taus: list[float] = []
    has_obd = "speed_mph" in df.columns and np.isfinite(df["speed_mph"]).any()
    if has_obd:
        for k, g in df.groupby("lap", sort=True):
            if int(k) in edge:
                continue
            if len(g) < POS_MIN_ROWS:
                continue
            if float((g["speed_mph"] < LAG_MIN_SPEED_MPH).mean()) > POS_MAX_SLOW_FRAC:
                continue
            tau = estimate_position_lag(g["t"], g["lat"], g["long"], g["speed_mph"])
            if np.isfinite(tau):
                taus.append(tau)

    if len(taus) >= POS_MIN_LAPS:
        return float(np.median(taus)), "path", len(taus)

    offset_taus = [e.tau_s - POS_SPEED_OFFSET_S for e in speed_lags.values() if np.isfinite(e.tau_s)]
    if not offset_taus:
        return float("nan"), "offset", 0
    return float(np.median(offset_taus)), "offset", len(offset_taus)


def apply_gps_lag(df: pd.DataFrame, tau_by_lap: dict[int, float],
                  tau_pos: float = 0.0) -> pd.DataFrame:
    """Re-attach speed_mph_gps (by τ_speed) and position + lap (by a constant τ_pos)
    to the row clock by SAMPLE-AND-HOLD (never interpolate — the output only ever
    holds values the receiver actually reported). `tau_by_lap` is the SPEED τ per lap;
    `tau_pos` is a single SESSION-level position τ (measured directly by
    `estimate_session_position_lag` — no blend needed since it is already one value
    for the whole session; default 0.0 so the disabled path is the identity for both
    groups). heading is left as logged (nothing downstream reads it).

    τ_speed is not applied as a step function of lap: at a lap boundary, a straight
    per-lap τ step can make the source index j go BACKWARD (e.g. lap k+1 has a smaller
    τ than lap k, so the first rows of lap k+1 would re-use fixes already consumed at
    the end of lap k), replaying a channel backward across the seam. Instead τ_speed(t)
    is a piecewise-linear function through one knot per ORIGINAL lap (read before the
    lap column is overwritten by the position re-timing) at that lap's time midpoint
    t_mid_k = (first t of lap k + last t of lap k) / 2, value tau_by_lap[k] (laps
    missing from the dict, or with a NaN τ, contribute no knot; with no knots at all,
    τ_speed = 0 everywhere). At each lap's own midpoint the applied τ_speed equals
    exactly that lap's estimate; between midpoints it blends linearly, and outside the
    first/last knot it holds constant (np.interp's default). This keeps the speed
    group's source index j monotone non-decreasing as long as dτ/dt > −1, which is
    guaranteed here because |Δτ| between adjacent lap estimates is at most ~1.8 s
    spread over at least half a lap of t on each side (dτ/dt therefore stays close to
    0, never near −1). τ_pos is constant over the whole session, so its source index is
    trivially monotone.

    Row i takes col[j], j = last index with t[j] <= t[i] + τ_group(t[i]) (never
    interpolated). Rows within τ of the log end hold the last fix. Each group applies
    its own duplicate-timestamp identity guard (τ_group == 0 -> identity).
    """
    out = df.copy()
    t = df["t"].to_numpy(dtype=float)
    lap = df["lap"].to_numpy()  # ORIGINAL lap column — knots come from this, before overwrite

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
        tau_speed = np.interp(t, knot_t, knot_tau)
    else:
        tau_speed = np.zeros(len(t), dtype=float)

    tau_pos_arr = np.full(len(t), float(tau_pos), dtype=float)

    def _retime(cols, tau):
        j = np.searchsorted(t, t + tau, side="right") - 1
        j = np.clip(j, 0, len(t) - 1)
        # τ = 0 must be the identity: with duplicate timestamps searchsorted lands on
        # the last duplicate, which would overwrite rows that should pass through
        # untouched.
        j = np.where(tau == 0, np.arange(len(t)), j)
        for col in cols:
            if col in df.columns:
                out[col] = df[col].to_numpy()[j]

    _retime(SPEED_COLUMNS, tau_speed)
    _retime(POSITION_COLUMNS, tau_pos_arr)

    out["gps_lag_s"] = tau_speed.astype(np.float32)
    out["gps_pos_lag_s"] = tau_pos_arr.astype(np.float32)
    return out


def summarize_lags(lags: dict[int, LagEstimate],
                   pos_lag: tuple[float, str, int] | None = None) -> dict:
    """The meta.json `gps_lag` block (method_version is the staleness marker).

    `session_median_s` is the median of the applied τ_speed (`tau_s`) across ALL laps
    in the session — including laps that fell back to the session median or an
    edge-lap default — not just the laps that set their own estimate. `pos_speed_offset_s`
    is kept for the fallback method's provenance even though it is no longer applied to
    every lap. `pos_lag`, when given, is `estimate_session_position_lag`'s
    (tau_pos, source, n_laps_used) — surfaced as `pos_lag_s`, `pos_lag_source`,
    `pos_lag_n_laps`.
    """
    counts = Counter(e.source for e in lags.values())
    taus = [e.tau_s for e in lags.values() if np.isfinite(e.tau_s)]
    block = {
        "method_version": METHOD_VERSION,
        "pos_speed_offset_s": POS_SPEED_OFFSET_S,
        "session_median_s": round(float(np.median(taus)), 3) if taus else None,
        "n_lap": counts["lap"], "n_session": counts["session"],
        "n_accel": counts["accel"], "n_default": counts["default"],
    }
    if counts["disabled"]:
        block["n_disabled"] = counts["disabled"]
    if pos_lag is not None:
        tau_pos, pos_source, n_laps_used = pos_lag
        block["pos_lag_s"] = round(float(tau_pos), 3) if np.isfinite(tau_pos) else None
        block["pos_lag_source"] = pos_source
        block["pos_lag_n_laps"] = n_laps_used
    return block
