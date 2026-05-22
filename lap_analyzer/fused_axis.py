"""Fused distance axis: a complementary filter of OBD distance and GPS position.

`track_dist_m` (GPS centerline projection) is pinned to physical track position
but jitters sample-to-sample and walks backwards on glitched laps. `dist_lap_m`
(OBD speed integration, rescaled to the canonical lap length) is smooth and
monotonic but carries no track-position information between the lap endpoints.

This fuses them the way a complementary filter does: OBD supplies the smooth,
monotonic, jitter-free base; the *low-frequency* part of (GPS - OBD) supplies the
real track-position signal — including genuine racing-line-length differences,
where a lap covers a stretch of track in more or fewer metres than the centerline.
High-frequency GPS jitter is discarded.

The result is monotonic by construction and tracks the centerline at corner scale
while staying noise-free at sample scale. It generalizes normalize.py's dist_lap_m
rescaling, which corrects the OBD-vs-track offset only at the two lap endpoints;
this corrects it continuously.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A sample whose GPS projection disagrees with OBD distance by more than this is a
# teleport; its offset is meaningless and must not enter the low-pass. Callers
# normally pass already glitch-filtered samples, so this is a backstop.
_GLITCH_OFFSET_M = 50.0

# Low-pass window (samples). At TrackAddict's ~10 Hz this spans a couple of
# seconds — long enough to kill sample jitter, short enough to keep corner-scale
# line-length structure (corners are 100-200 m of track).
_SMOOTH_WINDOW = 25


def compute_fused_dist(samples: pd.DataFrame, smooth_window: int = _SMOOTH_WINDOW) -> np.ndarray:
    """Per-sample fused distance for one lap, in the canonical track frame.

    Needs columns t, dist_lap_m, track_dist_m. Rows may be in any order; the
    returned array is aligned to `samples`' current row order.

    fused = dist_lap_m + lowpass(track_dist_m - dist_lap_m). `dist_lap_m` is the
    OBD base (monotone, jitter-free); the smoothed offset bends it onto the
    centerline at corner scale. Pass glitch-filtered samples (see
    visualizer/shared.py drop_gps_glitches) — gross teleports are masked out of
    the offset here as a backstop, but the lap-wraparound trim is the caller's job.
    """
    s = samples.reset_index(drop=True)
    obd = s["dist_lap_m"].to_numpy(dtype=float)
    if len(s) < 2:
        return obd.copy()

    t_order = np.argsort(s["t"].to_numpy(), kind="stable")
    obd_t = obd[t_order]
    gps_t = s["track_dist_m"].to_numpy(dtype=float)[t_order]

    offset = gps_t - obd_t
    good = np.isfinite(offset) & (np.abs(offset) < _GLITCH_OFFSET_M)
    idx = np.arange(len(offset))
    if good.all():
        pass
    elif good.sum() >= 2:
        offset = np.interp(idx, idx[good], offset[good])
    elif good.sum() == 1:
        offset = np.full(len(offset), float(offset[good][0]))
    else:
        offset = np.zeros(len(offset))

    # Low-pass: median (rejects residual spikes) then mean (smooths).
    o = pd.Series(offset).rolling(smooth_window, center=True, min_periods=1).median()
    o = o.rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()

    fused_t = np.maximum.accumulate(obd_t + o)
    out = np.empty(len(s))
    out[t_order] = fused_t
    return out
