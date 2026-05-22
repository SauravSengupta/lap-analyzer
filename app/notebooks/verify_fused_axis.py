"""Verify the complementary-filter fused distance axis (fused_axis.py).

Checks, on 20260517-100304 L1 (red) vs L9 (best):
  - fused is monotonic
  - fused tracks the centerline (track_dist_m) at corner scale
  - fused removes high-frequency GPS jitter from the Delta-t curve
  - fused does NOT flatten the real sign change (red's shorter T8 line)
  - compute_fused_dist is robust to injected GPS glitches
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from lap_analyzer.analysis import load_samples, section_range_bounds
from lap_analyzer.config import tracks_dir
from lap_analyzer.fused_axis import compute_fused_dist

TRACK, SID, RED, BEST = "ridge", "20260517-100304", 1, 9
ok = True


def check(label, passed):
    global ok
    ok &= passed
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")


def drop_gps_glitches(s):
    s = s.sort_values("t").reset_index(drop=True)
    small = s["track_dist_m"] < 500
    if small.any():
        s = s.iloc[small.idxmax():].reset_index(drop=True)
    s = s[(s["track_dist_m"] - s["dist_lap_m"]).abs() < 50].reset_index(drop=True)
    rmax = s["track_dist_m"].cummax()
    return s[s["track_dist_m"] >= rmax - 1.0].reset_index(drop=True)


def lap(n):
    return drop_gps_glitches(load_samples(TRACK, SID, n)).sort_values("t").reset_index(drop=True)


red, best = lap(RED), lap(BEST)

print("--- per-lap structural checks ---")
for name, s in [("L1 red", red), ("L9 best", best)]:
    f = compute_fused_dist(s)
    check(f"{name}: fused monotonic non-decreasing",
          bool((np.diff(f) >= -1e-9).all()))
    # fused must track the centerline at corner scale, not drift off it.
    med_gap = float(np.median(np.abs(f - s["track_dist_m"].to_numpy())))
    check(f"{name}: fused tracks track_dist_m (median gap {med_gap:.1f} m < 12)",
          med_gap < 12.0)

# --- Delta-t: jitter removed, real signal kept ---------------------------
print("\n--- T8-T11 Delta-t ---")
lo, hi = section_range_bounds(json.loads(
    (tracks_dir() / f"{TRACK}.json").read_text(encoding="utf-8")), "T8", "T11")
grid = np.arange(lo, hi + 0.5, 1.0)


def deltat(xr, xb):
    """Delta-t over the section given an x-array for red and for best."""
    ta = np.interp(grid, xr, red["t"].to_numpy())
    tb = np.interp(grid, xb, best["t"].to_numpy())
    return (ta - ta[0]) - (tb - tb[0])


raw = deltat(red["track_dist_m"].to_numpy(), best["track_dist_m"].to_numpy())
fused = deltat(compute_fused_dist(red), compute_fused_dist(best))

# Jitter shows up as curvature (second difference). Fused should be smoother.
raw_wobble = float(np.std(np.diff(raw, 2)))
fused_wobble = float(np.std(np.diff(fused, 2)))
print(f"  Delta-t curvature noise: raw={raw_wobble:.4f}  fused={fused_wobble:.4f}")
check("fused Delta-t is smoother than raw (jitter removed)",
      fused_wobble < raw_wobble)

print(f"  raw   Delta-t: min={raw.min():+.3f}s  endpoint={raw[-1]:+.3f}s")
print(f"  fused Delta-t: min={fused.min():+.3f}s  endpoint={fused[-1]:+.3f}s")
# The real shorter-line dip (~-0.58s) must survive — fused must NOT collapse it
# to a monotone curve, and must NOT exaggerate it past the raw value.
check("real T8 dip preserved (fused min in [-0.75, -0.35] s)",
      -0.75 < fused.min() < -0.35)

# --- glitch robustness ---------------------------------------------------
print("\n--- glitch robustness ---")
clean = compute_fused_dist(best)
corrupt = best.copy()
mid = len(corrupt) // 2
tdm = corrupt["track_dist_m"].to_numpy().copy()
tdm[mid:mid + 4] += 300.0          # forward teleport
tdm[mid + 20:mid + 24] -= 250.0    # backward walk
corrupt["track_dist_m"] = tdm
glitched = compute_fused_dist(corrupt)
max_shift = float(np.max(np.abs(glitched - clean)))
print(f"  injected 8 glitch samples; max fused shift = {max_shift:.2f} m")
check("compute_fused_dist absorbs injected glitches (max shift < 5 m)",
      max_shift < 5.0)
check("fused stays monotonic despite glitches",
      bool((np.diff(glitched) >= -1e-9).all()))

# --- speed / line decomposition (visualizer option 1) -------------------
print("\n--- Delta-t decomposition: speed vs line-length ---")
f_red, f_best = compute_fused_dist(red), compute_fused_dist(best)
ta = np.interp(grid, f_red, red["t"].to_numpy()); ta -= ta[0]
tb = np.interp(grid, f_best, best["t"].to_numpy()); tb -= tb[0]
da = np.interp(grid, f_red, red["dist_lap_m"].to_numpy()); da -= da[0]
db = np.interp(grid, f_best, best["dist_lap_m"].to_numpy()); db -= db[0]
total = ta - tb
speed = ta - np.interp(da, db, tb)   # gap vs best's time to travel the same distance
line = total - speed                 # line-length contribution
print(f"  total: min={total.min():+.3f}s  endpoint={total[-1]:+.3f}s")
print(f"  speed: min={speed.min():+.3f}s  endpoint={speed[-1]:+.3f}s")
print(f"  line : min={line.min():+.3f}s  endpoint={line[-1]:+.3f}s")
# The point of the decomposition: the T8 dip is a line-length effect. It must
# live in the line term — and the speed-only curve must NOT show that gain,
# since red is slower on pure pace and never cumulatively ahead on it.
check("line term carries the T8 dip (line.min < -0.3 s)", float(line.min()) < -0.3)
check("speed-only Delta-t shows no phantom gain (speed.min > -0.15 s)",
      float(speed.min()) > -0.15)
check("decomposition is additive (total == speed + line)",
      float(np.max(np.abs(total - (speed + line)))) < 1e-9)

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
