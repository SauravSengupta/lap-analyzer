"""Decompose the cross-lap spread at each corner into:
  - GPS sample noise (the per-sample random error TrackAddict reports)
  - Drift correction residual (how well median-of-17 anchors corrects per-lap drift)
  - Driver line variation (residual after subtracting the above)

Method: at each Maps pin, after drift correction, the lap's GPS *should* be at the pin
(if the user reliably touches the kerb). The spread of "closest-sample-to-pin" across
laps tells us calibration_noise² + line_variation².

For T13 (the user's most consistent kerb-touch corner), we assume line variation ≈ 0,
so the observed spread is pure calibration noise. We treat T13 as the calibration-noise
calibration and back out line variation for other corners.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))
ROOT = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
PINS = {k: v for k, v in json.loads(
    Path(r"D:\Projects\lap-analyzer\data\notes\ridge_apex_pins.json").read_text(encoding="utf-8")
).items() if not k.startswith("_")}


# Pull all reliable transits per corner
transits = pd.concat([pd.read_parquet(p) for p in ROOT.rglob("corners.parquet")], ignore_index=True)
reliable = transits[transits.transit_reliable].copy()

# Per-lap closest-sample distance to each Maps pin
rows = []
gps_accs = []
for (sid, lap), _ in reliable.groupby(["session_id", "lap"]):
    sp = ROOT / sid / "samples.parquet"
    if not sp.exists():
        continue
    s = pd.read_parquet(sp, columns=["lap", "lat", "long",
                                     "gps_drift_lat_m", "gps_drift_lon_m",
                                     "gps_accuracy_m", "corner"])
    ls = s[s["lap"] == int(lap)]
    if len(ls) == 0:
        continue
    dlat = float(ls.gps_drift_lat_m.iloc[0])
    dlon = float(ls.gps_drift_lon_m.iloc[0])
    lat_c = ls["lat"].values - dlat / M_PER_DEG_LAT
    lon_c = ls["long"].values - dlon / M_PER_DEG_LON

    for cid, pin in PINS.items():
        # only count this lap-at-corner if the corner's transit is reliable
        corner_id_for_lookup = cid if not cid.endswith("b") else cid[:-1]
        ok = ((reliable.session_id == sid) & (reliable.lap == int(lap)) & (reliable.corner_id == corner_id_for_lookup)).any()
        if not ok:
            continue
        # Restrict search to the corner's region for cleanliness
        in_region = ls.corner == corner_id_for_lookup
        if not in_region.any():
            continue
        ll_c = lat_c[in_region]
        ln_c = lon_c[in_region]
        d = np.sqrt(((ll_c - pin["lat"]) * M_PER_DEG_LAT) ** 2 + ((ln_c - pin["long"]) * M_PER_DEG_LON) ** 2)
        rows.append({"session_id": sid, "lap": int(lap), "corner": cid, "dist_m": float(d.min())})
    # Sample-level GPS accuracy (whole lap)
    gps_accs.append(float(ls["gps_accuracy_m"].median()))

df = pd.DataFrame(rows)
gps_acc_med = float(np.median(gps_accs))

print(f"GPS sample accuracy (median across reliable laps): {gps_acc_med:.2f}m\n")

print("=== Per-corner closest-sample-to-pin distance (drift-corrected) ===")
print(f"  {'pin':<5s} {'n':>4s} {'median':>8s} {'mean':>7s} {'std':>7s}")
stats = {}
for cid in PINS.keys():
    sub = df[df.corner == cid]
    if len(sub) == 0: continue
    stats[cid] = {"n": len(sub), "median": sub.dist_m.median(), "mean": sub.dist_m.mean(), "std": sub.dist_m.std()}
    print(f"  {cid:<5s} {len(sub):>4d} {sub.dist_m.median():>6.2f}m {sub.dist_m.mean():>5.2f}m {sub.dist_m.std():>5.2f}m")

# T13 is our calibration reference — user reliably touches the kerb there.
# Observed spread at T13 ≈ calibration noise (drift residual + GPS noise).
t13 = stats["T13"]
print(f"\n=== Decomposition (using T13 as the line-variation=0 baseline) ===")
print(f"T13 mean: {t13['mean']:.2f}m  (= GPS sample noise floor + drift correction residual)")
print(f"T13 std:  {t13['std']:.2f}m   (= per-lap variation of drift correction residual)")
print()
print(f"For other corners: observed_spread² = calibration_noise² + line_variation²")
print(f"So line_variation = sqrt(observed_mean^2 - T13_mean^2) when observed > T13_mean.\n")
print(f"  {'pin':<5s} {'mean':>7s} {'calib':>7s} {'line var':>9s}  interpretation")
calib_mean = t13["mean"]
for cid, s in stats.items():
    if cid == "T13":
        print(f"  {cid:<5s} {s['mean']:>5.2f}m {calib_mean:>5.2f}m {'~ 0':>9s}  baseline (user reliably touches kerb)")
        continue
    if s["mean"] <= calib_mean:
        line_var = 0
        note = "below baseline — even tighter than T13"
    else:
        line_var = (s["mean"] ** 2 - calib_mean ** 2) ** 0.5
        if line_var < 2:
            note = "essentially on the kerb"
        elif line_var < 5:
            note = "small line variation (1-2 car widths)"
        elif line_var < 10:
            note = "moderate line variation"
        else:
            note = "significant line variation (multi-line / user doesn't always touch)"
    print(f"  {cid:<5s} {s['mean']:>5.2f}m {calib_mean:>5.2f}m {line_var:>7.2f}m  {note}")
