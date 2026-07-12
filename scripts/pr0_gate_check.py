"""PR 0 corpus gate check (local; needs real data/, not run in CI).

Verifies the two corpus acceptance gates for the σ=10 m smoothed-tangent
build_gate change (plan §5 PR 0):

  A. Gate rotation p90 < 10° vs the smoothed-tangent reference (baseline: the old
     2-sample tangent has p90 ~46°, max ~59° at T14-exit / T10-entry / T3-exit).
  B. Driven-distance p2/p98 tails tighten (old 2-sample gate -> new smoothed
     gate), reproducing corner_bounds_verdict.csv v2. Expected (2026-07-11):
     T10 ~2.7x/3.6x, T14 ~4.1x/1.9x, T3 ~2.1x/2.0x, T4 ~1.3x/1.1x.

The old-gate baseline is implemented INLINE here (build_gate_2sample) — importing
the working-tree build_gate would measure the NEW gate on both sides.

Run:  python scripts/pr0_gate_check.py
Exit: 0 if rotation p90 < 10°, 1 otherwise, 2 if real data/ is unavailable.
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
from lap_analyzer.gates import Gate, TrackFrame, build_gate, gate_crossing_time  # noqa: E402

TRACK = "ridge"
ARTIFACTS = ROOT / "docs" / "superpowers" / "artifacts" / "gps-trust-2026-07-11"
ROTATION_P90_GATE_DEG = 10.0


def build_gate_2sample(cd, x, y, dist_m, half_width_m=40.0) -> Gate:
    """The retired 2-sample-tangent gate (production before PR 0), inline."""
    i = int(np.argmin(np.abs(cd - dist_m)))
    i0, i1 = max(0, i - 1), min(len(cd) - 1, i + 1)
    tx, ty = x[i1] - x[i0], y[i1] - y[i0]
    tn = math.hypot(tx, ty) or 1.0
    px, py = -ty / tn, tx / tn
    return Gate(p1=np.array([x[i] + px * half_width_m, y[i] + py * half_width_m]),
               p2=np.array([x[i] - px * half_width_m, y[i] - py * half_width_m]))


def ref_tangent(cd, x, y, dist_m, sigma=10.0, span=25.0, half=10.0):
    """Independent smoothed-tangent reference: Gaussian-smoothed positions with a
    ±10 m central difference (the v2 / consistency_check method)."""
    def sm(s):
        m = np.abs(cd - s) <= span
        w = np.exp(-0.5 * ((cd[m] - s) / sigma) ** 2)
        return np.sum(w * x[m]) / w.sum(), np.sum(w * y[m]) / w.sum()
    x1, y1 = sm(dist_m + half)
    x0, y0 = sm(dist_m - half)
    return x1 - x0, y1 - y0


def tangent_of(gate):
    d = gate.p1 - gate.p2            # gate segment is perpendicular to the tangent
    return d[1], -d[0]


def acute_deg(vx, vy, wx, wy):
    vn, wn = math.hypot(vx, vy), math.hypot(wx, wy)
    return math.degrees(math.acos(min(1.0, abs((vx * wx + vy * wy) / (vn * wn)))))


def crossing(lat, lon, t, td, gate, frame, seed):
    m = np.abs(td - seed) <= 160.0
    idx = np.where(m)[0]
    if len(idx) < 2:
        return None
    i0, i1 = max(0, idx.min() - 1), min(len(t), idx.max() + 2)
    return gate_crossing_time(lat[i0:i1], lon[i0:i1], t[i0:i1], td[i0:i1], gate, frame, seed)


def main() -> int:
    sess_dir = sessions_dir(TRACK)
    if not sess_dir.exists():
        print(f"real session data not found at {sess_dir} — run locally with data/ present")
        return 2

    cl = load_centerline(TRACK)
    frame = TrackFrame.from_centerline(cl)
    bounds = section_bounds(json.loads((ROOT / "tracks" / "ridge.json").read_text()))
    cd = cl["track_dist_m"].to_numpy()
    cx, cy = frame.to_xy(cl["lat"].to_numpy(), cl["long"].to_numpy())
    gate_dists = sorted({d for ab in bounds.values() for d in ab})
    label = {}
    for cid, (a, b) in bounds.items():
        label.setdefault(a, f"{cid}-entry")
        label.setdefault(b, f"{cid}-exit")

    # ---- Part A: gate rotation vs smoothed-tangent reference ----
    old_rot, new_rot, worst = [], [], []
    for d in gate_dists:
        rx, ry = ref_tangent(cd, cx, cy, d)
        ox, oy = tangent_of(build_gate_2sample(cd, cx, cy, d))
        nx, ny = tangent_of(build_gate(cl, d, frame))
        ro, rn = acute_deg(ox, oy, rx, ry), acute_deg(nx, ny, rx, ry)
        old_rot.append(ro)
        new_rot.append(rn)
        worst.append((ro, label.get(d, f"{d:.0f}m")))
    old_rot, new_rot = np.array(old_rot), np.array(new_rot)
    new_p90 = float(np.percentile(new_rot, 90))
    print("=== Part A: gate rotation vs smoothed-tangent reference (32 corner gates) ===")
    print(f"OLD 2-sample:  median {np.median(old_rot):5.1f}  p90 {np.percentile(old_rot,90):5.1f}"
          f"  max {old_rot.max():5.1f}   (baseline ~9 / 46 / 59)")
    print(f"NEW smoothed:  median {np.median(new_rot):5.1f}  p90 {new_p90:5.1f}"
          f"  max {new_rot.max():5.1f}   (GATE p90 < {ROTATION_P90_GATE_DEG:.0f})")
    print("worst OLD:", ", ".join(f"{n} {r:.0f}deg" for r, n in sorted(worst, reverse=True)[:4]))

    # ---- Part B: driven-distance tails, same pipeline, swap gate ----
    gates_old = {d: build_gate_2sample(cd, cx, cy, d) for d in gate_dists}
    gates_new = {d: build_gate(cl, d, frame) for d in gate_dists}
    rows_old, rows_new = [], []
    for sess in sorted(sess_dir.iterdir()):
        f = sess / "samples.parquet"
        if not f.exists():
            continue
        try:
            df = pd.read_parquet(f, columns=["lap", "t", "dist_lap_m", "track_dist_m", "lat", "long"])
        except Exception:
            continue
        if df["track_dist_m"].isna().all():
            continue
        for _lap, g in df.groupby("lap"):
            g = g.dropna(subset=["t", "dist_lap_m", "track_dist_m", "lat", "long"]).sort_values("t")
            if len(g) < 100:
                continue
            t = g["t"].to_numpy(float)
            dl = g["dist_lap_m"].to_numpy(float)
            td = g["track_dist_m"].to_numpy(float)
            lat = g["lat"].to_numpy(float)
            lon = g["long"].to_numpy(float)
            dev = td - dl
            for cid, (a, b) in bounds.items():
                if td.min() > a - 20 or td.max() < b + 20:
                    continue
                near = (np.abs(td - a) <= 150.0) | (np.abs(td - b) <= 150.0)
                if not near.any() or np.nanmax(np.abs(dev[near])) > 40.0:
                    continue
                for gates, rows in ((gates_old, rows_old), (gates_new, rows_new)):
                    ta = crossing(lat, lon, t, td, gates[a], frame, a)
                    tb = crossing(lat, lon, t, td, gates[b], frame, b)
                    if ta is None or tb is None or tb <= ta:
                        continue
                    rows.append((cid, float(np.interp(tb, t, dl) - np.interp(ta, t, dl))))

    def tails(rows):
        df = pd.DataFrame(rows, columns=["corner", "driven"])
        return {cid: (-(g["driven"] - g["driven"].median()).quantile(0.02),
                      (g["driven"] - g["driven"].median()).quantile(0.98))
                for cid, g in df.groupby("corner")}

    TO, TN = tails(rows_old), tails(rows_new)
    v2 = pd.read_csv(ARTIFACTS / "corner_bounds_verdict.csv").set_index("corner")
    print("\n=== Part B: driven-distance p2/p98 tails (old 2-sample -> new smoothed) ===")
    print(f"{'corner':7}{'old tlo/thi':>14}{'new tlo/thi':>14}{'tighten':>13}{'csv v2':>13}")
    for cid in ["T14", "T10", "T3", "T4", "T8", "T1", "T6", "T11", "T15"]:
        if cid not in TO or cid not in TN:
            continue
        olo, ohi = TO[cid]
        nlo, nhi = TN[cid]
        rl, rh = olo / max(nlo, 1e-6), ohi / max(nhi, 1e-6)
        cv = f"{v2.loc[cid].v2_tlo:.1f}/{v2.loc[cid].v2_thi:.1f}"
        print(f"{cid:7}{olo:6.1f}/{ohi:6.1f}{nlo:7.1f}/{nhi:6.1f}{rl:6.1f}x/{rh:4.1f}x{cv:>13}")

    ok = new_p90 < ROTATION_P90_GATE_DEG
    print(f"\nGATE rotation p90 < {ROTATION_P90_GATE_DEG:.0f}deg -> "
          f"{'PASS' if ok else 'FAIL'} ({new_p90:.1f}deg)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
