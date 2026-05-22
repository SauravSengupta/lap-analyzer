"""Seed tracks/pir.json's `corners[]` and `calibration_anchors[]` from
candidate clusters + user-pinned apex coords.

Default: dry-run report only. Pass --write to mutate tracks/pir.json.

Algorithm:
  1. Project each user pin's GPS onto sample-lap trajectories to derive its
     lap-distance (median across several clean laps; precision ~3-6m).
  2. Cluster the candidate `peak_dist_m` values per direction (R/L) using
     density-peak detection — this captures the per-lat-G-peak position
     reliably, even within sustained-G regions like the Festival Curves
     chicane where multiple peaks share a single window's apex_lat/long.
  3. Map each user pin to the nearest direction-matching cluster.
  4. For T4: primary apex = T4b (user's stated "real" apex, matched to its
     own cluster), secondary = T4a's cluster. One corner total.
  5. For T8 / T9: pinned position retained even when the lat-G cluster is
     weak or missing (user noted "full throttle through this corner — apex
     more nominal than driven"). T9 in particular has no nearby cluster;
     using pin-projected lap-distance directly.
  6. calibration_anchors = user pin coords (visual / kerb-touch) — same
     convention as Ridge. Excludes T4a (per user note: "throwaway / rarely
     hit"; would skew drift correction).
  7. start_m / end_m = apex_m +/- 80m (labeler refines later via
     compute_corner_bounds_from_data).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# Corner directions in lap order (clockwise PIR). Source: user statement
# 2026-05-20 (T1-T4 explicit; T5-T12 derived from data and confirmed by
# matching pin positions to clusters of the expected direction).
EXPECTED = [
    ("T1",  "R"),
    ("T2",  "L"),
    ("T3",  "R"),
    ("T4",  "R"),  # primary apex = T4b; secondary = T4a
    ("T5",  "R"),
    ("T6",  "L"),
    ("T7",  "R"),
    ("T8",  "L"),  # weak / full-throttle
    ("T9",  "R"),  # weak / full-throttle (no cluster — use pin-projected)
    ("T10", "L"),
    ("T11", "R"),
    ("T12", "R"),
]

# Pins that anchor each numbered corner for calibration purposes.
# Default: same name as corner. T4 overridden to use T4b ("real apex").
ANCHOR_PIN = {"T4": "T4b"}

# Corner that takes the pin-projected lap-distance directly (no candidate
# cluster within search radius).
USE_PIN_LAP_DIST = {"T9"}


REPO_ROOT = Path(__file__).resolve().parents[2]
PINS_PATH = REPO_ROOT / "data" / "notes" / "pir_apex_pins.json"
CANDIDATES_PATH = REPO_ROOT / "data" / "corpus" / "pir_candidates.csv"
SESSIONS_DIR = REPO_ROOT / "data" / "sessions" / "pir"
TRACK_PATH = REPO_ROOT / "app" / "tracks" / "pir.json"


def hav(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def project_pin_to_lap_distance(pin, sessions, n_laps_per_session=3, max_match_m=50.0):
    """Median lap-distance across multiple clean laps where the pin GPS
    has a nearest-neighbor sample within max_match_m."""
    vals = []
    for sp in sessions:
        df = pd.read_parquet(sp, columns=["lap", "dist_lap_m", "lat", "long"])
        for lap_num in df["lap"].unique()[:n_laps_per_session]:
            lap = df[df["lap"] == lap_num]
            if lap["dist_lap_m"].max() < 3000:
                continue
            d = hav(lap["lat"].values, lap["long"].values, pin["lat"], pin["long"])
            min_idx = d.argmin()
            if d[min_idx] < max_match_m:
                vals.append(float(lap["dist_lap_m"].iloc[min_idx]))
    if not vals:
        return None, None
    return int(round(float(np.median(vals)))), int(round(float(np.std(vals))))


def discover_clusters(cand, n_laps, bin_m=5.0, min_prom_frac=0.05, win=6):
    """Find density peaks in peak_dist_m, separately for R and L candidates."""
    edges = np.arange(0, 3500 + bin_m, bin_m)
    centers = (edges[:-1] + edges[1:]) / 2
    out = []
    for label, sub in [("R", cand[cand["peak_lat_g"] > 0]),
                       ("L", cand[cand["peak_lat_g"] < 0])]:
        hist, _ = np.histogram(sub["peak_dist_m"], bins=edges)
        smooth = np.convolve(hist, np.ones(win) / win, mode="same")
        prom = max(2, n_laps * min_prom_frac)
        pks, _ = find_peaks(smooth, distance=10, prominence=prom)
        for idx in pks:
            center_m = centers[idx]
            win_cand = sub[(sub["peak_dist_m"] >= center_m - 30)
                           & (sub["peak_dist_m"] <= center_m + 30)]
            if len(win_cand) < 5:
                continue
            out.append({
                "dir": label,
                "apex_m": int(round(float(win_cand["peak_dist_m"].median()))),
                "n": int(len(win_cand)),
                "per_lap": len(win_cand) / n_laps,
                "lat_g_med": float(win_cand["peak_lat_g"].median()),
                "speed_med": float(win_cand["min_speed_mph"].median()),
            })
    return sorted(out, key=lambda c: c["apex_m"])


def map_pin_to_cluster(pin_lap_dist, pin_dir, clusters, max_gap_m=80.0):
    """Pick the direction-matching cluster nearest to the pin's lap-distance.
    Returns None if no cluster within max_gap_m."""
    candidates = [c for c in clusters if c["dir"] == pin_dir]
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(c["apex_m"] - pin_lap_dist))
    if abs(best["apex_m"] - pin_lap_dist) > max_gap_m:
        return None
    return best


def build_corners_and_anchors(pins, sessions, candidates, n_laps):
    clusters = discover_clusters(candidates, n_laps)
    pin_dists = {}
    for cid, pin in pins.items():
        d, _ = project_pin_to_lap_distance(pin, sessions)
        pin_dists[cid] = d

    corners = []
    anchors = []
    report_rows = []

    for cid, direction in EXPECTED:
        anchor_pin_id = ANCHOR_PIN.get(cid, cid)
        pin = pins[anchor_pin_id]
        pin_d = pin_dists[anchor_pin_id]

        # Cluster mapping (skipped for USE_PIN_LAP_DIST corners)
        cluster = None
        if cid not in USE_PIN_LAP_DIST:
            cluster = map_pin_to_cluster(pin_d, direction, clusters)

        if cluster is not None:
            apex_m = cluster["apex_m"]
            src = f"cluster (n={cluster['n']}, latG={cluster['lat_g_med']:+.2f})"
        else:
            apex_m = pin_d
            src = "pin-projected (no cluster nearby)"

        # T4: pull secondary apex from T4a cluster
        secondary_apex_m = None
        secondary_note = None
        if cid == "T4":
            t4a_pin = pins["T4a"]
            t4a_d = pin_dists["T4a"]
            t4a_cluster = map_pin_to_cluster(t4a_d, "R", clusters)
            if t4a_cluster is not None and abs(t4a_cluster["apex_m"] - apex_m) > 30:
                secondary_apex_m = t4a_cluster["apex_m"]
                secondary_note = (f"T4a 'throwaway' apex (user rarely hits); "
                                  f"data shows {t4a_cluster['n']} passes per "
                                  f"{n_laps} flying laps")

        corner = {
            "id": cid,
            "name": None,
            "type": "right" if direction == "R" else "left",
            "start_m": apex_m - 80,
            "end_m": apex_m + 80,
            "apex_m": apex_m,
            "notes": None,
        }
        if secondary_apex_m is not None:
            corner["secondary_apex_m"] = secondary_apex_m
            corner["notes"] = secondary_note
        corners.append(corner)

        anchor = {
            "corner_id": cid,
            "ref_lat": round(pin["lat"], 5),
            "ref_long": round(pin["long"], 5),
            "confidence": "high" if cluster is not None else "low",
        }
        if pin.get("note"):
            anchor["notes"] = pin["note"]
        anchors.append(anchor)

        report_rows.append({
            "corner": cid, "dir": direction, "pin_d": pin_d,
            "apex_m": apex_m, "source": src,
            "secondary_m": secondary_apex_m,
        })

    return corners, anchors, report_rows, clusters


def print_report(rows, clusters, n_laps):
    print(f"\n=== {n_laps} flying laps, {len(clusters)} discovered clusters ===\n")
    print(f"{'corner':<8} {'dir':>3} {'pin_d':>6} {'apex_m':>7} {'sec_m':>6}  source")
    print("-" * 80)
    for r in rows:
        sec = str(r["secondary_m"]) if r["secondary_m"] else "-"
        print(f"{r['corner']:<8} {r['dir']:>3} {r['pin_d']:>6} {r['apex_m']:>7} {sec:>6}  {r['source']}")
    print()
    # Unused clusters (not mapped to any corner)
    used = {r["apex_m"] for r in rows} | {r["secondary_m"] for r in rows if r["secondary_m"]}
    leftover = [c for c in clusters if c["apex_m"] not in used
                and not any(abs(c["apex_m"] - u) < 30 for u in used)]
    if leftover:
        print("Discovered clusters NOT mapped to any corner (informational):")
        for c in leftover:
            print(f"  {c['dir']} @ {c['apex_m']}m  n={c['n']}  latG={c['lat_g_med']:+.2f}  speed={c['speed_med']:.0f}mph")
        print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true",
                   help="Mutate tracks/pir.json. Default: dry-run report only.")
    args = p.parse_args(argv)

    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    pins = {k: v for k, v in pins.items() if not k.startswith("_")}

    candidates = pd.read_csv(CANDIDATES_PATH)
    n_laps = candidates.groupby(["session_id", "lap"]).ngroups

    sessions = sorted(SESSIONS_DIR.glob("*/samples.parquet"))[:5]

    corners, anchors, report_rows, clusters = build_corners_and_anchors(
        pins, sessions, candidates, n_laps)

    print_report(report_rows, clusters, n_laps)

    if not args.write:
        print("[dry-run] No files modified. Re-run with --write to update tracks/pir.json.")
        return 0

    track = json.loads(TRACK_PATH.read_text(encoding="utf-8"))
    track["corners"] = corners
    track["calibration_anchors"] = anchors
    TRACK_PATH.write_text(json.dumps(track, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(corners)} corners and {len(anchors)} calibration_anchors to {TRACK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
