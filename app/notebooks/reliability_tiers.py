"""Show what 'reliable' looks like at different filter aggressiveness levels.

Outputs three views:
  1. Per-transit reliability: how many transits qualify at each tier
  2. Per-lap reliability: laps where ALL transits qualify
  3. Per-corner reliability rate: what fraction of corpus laps are reliable per corner

Three tiers proposed:
  STRICT     — drift disagreement <= 15m, transit offset <= 30m, both neighbors <= 30m
  STANDARD   — drift disagreement <= 20m, transit offset <= 40m, both neighbors <= 40m
  PERMISSIVE — drift disagreement <= 30m, transit offset <= 50m, no neighbor check
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
NOTES = json.loads(Path(r"D:\Projects\lap-analyzer\data\notes\ridge.json").read_text())
EXCLUDED = {sid for sid, m in NOTES.items() if "exclude" in m}
GPS_BAD = {sid for sid, m in NOTES.items() if m.get("flag") == "gps_unreliable"}

# Load corner sequence from ridge.json to know neighbors
track = json.loads(Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.json").read_text(encoding="utf-8"))
corner_order = [c["id"] for c in track["corners"]]
prev_neighbor = {corner_order[i]: corner_order[(i - 1) % len(corner_order)] for i in range(len(corner_order))}
next_neighbor = {corner_order[i]: corner_order[(i + 1) % len(corner_order)] for i in range(len(corner_order))}

# Load all transits + laps
transits = pd.concat(
    [pd.read_parquet(p) for p in ROOT.rglob("corners.parquet")], ignore_index=True
)
transits = transits[~transits.session_id.isin(EXCLUDED | GPS_BAD)]

laps = pd.concat(
    [pd.read_csv(p) for p in ROOT.rglob("laps.csv")], ignore_index=True
)
clean = laps[laps.is_clean.astype(bool)]
clean = clean[~clean.session_id.isin(EXCLUDED | GPS_BAD)]
print(f"Corpus after session-level exclusions: {len(transits)} transits, "
      f"{clean.groupby(['session_id','lap']).ngroups} clean laps, "
      f"{clean.session_id.nunique()} sessions")

# Index per-transit offset for neighbor lookups
idx = transits.set_index(["session_id", "lap", "corner_id"])["track_dist_offset_max_m"].to_dict()

def neighbor_off(sid, lap, cid):
    """Max offset across this corner and its two neighbors. NaN if any neighbor missing."""
    own = idx.get((sid, int(lap), cid))
    prev_off = idx.get((sid, int(lap), prev_neighbor[cid]))
    next_off = idx.get((sid, int(lap), next_neighbor[cid]))
    if own is None or prev_off is None or next_off is None:
        return np.nan
    return max(own, prev_off, next_off)

transits["neighborhood_offset_max_m"] = transits.apply(
    lambda r: neighbor_off(r.session_id, r.lap, r.corner_id), axis=1
)

tiers = {
    "STRICT":     dict(disagree=15, transit=30, neighborhood=30),
    "STANDARD":   dict(disagree=20, transit=40, neighborhood=40),
    "PERMISSIVE": dict(disagree=30, transit=50, neighborhood=None),
}

print()
for name, t in tiers.items():
    mask = (
        (transits.gps_drift_disagreement_m <= t["disagree"])
        & (transits.track_dist_offset_max_m <= t["transit"])
    )
    if t["neighborhood"] is not None:
        mask &= transits.neighborhood_offset_max_m <= t["neighborhood"]
    n_reliable = mask.sum()
    print(f"=== Tier: {name}  (disagree<={t['disagree']}, transit<={t['transit']}, "
          f"neighborhood<={t['neighborhood']}) ===")
    print(f"  Reliable transits: {n_reliable} / {len(transits)} ({100*n_reliable/len(transits):.1f}%)")

    # Per-corner reliability rate
    print(f"  Per-corner reliable-transit rate:")
    by_corner = transits.groupby("corner_id").agg(
        total=("corner_id", "size"),
        reliable=("track_dist_offset_max_m", lambda x: mask.loc[x.index].sum()),
    )
    by_corner["rate"] = by_corner["reliable"] / by_corner["total"]
    for cid in sorted(by_corner.index, key=lambda s: int(s[1:])):
        r = by_corner.loc[cid]
        bar = "#" * int(r["rate"] * 30)
        print(f"    {cid:<5s}  {int(r['reliable']):>3d}/{int(r['total']):>3d} = "
              f"{r['rate']:.0%}  {bar}")

    # Per-lap: lap is reliable if ALL its corners pass the filter
    lap_reliable = transits[mask].groupby(["session_id", "lap"]).size()
    total_per_lap = transits.groupby(["session_id", "lap"]).size()
    full_pass = lap_reliable[lap_reliable == total_per_lap.loc[lap_reliable.index]]
    print(f"  Fully-reliable laps (all corners pass): {len(full_pass)} / "
          f"{len(total_per_lap)} ({100*len(full_pass)/len(total_per_lap):.1f}%)")
    print()
