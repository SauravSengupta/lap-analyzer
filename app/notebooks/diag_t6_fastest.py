"""Why does 'Fastest through T6' pick L2 (15.38s) when L9 ran 15.00s?

Reproduces app.py's find_best_lap eligibility filtering for the T6 section and
shows exactly which gate drops L9.
"""
from __future__ import annotations

import json

from lap_analyzer.analysis import (load_all_laps, load_corpus,
                                   range_section_times, section_range_bounds)
from lap_analyzer.config import tracks_dir

TRACK = "ridge"
L9 = ("20260517-100304", 9)
td = json.loads((tracks_dir() / f"{TRACK}.json").read_text(encoding="utf-8"))

rst = range_section_times(TRACK, td, "T6", "T6")
print("range_section_times columns:", list(rst.columns))

corpus = load_corpus(TRACK)
laps_csv = load_all_laps(TRACK)
clean = set(map(tuple, laps_csv[laps_csv["is_clean"].fillna(False).astype(bool)]
                [["session_id", "lap"]].astype({"lap": int})
                .itertuples(index=False, name=None)))
t6 = corpus[corpus["corner_id"] == "T6"]
reliable = {(r.session_id, int(r.lap)) for r in t6.itertuples(index=False) if r.transit_reliable}

a, b = section_range_bounds(td, "T6", "T6")
ref_bound = max(8.0, 0.02 * (b - a))
print(f"T6 section bounds [{a:.0f},{b:.0f}]  ref obd-discrepancy bound = +/-{ref_bound:.1f} m\n")

# L9 status at each gate
r9 = rst[(rst["session_id"] == L9[0]) & (rst["lap"] == L9[1])]
print(f"=== L9 {L9[0]} L{L9[1]} ===")
if r9.empty:
    print("  NOT in range_section_times at all (no clean T6 crossing).")
else:
    r9 = r9.iloc[0]
    print(f"  section_time_s   = {r9['section_time_s']}")
    print(f"  obd_discrepancy_m= {r9.get('obd_discrepancy_m', 'n/a')}")
    print(f"  is_clean         = {L9 in clean}")
    print(f"  T6 reliable      = {L9 in reliable}")
    od = r9.get("obd_discrepancy_m")
    if od is not None:
        print(f"  passes obd gate  = {abs(od) <= ref_bound}  (|{od:.1f}| vs {ref_bound:.1f})")

# Reproduce find_best_lap's pick — current (absolute) gate vs proposed (median-relative).
elig_keys = reliable & clean
elig = rst[rst.set_index(["session_id", "lap"]).index.isin(elig_keys)].copy()
print(f"\neligible (clean & T6-reliable): {len(elig)} laps")
med_disc = elig["obd_discrepancy_m"].median()
print(f"median obd_discrepancy over eligible laps: {med_disc:+.1f} m  "
      f"(systematic, not a glitch)")

old_gate = elig[elig["obd_discrepancy_m"].abs() <= ref_bound]
new_gate = elig[(elig["obd_discrepancy_m"] - med_disc).abs() <= ref_bound]
print(f"\nCURRENT gate |disc| <= {ref_bound:.1f}:  {len(old_gate)}/{len(elig)} pass  "
      f"-> picks {old_gate.sort_values('section_time_s').iloc[0]['session_id']} "
      f"L{int(old_gate.sort_values('section_time_s').iloc[0]['lap'])} "
      f"@ {old_gate['section_time_s'].min():.2f}s")
print(f"PROPOSED gate |disc - median| <= {ref_bound:.1f}:  {len(new_gate)}/{len(elig)} pass  "
      f"-> picks {new_gate.sort_values('section_time_s').iloc[0]['session_id']} "
      f"L{int(new_gate.sort_values('section_time_s').iloc[0]['lap'])} "
      f"@ {new_gate['section_time_s'].min():.2f}s")

# Who actually has the fastest T6 section time, ignoring all gates?
fastest = rst.sort_values("section_time_s").head(8)
print("\nfastest 8 T6 section times overall (no gates):")
for r in fastest.itertuples(index=False):
    key = (r.session_id, int(r.lap))
    od = getattr(r, "obd_discrepancy_m", float("nan"))
    print(f"  {r.session_id} L{int(r.lap):<3d} {r.section_time_s:7.2f}s  "
          f"clean={key in clean}  reliable={key in reliable}  obd_disc={od:+.1f}")

# Is L9's -20m discrepancy a real shorter line (smooth offset) or a GPS glitch
# (jittery offset)? Compare a few of the fast laps.
from lap_analyzer.analysis import load_samples
import pandas as pd
print("\noffset (track_dist_m - dist_lap_m) over the T6 section — smooth=real line, jumpy=glitch:")
for sid, lap in [("20260517-100304", 9), ("20250307-110826", 3), ("20260516-100504", 2)]:
    s = load_samples(TRACK, sid, lap).sort_values("t")
    s = s[(s["track_dist_m"] >= a) & (s["track_dist_m"] <= b)]
    off = (s["track_dist_m"] - s["dist_lap_m"]).to_numpy()
    if len(off) < 5:
        print(f"  {sid} L{lap}: too few samples"); continue
    smooth = pd.Series(off).rolling(15, center=True, min_periods=1).median().to_numpy()
    jitter = float(((off - smooth) ** 2).mean() ** 0.5)
    print(f"  {sid} L{lap}: offset {off[0]:+.1f} -> {off[-1]:+.1f} m (swing {off[-1]-off[0]:+.1f}), "
          f"jitter RMS {jitter:.2f} m")
