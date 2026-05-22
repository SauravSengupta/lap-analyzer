"""Sanity-check the corpus rebuilt with the new corner bounds."""
from __future__ import annotations

from lap_analyzer.analysis import load_corpus

c = load_corpus("ridge")
print(f"transits {len(c)}   laps {c[['session_id', 'lap']].drop_duplicates().shape[0]}"
      f"   corners {len(c.corner_id.unique())}")
print(f"reliable transits {int(c.transit_reliable.sum())}/{len(c)} "
      f"({100 * c.transit_reliable.mean():.0f}%)   NaN entry_speed {int(c.entry_speed_mph.isna().sum())}")
print()
print(f"{'corner':7}{'entry':>9}{'min':>9}{'apex':>9}{'exit':>9}")
for cid in ["T1", "T5", "T6", "T11", "T12", "T13"]:
    s = c[c.corner_id == cid]
    print(f"{cid:7}{s.entry_speed_mph.median():9.1f}{s.min_speed_mph.median():9.1f}"
          f"{s.apex_speed_mph.median():9.1f}{s.exit_speed_mph.median():9.1f}")
print()
print("T1 entry was ~86 mph before (box started mid-braking); should now be ~120 (off-throttle).")
print("T1 exit should no longer equal T1 min (box no longer ends on the slowest point).")
