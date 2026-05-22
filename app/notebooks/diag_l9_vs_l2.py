"""L9 (selected) vs L2 (section-best) through T1-T5 — per-corner breakdown."""
from __future__ import annotations

import json

from lap_analyzer.analysis import load_corpus, section_times
from lap_analyzer.config import tracks_dir

TRACK, SID, SEL, BEST = "ridge", "20260517-100304", 9, 2
td = json.loads((tracks_dir() / f"{TRACK}.json").read_text(encoding="utf-8"))
T15 = ["T1", "T2", "T3", "T4", "T5"]

st = section_times(TRACK, td)
st = st[(st["session_id"] == SID) & st["corner_id"].isin(T15)]
sel = st[st["lap"] == SEL].set_index("corner_id")["section_time_s"]
bst = st[st["lap"] == BEST].set_index("corner_id")["section_time_s"]

print("per-corner section time (s):")
print(f"{'corner':7}{'L9 sel':>9}{'L2 best':>9}{'delta':>9}")
tot = 0.0
for c in T15:
    d = sel.get(c, float('nan')) - bst.get(c, float('nan'))
    tot += d if d == d else 0.0
    print(f"{c:7}{sel.get(c, float('nan')):9.3f}{bst.get(c, float('nan')):9.3f}{d:>+9.3f}")
print(f"{'TOTAL':7}{sel.reindex(T15).sum():9.3f}{bst.reindex(T15).sum():9.3f}{tot:>+9.3f}")

corpus = load_corpus(TRACK)
cc = corpus[(corpus["session_id"] == SID) & corpus["corner_id"].isin(T15)]
print("\nper-corner speeds (mph) — L9 sel / L2 best:")
print(f"{'corner':7}{'entry':>14}{'min':>14}{'exit':>14}{'max_latg':>16}")
for c in T15:
    r9 = cc[(cc["lap"] == SEL) & (cc["corner_id"] == c)]
    r2 = cc[(cc["lap"] == BEST) & (cc["corner_id"] == c)]
    if r9.empty or r2.empty:
        continue
    r9, r2 = r9.iloc[0], r2.iloc[0]
    print(f"{c:7}"
          f"{f'{r9.entry_speed_mph:.1f}/{r2.entry_speed_mph:.1f}':>14}"
          f"{f'{r9.min_speed_mph:.1f}/{r2.min_speed_mph:.1f}':>14}"
          f"{f'{r9.exit_speed_mph:.1f}/{r2.exit_speed_mph:.1f}':>14}"
          f"{f'{r9.max_lat_g:.2f}/{r2.max_lat_g:.2f}':>16}")
