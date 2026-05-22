"""Apply proposed corner bounds to tracks/ridge.json.

Backs up the original to ridge.json.bak first. Updates start_m/end_m for every
corner from proposed_corner_bounds.json; apex_m and everything else untouched.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from lap_analyzer.config import tracks_dir

rj = tracks_dir() / "ridge.json"
bak = rj.parent / "ridge.json.bak"
proposed_path = Path(__file__).parent / "proposed_corner_bounds.json"

shutil.copy(rj, bak)
print(f"backed up -> {bak}")

td = json.loads(rj.read_text(encoding="utf-8"))
proposed = json.loads(proposed_path.read_text(encoding="utf-8"))["bounds"]

print(f"\n{'corner':8}{'old':>16}{'new':>16}")
for c in td["corners"]:
    cid = c["id"]
    if cid not in proposed:
        print(f"{cid:8}  (no proposal — left unchanged)")
        continue
    o0, o1 = c["start_m"], c["end_m"]
    n0, n1 = proposed[cid]["start_m"], proposed[cid]["end_m"]
    c["start_m"], c["end_m"] = n0, n1
    print(f"{cid:8}{f'[{o0:.0f},{o1:.0f}]':>16}{f'[{n0:.0f},{n1:.0f}]':>16}")

rj.write_text(json.dumps(td, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {rj}")
