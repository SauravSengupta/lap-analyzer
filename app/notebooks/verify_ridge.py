"""Verify ridge.json against the handoff's checklist."""
import json
from pathlib import Path

JSON_PATH = Path(__file__).resolve().parents[1] / "tracks" / "ridge.json"
data = json.loads(JSON_PATH.read_text(encoding="utf-8"))

errors = []
warnings = []

corners = data["corners"]
print(f"Track: {data['name']} ({data['track_id']})")
print(f"Direction: {data['direction']}, length {data['official_length_m']} m (internal {data['lap_length_internal_m']} m)")
print(f"Corners: {len(corners)}")

# 16 corners
if len(corners) != 16:
    errors.append(f"Expected 16 corners, got {len(corners)}")

# IDs T1..T16 in order
expected_ids = [f"T{i}" for i in range(1, 17)]
actual_ids = [c["id"] for c in corners]
if actual_ids != expected_ids:
    errors.append(f"ID sequence wrong: {actual_ids}")

# Monotonic increasing apex_m and start_m
for i, c in enumerate(corners[1:], 1):
    prev = corners[i - 1]
    if c["start_m"] < prev["start_m"]:
        errors.append(f"{c['id']} start_m {c['start_m']} < {prev['id']} start_m {prev['start_m']}")
    if c["apex_m"] <= prev["apex_m"]:
        errors.append(f"{c['id']} apex_m {c['apex_m']} <= {prev['id']} apex_m {prev['apex_m']}")

# No overlaps (allow tiny touch)
for i, c in enumerate(corners[1:], 1):
    prev = corners[i - 1]
    if c["start_m"] < prev["end_m"]:
        overlap = prev["end_m"] - c["start_m"]
        if overlap > 0.5:
            warnings.append(f"{prev['id']}.end={prev['end_m']} overlaps {c['id']}.start={c['start_m']} by {overlap:.1f} m")

# Apex inside [start, end]
for c in corners:
    if not (c["start_m"] <= c["apex_m"] <= c["end_m"]):
        errors.append(f"{c['id']} apex_m {c['apex_m']} outside [{c['start_m']}, {c['end_m']}]")
    if "secondary_apex_m" in c and c["secondary_apex_m"] is not None:
        if not (c["start_m"] <= c["secondary_apex_m"] <= c["end_m"]):
            errors.append(f"{c['id']} secondary_apex_m {c['secondary_apex_m']} outside [{c['start_m']}, {c['end_m']}]")

# T1 starts at 300-500m
if not (300 <= corners[0]["start_m"] <= 500):
    warnings.append(f"T1 starts at {corners[0]['start_m']} m (handoff suggests 300-500)")

# T16 ends well before 3962
if corners[-1]["end_m"] >= data["lap_length_internal_m"] - 100:
    warnings.append(f"T16 ends at {corners[-1]['end_m']} m, only {data['lap_length_internal_m'] - corners[-1]['end_m']:.0f} m before lap end (front straight too short?)")

# Print full table
print(f"\n{'id':4s}  {'name':<10s}  {'start':>7s}  {'apex':>7s}  {'apex2':>7s}  {'end':>7s}  {'type':>5s}")
for c in corners:
    sec = c.get("secondary_apex_m")
    sec_str = f"{sec:7.0f}" if sec else "       "
    name_str = c.get("name") or ""
    print(f"{c['id']:4s}  {name_str:<10s}  {c['start_m']:7.0f}  {c['apex_m']:7.0f}  {sec_str}  {c['end_m']:7.0f}  {c['type']:>5s}")

print(f"\nFront-straight gap: {corners[0]['start_m']:.0f} m before T1, {data['lap_length_internal_m'] - corners[-1]['end_m']:.0f} m after T16 (total {corners[0]['start_m'] + data['lap_length_internal_m'] - corners[-1]['end_m']:.0f} m)")

if errors:
    print("\n=== ERRORS ===")
    for e in errors:
        print(f"  ✗ {e}")
if warnings:
    print("\n=== WARNINGS ===")
    for w in warnings:
        print(f"  ! {w}")
if not errors and not warnings:
    print("\n=== ALL CHECKS PASSED ===")
elif not errors:
    print("\n=== No errors. Warnings only. ===")
