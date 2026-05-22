"""Find the cleanest reference-lap candidate.

For each clean lap in the corpus, compute closest-sample distance to each Maps
pin. Two views:

  Option 1 — replace the reference lap wholesale:
    Find the lap with the lowest MAX distance across all pins. That lap is
    closest to ground truth everywhere we have a pin.

  Option 2 — splice the inner loop only:
    Find the lap with the lowest MAX distance across just T7 and T9 (the
    glitched-on-May-1 pins). That lap is the best candidate for the spliced
    T7-T9 segment. Bonus if it's also clean elsewhere — useful as fallback.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))

PINS_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge_apex_pins.json")
TRACK_PATH = Path(r"D:\Projects\lap-analyzer\app\tracks\ridge.json")
SESSIONS = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
NOTES_PATH = Path(r"D:\Projects\lap-analyzer\data\notes\ridge.json")

pins = {k: v for k, v in json.loads(PINS_PATH.read_text(encoding="utf-8")).items() if not k.startswith("_")}
track = json.loads(TRACK_PATH.read_text(encoding="utf-8"))
corner_ranges = {c["id"]: (c["start_m"], c["end_m"]) for c in track["corners"]}

notes = json.loads(NOTES_PATH.read_text(encoding="utf-8"))
excluded = {sid for sid, m in notes.items() if "exclude" in m}

laps_idx = pd.concat([pd.read_csv(p) for p in SESSIONS.rglob("laps.csv")], ignore_index=True)
clean_laps = laps_idx[laps_idx.is_clean.astype(bool) & ~laps_idx.session_id.isin(excluded)]
print(f"Scanning {len(clean_laps)} clean laps across {clean_laps.session_id.nunique()} sessions...\n")


def closest_in_lap(samples: pd.DataFrame, lap: int, pin_lat: float, pin_lon: float):
    sub = samples[samples["lap"] == lap]
    if len(sub) == 0:
        return None, None
    dlat_m = (sub["lat"].values - pin_lat) * M_PER_DEG_LAT
    dlon_m = (sub["long"].values - pin_lon) * M_PER_DEG_LON
    d = np.sqrt(dlat_m ** 2 + dlon_m ** 2)
    idx = int(d.argmin())
    return float(d[idx]), float(sub.iloc[idx]["track_dist_m"])


# Build per-(lap, pin) distance matrix
records = []
for sid in clean_laps.session_id.unique():
    sp = SESSIONS / sid / "samples.parquet"
    if not sp.exists():
        continue
    s = pd.read_parquet(sp)
    for ln in clean_laps[clean_laps.session_id == sid].lap.unique():
        rec = {"session_id": sid, "lap": int(ln)}
        for pin_id, p in pins.items():
            d, td = closest_in_lap(s, int(ln), p["lat"], p["long"])
            rec[f"{pin_id}_m"] = d
            rec[f"{pin_id}_td"] = td
        records.append(rec)

df = pd.DataFrame(records)
df = df.merge(clean_laps[["session_id", "lap", "lap_time_s"]], on=["session_id", "lap"], how="left")

pin_cols = [f"{p}_m" for p in pins.keys()]
inner_cols = [f"{p}_m" for p in ["T7", "T9"]]

# Distance summary across all pins per lap
df["max_all_m"] = df[pin_cols].max(axis=1)
df["sum_all_m"] = df[pin_cols].sum(axis=1)
df["max_inner_m"] = df[inner_cols].max(axis=1)
df["sum_inner_m"] = df[inner_cols].sum(axis=1)


# ---------- Option 1: best wholesale reference lap ----------
print("=== OPTION 1: best wholesale reference candidates ===")
print("(lowest max-distance across all 7 pins; this lap is closest to truth everywhere)\n")
print(f"  {'session_id':<22s} {'lap':>3s} {'lap_t':>6s} {'max':>6s} {'sum':>6s}  pin distances (m)")
header = "  " + " " * 22 + " " * 18 + "  " + " ".join(f"{p:>6s}" for p in pins.keys())
print(header)
for _, r in df.sort_values("max_all_m").head(8).iterrows():
    pin_dists = " ".join(f"{r[f'{p}_m']:>6.2f}" for p in pins.keys())
    print(f"  {r.session_id:<22s} {int(r.lap):>3d} {r.lap_time_s:>6.2f} "
          f"{r.max_all_m:>5.2f}m {r.sum_all_m:>5.1f}m  {pin_dists}")


# ---------- Option 2: best inner-loop splice donor ----------
print("\n=== OPTION 2: best inner-loop splice donor (clean at both T7 and T9) ===")
print("(lowest max-distance across T7 + T9; this lap's GPS through 1648-2237m gets used as the splice)\n")
print(f"  {'session_id':<22s} {'lap':>3s} {'lap_t':>6s} {'max_T7T9':>9s}  pin distances (m)")
print(header.replace("max", "max_T7T9").replace("sum", "        "))
for _, r in df.sort_values("max_inner_m").head(10).iterrows():
    pin_dists = " ".join(f"{r[f'{p}_m']:>6.2f}" for p in pins.keys())
    print(f"  {r.session_id:<22s} {int(r.lap):>3d} {r.lap_time_s:>6.2f}     "
          f"{r.max_inner_m:>5.2f}m  {pin_dists}")


# ---------- May-1 reference for comparison ----------
ref = df[(df.session_id == "20260501-101355") & (df.lap == 3)]
if len(ref) == 1:
    r = ref.iloc[0]
    pin_dists = " ".join(f"{r[f'{p}_m']:>6.2f}" for p in pins.keys())
    print(f"\n=== Current reference (May-1 L3) for comparison ===")
    print(f"  {r.session_id:<22s} {int(r.lap):>3d} {r.lap_time_s:>6.2f} "
          f"max_all={r.max_all_m:>5.2f}m max_T7T9={r.max_inner_m:>5.2f}m  {pin_dists}")

# Verify T3 is now correct
print("\n=== Pin sanity check (which corner each pin actually lands in on May-1 L3) ===")
if len(ref) == 1:
    r = ref.iloc[0]
    for pin_id in pins.keys():
        td = r[f"{pin_id}_td"]
        landed = "?"
        for cid, (s, e) in corner_ranges.items():
            if s <= td <= e:
                landed = cid
                break
        ok = "yes" if landed == pin_id else "NO"
        print(f"  {pin_id:<5s}  closest May-1 sample at track_dist_m={td:.0f}  → {landed:<5s}  match={ok}")
