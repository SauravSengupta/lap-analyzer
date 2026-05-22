"""GPS calibration via three known kerb-touch reference points.

User says T1, T5, T14 apexes are where they most consistently touch the kerb,
and provided May-1's GPS coordinates at those moments. For each other fast lap:

  1. Find the GPS sample within each of T1/T5/T14 closest to the May-1 reference.
  2. Compute the offset (lap_pos − ref_pos) at each kerb in meters.
  3. If all three offsets agree (same direction/magnitude), it's coherent GPS drift.
     If they disagree, it's mostly line variation + noise.
  4. Use the average of the three offsets as the lap's drift estimate.
  5. Apply that correction to T11 and re-measure spread.

Reference coords are from 20260501-101355 L3 (the May-1 lap).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))

# May-1 reference kerb-touches (user-provided) and their corner ranges from ridge.json
ALL_REFERENCES = {
    "T1":  {"lat": 47.25438312, "long": -123.18662753, "start": 380.0,  "end": 530.0,  "apex_mph": 89},
    "T2":  {"lat": 47.25511990, "long": -123.18463678, "start": 530.0,  "end": 670.0,  "apex_mph": 60},
    "T5":  {"lat": 47.25759973, "long": -123.18488476, "start": 880.0,  "end": 1027.0, "apex_mph": 77},
    "T11": {"lat": 47.25782684, "long": -123.19350065, "start": 2529.0, "end": 2681.0, "apex_mph": 41},
    "T13": {"lat": 47.25580540, "long": -123.19822240, "start": 3203.0, "end": 3258.0, "apex_mph": 27},
    "T14": {"lat": 47.25546482, "long": -123.19812462, "start": 3268.0, "end": 3330.0, "apex_mph": 37},
}
# T11 is our test corner — exclude from calibration anchors (would be circular)
HELD_OUT = "T11"
REFERENCES = {k: v for k, v in ALL_REFERENCES.items() if k != HELD_OUT}
VALIDATION_THRESHOLD_M = 8.0  # auto-reject any anchor whose median spread exceeds this
T11_START, T11_END = 2529.0, 2681.0
PRE_T11, POST_T11 = 100.0, 80.0

root = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
laps = pd.concat([pd.read_csv(p) for p in root.rglob("laps.csv")], ignore_index=True)
clean = laps[laps.is_clean.astype(bool)]
threshold = clean.lap_time_s.quantile(0.10)
fast = clean[clean.lap_time_s <= threshold][["session_id", "lap", "lap_time_s"]]
extra = pd.DataFrame([{"session_id": "20260501-101355", "lap": 3, "lap_time_s": 124.5}])
all_laps = pd.concat([fast, extra]).drop_duplicates(["session_id", "lap"]).reset_index(drop=True)
print(f"Analyzing {len(all_laps)} laps")


def closest_sample_to(samples: pd.DataFrame, lap: int, ref: dict) -> pd.Series | None:
    """Find the sample in this lap inside [start, end] dist range with min distance to ref point."""
    seg = samples[
        (samples["lap"] == lap)
        & (samples["dist_lap_m"] >= ref["start"])
        & (samples["dist_lap_m"] <= ref["end"])
    ]
    if len(seg) == 0:
        return None
    dlat_m = (seg["lat"] - ref["lat"]) * M_PER_DEG_LAT
    dlon_m = (seg["long"] - ref["long"]) * M_PER_DEG_LON
    d2 = dlat_m ** 2 + dlon_m ** 2
    return seg.loc[d2.idxmin()]


# Step 1: collect closest-point offsets for ALL candidate references (including held-out)
records = []
for _, lap in all_laps.iterrows():
    sid, ln = lap.session_id, int(lap.lap)
    s = pd.read_parquet(root / sid / "samples.parquet")
    rec = {"session_id": sid, "lap": ln, "lap_time_s": lap.lap_time_s}
    for cname, ref in ALL_REFERENCES.items():
        c = closest_sample_to(s, ln, ref)
        if c is None:
            rec[f"{cname}_off_lat_m"] = np.nan
            rec[f"{cname}_off_lon_m"] = np.nan
            rec[f"{cname}_dist_m"] = np.nan
            continue
        rec[f"{cname}_off_lat_m"] = (c["lat"] - ref["lat"]) * M_PER_DEG_LAT
        rec[f"{cname}_off_lon_m"] = (c["long"] - ref["long"]) * M_PER_DEG_LON
        rec[f"{cname}_dist_m"] = np.sqrt(rec[f"{cname}_off_lat_m"]**2 + rec[f"{cname}_off_lon_m"]**2)
    records.append(rec)

df = pd.DataFrame(records)
print(f"\n=== Reference candidate validation ({len(df)} laps) ===")
print(f"Test: median distance from May-1 reference across laps. Threshold to keep: {VALIDATION_THRESHOLD_M:.1f}m")
print(f"Note: {HELD_OUT} held out from calibration (it's the test corner)\n")
print(f"  {'corner':<6s} {'apex mph':>9s} {'median':>10s} {'90th-pct':>10s}  decision")
validated = []
for cname, ref in ALL_REFERENCES.items():
    med = df[f"{cname}_dist_m"].median()
    p90 = df[f"{cname}_dist_m"].quantile(0.9)
    if cname == HELD_OUT:
        decision = "HELD OUT (test corner)"
    elif med <= VALIDATION_THRESHOLD_M:
        decision = "ACCEPT"
        validated.append(cname)
    else:
        decision = "REJECT (too much spread)"
    print(f"  {cname:<6s} {ref['apex_mph']:>7d}    {med:>7.2f}m  {p90:>7.2f}m   {decision}")

if not validated:
    print("\nNo references passed validation. Aborting.")
    raise SystemExit
print(f"\nUsing {len(validated)} validated anchors: {validated}")

# Step 2: compute per-lap drift using validated anchors only
def lap_drift(rec, anchors):
    lat = np.mean([rec[f"{c}_off_lat_m"] for c in anchors])
    lon = np.mean([rec[f"{c}_off_lon_m"] for c in anchors])
    devs_lat = [rec[f"{c}_off_lat_m"] - lat for c in anchors]
    devs_lon = [rec[f"{c}_off_lon_m"] - lon for c in anchors]
    disagree = np.sqrt(np.mean(np.array(devs_lat)**2 + np.array(devs_lon)**2)) if len(anchors) > 1 else 0.0
    return lat, lon, np.sqrt(lat**2 + lon**2), disagree

drifts = df.apply(lambda r: lap_drift(r, validated), axis=1, result_type="expand")
df[["drift_lat_m", "drift_lon_m", "drift_mag_m", "kerb_disagreement_m"]] = drifts

print(f"\nMean-of-three drift estimate magnitude per lap:")
print(f"  median {df['drift_mag_m'].median():.2f}m  90th-pct {df['drift_mag_m'].quantile(0.9):.2f}m  max {df['drift_mag_m'].max():.2f}m")

print(f"\nKerb disagreement within a lap (std around the lap's mean drift):")
print(f"  median {df['kerb_disagreement_m'].median():.2f}m  90th-pct {df['kerb_disagreement_m'].quantile(0.9):.2f}m")
print(f"  If drift were coherent across the track, this would be ~GPS noise (~3m).")
print(f"  Larger values mean line-variation at the 'reference' kerbs is competing with drift.")

# Show a few sample laps' offsets
print(f"\n=== Example offsets (10 laps) ===")
sample = df.sample(min(10, len(df)), random_state=42).sort_values("lap_time_s")
cols = ["session_id", "lap", "lap_time_s",
        "T1_off_lat_m", "T1_off_lon_m",
        "T5_off_lat_m", "T5_off_lon_m",
        "T14_off_lat_m", "T14_off_lon_m",
        "drift_mag_m", "kerb_disagreement_m"]
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)
print(sample[cols].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))


# Step 3+4: apply per-lap drift correction to T11 traces
def get_t11(sid: str, lap: int) -> pd.DataFrame:
    s = pd.read_parquet(root / sid / "samples.parquet")
    return s[(s["lap"] == lap) & (s["dist_lap_m"] >= T11_START - PRE_T11) & (s["dist_lap_m"] <= T11_END + POST_T11)].copy()


def t11_top_lat(seg: pd.DataFrame) -> tuple[float, float]:
    """Return (lat, long) of the max-latitude point within T11's [start, end]."""
    inside = seg[(seg["dist_lap_m"] >= T11_START) & (seg["dist_lap_m"] <= T11_END)]
    row = inside.loc[inside["lat"].idxmax()]
    return float(row["lat"]), float(row["long"])


t11_segments = {}
for _, r in df.iterrows():
    seg = get_t11(r.session_id, r.lap)
    if len(seg) == 0:
        continue
    seg["lat_corrected"]  = seg["lat"]  - r.drift_lat_m / M_PER_DEG_LAT
    seg["long_corrected"] = seg["long"] - r.drift_lon_m / M_PER_DEG_LON
    t11_segments[(r.session_id, r.lap)] = (seg, r.drift_lat_m, r.drift_lon_m)

# Compute top-of-curve before vs after correction
tops = []
for (sid, ln), (seg, dlat, dlon) in t11_segments.items():
    inside = seg[(seg["dist_lap_m"] >= T11_START) & (seg["dist_lap_m"] <= T11_END)]
    raw_max = inside["lat"].max()
    cor_max = inside["lat_corrected"].max()
    tops.append({"session_id": sid, "lap": ln, "raw_max_lat": raw_max, "cor_max_lat": cor_max,
                 "drift_lat_m": dlat, "drift_lon_m": dlon})
tops_df = pd.DataFrame(tops)

raw_spread = (tops_df.raw_max_lat.max() - tops_df.raw_max_lat.min()) * M_PER_DEG_LAT
cor_spread = (tops_df.cor_max_lat.max() - tops_df.cor_max_lat.min()) * M_PER_DEG_LAT
raw_std = tops_df.raw_max_lat.std() * M_PER_DEG_LAT
cor_std = tops_df.cor_max_lat.std() * M_PER_DEG_LAT
print(f"\n=== T11 max-latitude spread BEFORE vs AFTER kerb-anchored correction ===")
print(f"  BEFORE: spread {raw_spread:.2f}m  std {raw_std:.2f}m")
print(f"  AFTER:  spread {cor_spread:.2f}m  std {cor_std:.2f}m")

# Focal laps: BEST FAST vs MAY-1 delta after correction
focus_pairs = [("BEST FAST", "20240518-111554", 2), ("MAY-1", "20260501-101355", 3)]
print(f"\n=== T11 max-lat for focal laps ===")
focus_results = []
for label, sid, ln in focus_pairs:
    row = tops_df[(tops_df.session_id == sid) & (tops_df.lap == ln)]
    if len(row) == 0:
        continue
    r = row.iloc[0]
    raw_m = r.raw_max_lat * M_PER_DEG_LAT
    cor_m = r.cor_max_lat * M_PER_DEG_LAT
    print(f"{label}: raw {r.raw_max_lat:.7f}  corrected {r.cor_max_lat:.7f}  drift_applied: ({r.drift_lat_m:+.2f}, {r.drift_lon_m:+.2f}) m")
    focus_results.append((label, r))

if len(focus_results) == 2:
    raw_d = (focus_results[0][1].raw_max_lat - focus_results[1][1].raw_max_lat) * M_PER_DEG_LAT
    cor_d = (focus_results[0][1].cor_max_lat - focus_results[1][1].cor_max_lat) * M_PER_DEG_LAT
    print(f"\nBEST FAST minus MAY-1 max-lat delta:")
    print(f"  BEFORE: {raw_d:+.2f}m")
    print(f"  AFTER:  {cor_d:+.2f}m")

# Plot before/after envelope
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for col, (title, lat_key, lon_key) in enumerate([("RAW T11 envelope", "lat", "long"),
                                                  ("KERB-CORRECTED T11 envelope", "lat_corrected", "long_corrected")]):
    ax = axes[col]
    for (sid, ln), (seg, _, _) in t11_segments.items():
        ax.plot(seg[lon_key], seg[lat_key], color="gray", alpha=0.3, linewidth=1)
    for label, sid, ln, color in [("BEST FAST", "20240518-111554", 2, "tab:blue"), ("MAY-1", "20260501-101355", 3, "tab:red")]:
        if (sid, ln) in t11_segments:
            seg = t11_segments[(sid, ln)][0]
            ax.plot(seg[lon_key], seg[lat_key], color=color, linewidth=3, label=label)
    spread_m = (tops_df[("cor_max_lat" if col == 1 else "raw_max_lat")].max() - tops_df[("cor_max_lat" if col == 1 else "raw_max_lat")].min()) * M_PER_DEG_LAT
    ax.set_title(f"{title}\nmax-lat spread across laps: {spread_m:.1f}m")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)

plt.tight_layout()
out = Path(__file__).parent / "t11_kerb_corrected.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nwrote {out}")
