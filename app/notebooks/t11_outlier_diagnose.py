"""Identify the visual outliers in the corrected T11 envelope and diagnose
whether their drift estimate is unreliable (high anchor disagreement)."""
import json
import pandas as pd
import numpy as np
from pathlib import Path

T11_START = 2529.0
T11_END = 2681.0
M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))

# Anchors from ridge.json
ANCHORS = [
    ("T2",  47.25511990, -123.18463678,  530.0,  670.0),
    ("T13", 47.25580540, -123.19822240, 3203.0, 3258.0),
    ("T14", 47.25546482, -123.19812462, 3268.0, 3330.0),
]

root = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
notes = json.loads(Path(r"D:\Projects\lap-analyzer\data\notes\ridge.json").read_text())
gps_bad = {sid for sid, info in notes.items() if info.get("flag") == "gps_unreliable"}

corners_df = pd.concat([pd.read_parquet(p) for p in root.rglob("corners.parquet")], ignore_index=True)
laps = pd.concat([pd.read_csv(p) for p in root.rglob("laps.csv")], ignore_index=True)
clean = laps[laps.is_clean.astype(bool)]
threshold = clean.lap_time_s.quantile(0.10)
fast = clean[clean.lap_time_s <= threshold][["session_id", "lap", "lap_time_s"]]
fast = fast[~fast.session_id.isin(gps_bad)]

# Compute corrected top-of-curve for each fast lap PLUS per-anchor offsets to detect disagreement
records = []
for _, r in fast.iterrows():
    sid, ln = r.session_id, int(r.lap)
    s = pd.read_parquet(root / sid / "samples.parquet")
    lap_s = s[s["lap"] == ln]
    if len(lap_s) < 50:
        continue

    # Per-anchor offset (re-compute so we have individual anchor disagreement)
    per_anchor = []
    for cname, ref_lat, ref_lon, sm, em in ANCHORS:
        seg = lap_s[(lap_s["dist_lap_m"] >= sm) & (lap_s["dist_lap_m"] <= em)]
        if len(seg) == 0:
            continue
        dlat_m = (seg["lat"] - ref_lat) * M_PER_DEG_LAT
        dlon_m = (seg["long"] - ref_lon) * M_PER_DEG_LON
        d2 = dlat_m.values ** 2 + dlon_m.values ** 2
        i = int(np.argmin(d2))
        per_anchor.append((cname, float(dlat_m.iloc[i]), float(dlon_m.iloc[i])))

    if not per_anchor:
        continue
    mean_lat = np.mean([a[1] for a in per_anchor])
    mean_lon = np.mean([a[2] for a in per_anchor])
    devs = [(a[1] - mean_lat, a[2] - mean_lon) for a in per_anchor]
    disagreement = float(np.sqrt(np.mean([dl**2 + dn**2 for dl, dn in devs])))

    # Corrected top-of-curve
    in_corner = lap_s[(lap_s["dist_lap_m"] >= T11_START) & (lap_s["dist_lap_m"] <= T11_END)]
    if len(in_corner) < 5:
        continue
    corrected_lat = in_corner["lat"] - mean_lat / M_PER_DEG_LAT
    top_lat = float(corrected_lat.max())

    rec = {"session_id": sid, "lap": ln, "lap_time_s": r.lap_time_s,
           "drift_lat_m": mean_lat, "drift_lon_m": mean_lon,
           "drift_mag_m": np.sqrt(mean_lat**2 + mean_lon**2),
           "kerb_disagreement_m": disagreement,
           "corrected_top_lat": top_lat}
    for cname, dl, dn in per_anchor:
        rec[f"{cname}_dlat"] = dl
        rec[f"{cname}_dlon"] = dn
    records.append(rec)

df = pd.DataFrame(records)
median_top = df.corrected_top_lat.median()
df["dist_from_median_m"] = (df.corrected_top_lat - median_top) * M_PER_DEG_LAT

print(f"=== Top-of-curve corrected, distance from median (n={len(df)}) ===")
print(f"Median lat: {median_top:.7f}")
print(f"Std: {df.dist_from_median_m.std():.2f}m   p10: {df.dist_from_median_m.quantile(0.1):.1f}m   p90: {df.dist_from_median_m.quantile(0.9):.1f}m")
print()

print("Most extreme laps (by absolute distance from median top-lat):")
df["abs_dev"] = df.dist_from_median_m.abs()
print(df.sort_values("abs_dev", ascending=False).head(8)[
    ["session_id","lap","lap_time_s","dist_from_median_m","drift_mag_m","kerb_disagreement_m",
     "T2_dlat","T2_dlon","T13_dlat","T13_dlon","T14_dlat","T14_dlon"]
].to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

print()
print("Are the outliers laps with HIGH kerb_disagreement (= unreliable drift estimate)?")
top_devs = df.nlargest(5, "abs_dev")
rest = df.nsmallest(len(df) - 5, "abs_dev")
print(f"  Top-5 outliers: median kerb_disagreement = {top_devs.kerb_disagreement_m.median():.2f}m")
print(f"  Rest:           median kerb_disagreement = {rest.kerb_disagreement_m.median():.2f}m")
print(f"  Top-5 outliers: median drift_mag         = {top_devs.drift_mag_m.median():.2f}m")
print(f"  Rest:           median drift_mag         = {rest.drift_mag_m.median():.2f}m")
