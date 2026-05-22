"""Cross-check T11 apex-offset finding using lat-G peak distance instead of speed-min."""
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", None)

T11_START = 2529.0
T11_END = 2681.0
T11_REF_APEX = 2600.0  # tracks/ridge.json T11.apex_m

root = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
laps = pd.concat([pd.read_csv(p) for p in root.rglob("laps.csv")], ignore_index=True)
clean = laps[laps.is_clean.astype(bool)]
near_may1_laps = clean[(clean.lap_time_s >= 123) & (clean.lap_time_s <= 126)][["session_id", "lap", "lap_time_s"]]

print(f"Pulling T11 lat-G + speed traces for {len(near_may1_laps)} laps in 123-126s range...")

rows = []
for _, lap_row in near_may1_laps.iterrows():
    sid, ln = lap_row["session_id"], int(lap_row["lap"])
    samples = pd.read_parquet(root / sid / "samples.parquet")
    in_corner = samples[
        (samples["lap"] == ln)
        & (samples["dist_lap_m"] >= T11_START)
        & (samples["dist_lap_m"] <= T11_END)
    ]
    if len(in_corner) < 5:
        continue

    # Speed-min based apex
    smin_idx = in_corner["speed_mph"].idxmin()
    smin_dist = float(in_corner.loc[smin_idx, "dist_lap_m"])
    smin_speed = float(in_corner.loc[smin_idx, "speed_mph"])

    # Lat-G peak based apex (T11 is a left, so most-negative lat_g; using abs() to be safe)
    latg_abs = in_corner["lat_g"].abs()
    lpk_idx = latg_abs.idxmax()
    lpk_dist = float(in_corner.loc[lpk_idx, "dist_lap_m"])
    lpk_value = float(in_corner.loc[lpk_idx, "lat_g"])

    rows.append({
        "session_id": sid,
        "lap": ln,
        "lap_time_s": lap_row["lap_time_s"],
        "speed_min_dist": round(smin_dist, 1),
        "speed_min_offset": round(smin_dist - T11_REF_APEX, 1),
        "speed_min_mph": round(smin_speed, 1),
        "latg_peak_dist": round(lpk_dist, 1),
        "latg_peak_offset": round(lpk_dist - T11_REF_APEX, 1),
        "latg_peak_g": round(lpk_value, 2),
        "smin_to_latgpk_m": round(lpk_dist - smin_dist, 1),
    })

df = pd.DataFrame(rows).sort_values("lap_time_s")
print(df.to_string(index=False))

# Correlation between the two offset metrics
corr = df[["speed_min_offset", "latg_peak_offset"]].corr().iloc[0, 1]
print(f"\nPearson correlation between speed-min-offset and lat-G-peak-offset: {corr:.3f}")

# Show how May-1 ranks under both metrics
may1 = df[(df.session_id == "20260501-101355") & (df.lap == 3)]
print(f"\nMay-1 L3:")
print(f"  by speed-min apex offset: {may1.iloc[0].speed_min_offset:+.1f}m  (vs median {df.speed_min_offset.median():+.1f})")
print(f"  by lat-G peak offset:     {may1.iloc[0].latg_peak_offset:+.1f}m  (vs median {df.latg_peak_offset.median():+.1f})")
