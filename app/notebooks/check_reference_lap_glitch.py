"""Check whether the May-1 reference lap (20260501-101355 L3) is itself glitched
at T7, T8, T9.

If the reference lap's GPS recording at the apex is far from the user-provided
Google Maps pin (the true physical apex location), then the reference lap is
contaminated by the inner-loop glitch — and every track_dist_m mapping in that
region inherits the contamination.

Reference lap GPS samples come from samples.parquet. True apex coords come from
the user via Google Maps pins.

For each corner:
  - Find the reference lap's sample closest to the Maps pin.
  - Report the distance.
  - Also compare to the RaceRender screenshot coord (the recorded GPS at the
    moment the user identified as "apex" in the video).
"""
from pathlib import Path
import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))

# User-provided Google Maps pins for corner apexes
MAPS_PINS = {
    "T7":  (47.25718, -123.18758),
    "T12": (47.25611, -123.19292),
    # T8b, T9 not yet pinned by user
}

# RaceRender screenshot coords — these are the recorded GPS at the visual-apex
# moment on May-1 L3 (i.e. the reference lap's recorded position at apex)
RACERENDER_COORDS = {
    "T7":  (47.25677391, -123.18883709),
    "T8b": (47.25624614, -123.18643640),
    "T9":  (47.25555535, -123.18834718),
}

# Corner ranges (in track_dist_m, the GPS-aligned canonical coord)
CORNER_RANGES = {
    "T7":  (1600.0, 1750.0),
    "T8b": (1880.0, 2010.0),
    "T9":  (2050.0, 2200.0),
    "T12": (2718.0, 2907.0),
}

REF_PATH = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge\20260501-101355\samples.parquet")
s = pd.read_parquet(REF_PATH)
ref = s[s["lap"] == 3].copy()
print(f"Reference lap: {len(ref)} samples on lap 3, lap_time should be ~124.4s")


def m_dist(lat1, lon1, lat2, lon2) -> float:
    return float(np.sqrt(((lat2 - lat1) * M_PER_DEG_LAT) ** 2
                       + ((lon2 - lon1) * M_PER_DEG_LON) ** 2))


def closest(samples: pd.DataFrame, ref_lat: float, ref_lon: float) -> tuple[pd.Series, float]:
    dlat_m = (samples["lat"] - ref_lat) * M_PER_DEG_LAT
    dlon_m = (samples["long"] - ref_lon) * M_PER_DEG_LON
    d = np.sqrt(dlat_m ** 2 + dlon_m ** 2)
    idx = d.idxmin()
    return samples.loc[idx], float(d.loc[idx])


print("\n=== Reference lap GPS recording vs ground-truth (Google Maps) ===")
print("If the reference lap is clean, the recorded GPS at the apex moment should")
print("be within ~3m (GPS noise) of the Maps pin.")
print()
print(f"  {'corner':<5s} {'maps_pin':<25s} {'ref recorded':<25s} {'distance':<10s}")

for cid, pin in MAPS_PINS.items():
    pin_lat, pin_lon = pin
    rstart, rend = CORNER_RANGES[cid]
    seg = ref[(ref["track_dist_m"] >= rstart) & (ref["track_dist_m"] <= rend)]
    if len(seg) == 0:
        print(f"  {cid}: no samples in track_dist_m range [{rstart}, {rend}]")
        continue
    sample, dist = closest(seg, pin_lat, pin_lon)
    print(f"  {cid:<5s} pin ({pin_lat:.5f}, {pin_lon:.5f})  "
          f"ref recording closest ({sample.lat:.5f}, {sample.long:.5f})  closest dist: {dist:.2f}m")
    if cid in RACERENDER_COORDS:
        rr_lat, rr_lon = RACERENDER_COORDS[cid]
        rr_dist = m_dist(pin_lat, pin_lon, rr_lat, rr_lon)
        print(f"        RaceRender apex coord:              ({rr_lat:.5f}, {rr_lon:.5f})  vs maps pin: {rr_dist:.2f}m")
    print(f"        track_dist_m of closest sample:     {sample.track_dist_m:.1f}m")
    print(f"        OBD vs GPS speed at that sample:    {sample.speed_mph:.1f} / {sample.speed_mph_gps:.1f} mph "
          f"(delta {sample.speed_mph - sample.speed_mph_gps:+.1f})")
    print()


# Also dump the entire T7 segment's GPS trace so we can see if it's plausible
print("\n=== Reference lap GPS trace through T7 region (track_dist_m 1600-1750) ===")
seg = ref[(ref["track_dist_m"] >= 1600) & (ref["track_dist_m"] <= 1750)]
print(f"  {len(seg)} samples in T7 region")
print(f"  Lat range: {seg['lat'].min():.6f} to {seg['lat'].max():.6f}  span: {(seg['lat'].max()-seg['lat'].min())*M_PER_DEG_LAT:.1f}m")
print(f"  Lon range: {seg['long'].min():.6f} to {seg['long'].max():.6f}  span: {(seg['long'].max()-seg['long'].min())*M_PER_DEG_LON:.1f}m")
print(f"  OBD speed range: {seg['speed_mph'].min():.1f} to {seg['speed_mph'].max():.1f} mph")
print(f"  GPS speed range: {seg['speed_mph_gps'].min():.1f} to {seg['speed_mph_gps'].max():.1f} mph")
print(f"  GPS-OBD speed delta range: {(seg['speed_mph']-seg['speed_mph_gps']).min():+.1f} to {(seg['speed_mph']-seg['speed_mph_gps']).max():+.1f} mph")
print(f"  (Big positive delta = GPS shows car going slower than OBD = possible glitch)")
