"""Fetch the OpenStreetMap Road Course polyline for Ridge Motorsports Park
and compare it against the May-1 reference lap's GPS trace.

OSM Way 182180341 = "Road Course", highway=raceway, 162 points, closed loop.
"""
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))

OSM_URL = "https://overpass-api.de/api/interpreter"
QUERY = """[out:json][timeout:25];
way(182180341);
out geom;
"""

OUT_JSON = Path(r"D:\Projects\lap-analyzer\data\corpus\ridge_osm_road_course.json")
OUT_PARQUET = Path(r"D:\Projects\lap-analyzer\data\corpus\ridge_osm_road_course.parquet")
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

print("Fetching OSM way 182180341...")
req = Request(OSM_URL, data=f"data={QUERY}".encode("utf-8"),
              headers={"User-Agent": "lap-analyzer/0.1"})
with urlopen(req, timeout=30) as resp:
    raw = json.loads(resp.read().decode("utf-8"))

OUT_JSON.write_text(json.dumps(raw, indent=2))
print(f"  wrote raw response to {OUT_JSON}")

# Extract polyline
ways = [el for el in raw["elements"] if el["type"] == "way"]
if len(ways) != 1:
    raise SystemExit(f"expected 1 way, got {len(ways)}")
geom = ways[0]["geometry"]
poly = pd.DataFrame([{"i": i, "lat": p["lat"], "long": p["lon"]} for i, p in enumerate(geom)])
print(f"  {len(poly)} polyline points (closed loop: first=({poly.iloc[0].lat:.7f}, {poly.iloc[0].long:.7f}), last=({poly.iloc[-1].lat:.7f}, {poly.iloc[-1].long:.7f}))")


# Compute cumulative distance along polyline (osm_dist_m)
def cumulative_distance(lat, lon):
    lat = np.asarray(lat); lon = np.asarray(lon)
    dlat = np.diff(lat) * M_PER_DEG_LAT
    dlon = np.diff(lon) * M_PER_DEG_LON
    seg = np.sqrt(dlat ** 2 + dlon ** 2)
    return np.concatenate(([0.0], np.cumsum(seg)))


poly["osm_dist_m"] = cumulative_distance(poly["lat"], poly["long"])
total = poly.osm_dist_m.iloc[-1]
print(f"  total polyline length: {total:.1f}m  (lap_length_internal_m=3962, official=3975)")
print(f"  point spacing: median {np.diff(poly.osm_dist_m).mean():.1f}m  range "
      f"{np.diff(poly.osm_dist_m).min():.1f}-{np.diff(poly.osm_dist_m).max():.1f}m")

poly.to_parquet(OUT_PARQUET, index=False)
print(f"  wrote polyline to {OUT_PARQUET}")


# Compare to May-1 reference lap
REF = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge\20260501-101355\samples.parquet")
ref = pd.read_parquet(REF)
ref = ref[ref["lap"] == 3].reset_index(drop=True)


# For each ref sample, find min distance to polyline (point-to-segment)
# Vectorized over polyline edges for each sample.
def min_dist_to_polyline(sample_lat, sample_lon, poly_lat, poly_lon):
    """For each sample, distance to the nearest segment of the polyline. Returns (min_dist_m, nearest_seg_idx)."""
    # Convert poly to local meters relative to a fixed origin
    lat0, lon0 = sample_lat[0], sample_lon[0]
    p_lat_m = (poly_lat - lat0) * M_PER_DEG_LAT
    p_lon_m = (poly_lon - lon0) * M_PER_DEG_LON
    s_lat_m = (sample_lat - lat0) * M_PER_DEG_LAT
    s_lon_m = (sample_lon - lon0) * M_PER_DEG_LON

    # Edge vectors
    ex = p_lon_m[1:] - p_lon_m[:-1]   # (N_edges,)
    ey = p_lat_m[1:] - p_lat_m[:-1]
    edge_len2 = ex * ex + ey * ey

    out_d = np.empty(len(sample_lat))
    out_seg = np.empty(len(sample_lat), dtype=int)
    for i in range(len(sample_lat)):
        # Sample relative to each edge start
        dx = s_lon_m[i] - p_lon_m[:-1]
        dy = s_lat_m[i] - p_lat_m[:-1]
        # Projection parameter t in [0,1]
        t = np.clip((dx * ex + dy * ey) / np.maximum(edge_len2, 1e-9), 0, 1)
        # Foot of perpendicular
        fx = p_lon_m[:-1] + t * ex
        fy = p_lat_m[:-1] + t * ey
        d = np.sqrt((s_lon_m[i] - fx) ** 2 + (s_lat_m[i] - fy) ** 2)
        idx = d.argmin()
        out_d[i] = d[idx]
        out_seg[i] = idx
    return out_d, out_seg


print("\nComputing per-sample distance from May-1 L3 to OSM polyline...")
d, seg = min_dist_to_polyline(
    ref["lat"].values, ref["long"].values,
    poly["lat"].values, poly["long"].values,
)
ref["osm_dist_off_m"] = d

# Summary by track region (using existing track_dist_m on ref samples)
print("\n=== May-1 L3 distance from OSM polyline, by lap region ===")
print(f"  {'region':<14s} {'samples':>8s}  {'median':>9s} {'p75':>9s} {'p90':>9s} {'p95':>9s} {'max':>9s}")
regions = [
    ("Front straight", 0, 380),
    ("T1 (380-530)",  380, 530),
    ("T2 (530-670)",  530, 670),
    ("T3-T5",         670, 1100),
    ("Carousel T6",  1100, 1648),
    ("T7 (1648-1745)", 1648, 1745),
    ("T8 (1745-2020)", 1745, 2020),
    ("T9 (2035-2237)", 2035, 2237),
    ("T10 (2279-2443)", 2279, 2443),
    ("T11 (2529-2681)", 2529, 2681),
    ("T12 (2718-2907)", 2718, 2907),
    ("Back straight", 2907, 3203),
    ("T13-T15 chicane", 3203, 3490),
    ("T16 + finish", 3490, 3962),
]
for name, lo, hi in regions:
    sub = ref[(ref["track_dist_m"] >= lo) & (ref["track_dist_m"] < hi)]
    if len(sub) == 0:
        continue
    d = sub["osm_dist_off_m"]
    print(f"  {name:<14s} {len(sub):>8d}  {d.median():>7.2f}m {d.quantile(0.75):>7.2f}m "
          f"{d.quantile(0.9):>7.2f}m {d.quantile(0.95):>7.2f}m {d.max():>7.2f}m")

print(f"\nOverall: median {ref['osm_dist_off_m'].median():.2f}m  "
      f"p90 {ref['osm_dist_off_m'].quantile(0.9):.2f}m  "
      f"p99 {ref['osm_dist_off_m'].quantile(0.99):.2f}m  "
      f"max {ref['osm_dist_off_m'].max():.2f}m")

print("\nIf the May-1 lap is on-track everywhere, expect median <5m everywhere.")
print("Inflated values in T7-T9 (vs other regions) confirm the localized inner-loop glitch.")
