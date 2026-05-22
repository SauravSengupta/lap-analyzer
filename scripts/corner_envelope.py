"""Plot a drift-corrected envelope for any corner using the corpus reliability flags.

Filters: `transit_reliable` (corpus-wide flag in corners.parquet — STANDARD tier:
drift disagreement <= 20m, transit offset <= 40m, neighborhood offset <= 40m).
See lap_analyzer.quality for the definition.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]  # scripts/ -> repo root

M_PER_DEG_LAT = 111_132.0
M_PER_DEG_LON = 111_132.0 * np.cos(np.radians(47.255))
PRE = 100.0
POST = 80.0

FOCAL = [
    ("BEST FAST 2024-05-18 L2", "20240518-111554", 2, "tab:blue"),
    ("MAY-1     2026-05-01 L3", "20260501-101355", 3, "tab:red"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corner", required=True, help="e.g. T11, T12")
    p.add_argument("--pool", choices=["top-decile", "all-reliable"], default="top-decile",
                   help="Which laps to draw as gray traces")
    p.add_argument("--focal-laps", action="store_true",
                   help="Show the BEST FAST and MAY-1 focal laps as colored overlays")
    p.add_argument("--basemap", action="store_true",
                   help="Overlay on Esri World Imagery satellite basemap")
    args = p.parse_args()

    root = REPO_ROOT / "data" / "sessions" / "ridge"
    track_json = json.loads((REPO_ROOT / "tracks" / "ridge.json").read_text(encoding="utf-8"))
    pins_path = REPO_ROOT / "data" / "notes" / "ridge_apex_pins.json"
    pins = {k: v for k, v in json.loads(pins_path.read_text(encoding="utf-8")).items() if not k.startswith("_")}

    corner = next(c for c in track_json["corners"] if c["id"] == args.corner)
    c_start, c_end, c_apex = corner["start_m"], corner["end_m"], corner["apex_m"]
    is_left = corner["type"] == "left"
    print(f"Corner {args.corner}: start_m={c_start}, end_m={c_end}, apex_m={c_apex}, type={corner['type']}")

    # Pull all transits (already has transit_reliable from flag_quality)
    transits = pd.concat([pd.read_parquet(p) for p in root.rglob("corners.parquet")], ignore_index=True)
    laps = pd.concat([pd.read_csv(p) for p in root.rglob("laps.csv")], ignore_index=True)

    # Build the lap set
    target_transits = transits[transits.corner_id == args.corner]
    target_transits = target_transits[target_transits.transit_reliable]
    target_transits = target_transits.merge(laps[["session_id", "lap", "lap_time_s", "is_clean"]],
                                            on=["session_id", "lap"])
    target_transits = target_transits[target_transits.is_clean.astype(bool)]

    if args.pool == "top-decile":
        threshold = laps[laps.is_clean.astype(bool)].lap_time_s.quantile(0.10)
        target_transits = target_transits[target_transits.lap_time_s <= threshold]
        pool_label = f"top-10% laps (threshold {threshold:.2f}s)"
    else:
        pool_label = "all reliable clean laps"

    print(f"Pool: {pool_label}")
    print(f"Reliable {args.corner} transits in pool: {len(target_transits)} laps")

    # Collect drift-corrected traces and per-lap apex markers.
    # Apex marker = sample in the corner closest to the Maps pin (geometric, works for any
    # corner orientation). Falls back to extreme-latitude if no pin.
    pin_for_apex = pins.get(args.corner)
    traces = []
    tops = []
    for _, r in target_transits.iterrows():
        sid, ln = r.session_id, int(r.lap)
        s = pd.read_parquet(root / sid / "samples.parquet")
        seg = s[(s["lap"] == ln) & (s["track_dist_m"] >= c_start - PRE) & (s["track_dist_m"] <= c_end + POST)].copy()
        if len(seg) < 5:
            continue
        dlat = float(seg["gps_drift_lat_m"].iloc[0])
        dlon = float(seg["gps_drift_lon_m"].iloc[0])
        seg["lat_corr"] = seg["lat"] - dlat / M_PER_DEG_LAT
        seg["long_corr"] = seg["long"] - dlon / M_PER_DEG_LON
        traces.append((sid, ln, seg))
        in_corner = seg[(seg["track_dist_m"] >= c_start) & (seg["track_dist_m"] <= c_end)]
        if len(in_corner) == 0:
            continue
        if pin_for_apex:
            d = np.sqrt(
                ((in_corner["lat_corr"] - pin_for_apex["lat"]) * M_PER_DEG_LAT) ** 2
                + ((in_corner["long_corr"] - pin_for_apex["long"]) * M_PER_DEG_LON) ** 2
            )
            top_row = in_corner.loc[d.idxmin()]
            top_pin_dist = float(d.min())
        else:
            top_row = (in_corner.loc[in_corner["lat_corr"].idxmax()] if is_left
                       else in_corner.loc[in_corner["lat_corr"].idxmin()])
            top_pin_dist = np.nan
        tops.append({"session_id": sid, "lap": ln,
                     "top_lat": top_row["lat_corr"], "top_long": top_row["long_corr"],
                     "pin_dist": top_pin_dist})

    tops_df = pd.DataFrame(tops)
    # 2D cluster spread of the apex markers
    cx = tops_df.top_long.mean()
    cy = tops_df.top_lat.mean()
    dx = (tops_df.top_long - cx) * M_PER_DEG_LON
    dy = (tops_df.top_lat - cy) * M_PER_DEG_LAT
    radial = np.sqrt(dx ** 2 + dy ** 2)
    std_m = float(radial.std())
    spread = float(radial.max() - radial.min()) * 2  # rough peak-to-peak diameter

    # Focal laps (optional)
    focal_segs = []
    if args.focal_laps:
        for label, sid, ln, color in FOCAL:
            sp = root / sid / "samples.parquet"
            if not sp.exists():
                continue
            s = pd.read_parquet(sp)
            seg = s[(s["lap"] == ln) & (s["track_dist_m"] >= c_start - PRE) & (s["track_dist_m"] <= c_end + POST)].copy()
            if len(seg) == 0:
                continue
            dlat = float(seg["gps_drift_lat_m"].iloc[0])
            dlon = float(seg["gps_drift_lon_m"].iloc[0])
            seg["lat_corr"] = seg["lat"] - dlat / M_PER_DEG_LAT
            seg["long_corr"] = seg["long"] - dlon / M_PER_DEG_LON
            reliable_here = ((target_transits.session_id == sid) & (target_transits.lap == ln)).any()
            label_with_flag = label + ("" if reliable_here else "  [NOT reliable at this corner]")
            focal_segs.append((label_with_flag, seg, color))

    # Plot
    fig, ax = plt.subplots(figsize=(11, 9))

    # Satellite basemap (optional). Requires us to work in Web Mercator (EPSG:3857).
    if args.basemap:
        import contextily as cx
        from pyproj import Transformer
        to_wm = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        def to_xy(lon, lat):
            return to_wm.transform(np.asarray(lon), np.asarray(lat))

        for sid, ln, seg in traces:
            x, y = to_xy(seg["long_corr"], seg["lat_corr"])
            ax.plot(x, y, color="white", alpha=0.55, linewidth=1.2, zorder=3)
        for label, seg, color in focal_segs:
            x, y = to_xy(seg["long_corr"], seg["lat_corr"])
            ax.plot(x, y, color=color, linewidth=3, label=label, zorder=4)
        tx, ty = to_xy(tops_df.top_long.values, tops_df.top_lat.values)
        ax.scatter(tx, ty, s=25, c="cyan", edgecolors="black", linewidths=0.5,
                   zorder=5, label="per-lap apex marker (closest sample to pin)")
        pin = pins.get(args.corner)
        if pin:
            px, py = to_xy([pin["long"]], [pin["lat"]])
            ax.scatter(px, py, marker="*", s=400, c="gold",
                       edgecolors="black", linewidths=1.4, zorder=6,
                       label="Maps pin (visual apex)")
        # Set extent before adding basemap to control zoom
        all_x = np.concatenate([to_xy(seg["long_corr"], seg["lat_corr"])[0] for _, _, seg in traces])
        all_y = np.concatenate([to_xy(seg["long_corr"], seg["lat_corr"])[1] for _, _, seg in traces])
        pad = 30  # meters of padding around the trace bbox
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
        cx.add_basemap(ax, source=cx.providers.Esri.WorldImagery, crs="EPSG:3857")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        for sid, ln, seg in traces:
            ax.plot(seg["long_corr"], seg["lat_corr"], color="gray", alpha=0.30, linewidth=1)
        for label, seg, color in focal_segs:
            ax.plot(seg["long_corr"], seg["lat_corr"], color=color, linewidth=3, label=label)
        ax.scatter(tops_df.top_long, tops_df.top_lat, s=18, c="black", alpha=0.45, zorder=3,
                   label="per-lap apex marker (closest sample to pin)")

        pin = pins.get(args.corner)
        if pin:
            ax.scatter([pin["long"]], [pin["lat"]], marker="*", s=300, c="gold",
                       edgecolors="black", linewidths=1.2, zorder=5,
                       label="Maps pin (visual apex)")

    title_extra = ""
    if pin_for_apex is not None:
        title_extra = f"  |  median lap-to-pin: {tops_df.pin_dist.median():.2f}m"
    ax.set_title(
        f"{args.corner} — {len(traces)} reliable {pool_label.replace('top-10% ', '')}\n"
        f"apex-cluster radial std: {std_m:.1f}m{title_extra}  |  filter: STANDARD tier (transit_reliable)",
        fontsize=10
    )
    if not args.basemap:
        ax.set_xlabel("longitude (drift-corrected)")
        ax.set_ylabel("latitude (drift-corrected)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    # Label the four most-extreme laps from the cluster center (annotation positions
    # must be in the same coord system as the axes — Web Mercator if basemap, else lat/long).
    sorted_tops = tops_df.assign(rad=radial).sort_values("rad", ascending=False).head(4)
    if args.basemap:
        from pyproj import Transformer
        _to_wm = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        for _, r in sorted_tops.iterrows():
            ax_x, ax_y = _to_wm.transform(r.top_long, r.top_lat)
            ax.annotate(f"{r.session_id[:8]} L{int(r.lap)}", xy=(ax_x, ax_y),
                        xytext=(5, 5), textcoords="offset points", fontsize=7,
                        color="white",
                        path_effects=[__import__("matplotlib.patheffects", fromlist=["withStroke"]).withStroke(linewidth=2, foreground="black")])
    else:
        for _, r in sorted_tops.iterrows():
            ax.annotate(f"{r.session_id[:8]} L{int(r.lap)}", xy=(r.top_long, r.top_lat),
                        xytext=(5, 5), textcoords="offset points", fontsize=7, color="darkblue")

    plt.tight_layout()
    out = Path(__file__).parent / f"envelope_{args.corner}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
