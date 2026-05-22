"""Render PIR track map with corner zones overlaid.

Plots:
  - Centerline (gray)
  - Each corner [start_m, end_m] colored & labeled at its apex
  - Calibration-anchor pin positions (user's visual-apex Maps pins)
  - Start/finish line
  - Direction arrow on the front straight

Output: app/notebooks/pir_track_map.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CENTERLINE = REPO_ROOT / "data" / "corpus" / "pir_centerline.parquet"
TRACK_JSON = REPO_ROOT / "app" / "tracks" / "pir.json"
PINS_JSON = REPO_ROOT / "data" / "notes" / "pir_apex_pins.json"
OUT_PATH = Path(__file__).parent / "pir_track_map.png"


def latlon_to_local(lat, lon, lat0, lon0):
    """Convert lat/lon to local meters (east, north) relative to (lat0, lon0)."""
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = m_per_deg_lat * np.cos(np.radians(lat0))
    east = (lon - lon0) * m_per_deg_lon
    north = (lat - lat0) * m_per_deg_lat
    return east, north


def main():
    cl = pd.read_parquet(CENTERLINE)
    track = json.loads(TRACK_JSON.read_text(encoding="utf-8"))
    pins = json.loads(PINS_JSON.read_text(encoding="utf-8"))
    pins = {k: v for k, v in pins.items() if not k.startswith("_")}

    # Center on start/finish for readability
    lat0 = float(track["start_finish"]["lat"])
    lon0 = float(track["start_finish"]["long"])

    cl_e, cl_n = latlon_to_local(cl["lat"].values, cl["long"].values, lat0, lon0)
    td = cl["track_dist_m"].values

    fig, ax = plt.subplots(figsize=(14, 12))

    # Base centerline
    ax.plot(cl_e, cl_n, color="0.7", linewidth=2.0, zorder=1, label="centerline")

    # Color palette for corners — alternating to make adjacent corners distinct
    colors = plt.cm.tab20(np.linspace(0, 1, 12))

    for i, c in enumerate(track["corners"]):
        cid = c["id"]
        s, e, a = float(c["start_m"]), float(c["end_m"]), float(c["apex_m"])
        # Centerline points within [s, e]
        mask = (td >= s) & (td <= e)
        if not mask.any():
            continue
        col = colors[i]
        ax.plot(cl_e[mask], cl_n[mask], color=col, linewidth=5.0, zorder=3,
                solid_capstyle="round")

        # Apex marker — find centerline point closest in track_dist_m
        apex_idx = int(np.argmin(np.abs(td - a)))
        ax.scatter([cl_e[apex_idx]], [cl_n[apex_idx]], color=col, s=140,
                   edgecolor="black", linewidth=1.2, zorder=5, marker="o")

        # Secondary apex (for T4)
        if c.get("secondary_apex_m"):
            sec = float(c["secondary_apex_m"])
            sec_idx = int(np.argmin(np.abs(td - sec)))
            ax.scatter([cl_e[sec_idx]], [cl_n[sec_idx]], color=col, s=60,
                       edgecolor="black", linewidth=0.8, zorder=5, marker="s")
            ax.annotate(f"{cid}a", (cl_e[sec_idx], cl_n[sec_idx]),
                        textcoords="offset points", xytext=(8, -8),
                        fontsize=8, color=col, fontweight="bold")

        # Label at apex (offset outward from track direction)
        # Use a perpendicular offset based on track direction at the apex
        i_prev = max(0, apex_idx - 5)
        i_next = min(len(td) - 1, apex_idx + 5)
        dx = cl_e[i_next] - cl_e[i_prev]
        dy = cl_n[i_next] - cl_n[i_prev]
        norm = np.hypot(dx, dy) or 1.0
        # Perpendicular (rotate 90deg) — sign chosen by corner type to push outward.
        # For a right turn (clockwise), outside is to the left of direction (rotate +90).
        perp_sign = 1 if c["type"] == "right" else -1
        offset_x = -perp_sign * (dy / norm) * 35
        offset_y = perp_sign * (dx / norm) * 35
        ax.annotate(cid, (cl_e[apex_idx], cl_n[apex_idx]),
                    xytext=(cl_e[apex_idx] + offset_x, cl_n[apex_idx] + offset_y),
                    fontsize=12, fontweight="bold", color="black",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=col, linewidth=1.5))

    # User pin coords (visual / kerb-touch apex)
    for cid, p in pins.items():
        e, n = latlon_to_local(p["lat"], p["long"], lat0, lon0)
        marker = "+" if "a" in cid or "b" in cid else "x"
        ax.scatter([e], [n], color="black", s=80, marker=marker, linewidth=1.5,
                   zorder=4, label=("user pin" if cid == "T1" else None))

    # Start/finish line
    ax.scatter([0], [0], color="red", s=200, marker="*", zorder=6,
               edgecolor="black", linewidth=1, label="start/finish")

    # Direction arrow — show where lap-distance increases from 0
    arrow_start_idx = 30
    arrow_end_idx = 80
    ax.annotate("", xy=(cl_e[arrow_end_idx], cl_n[arrow_end_idx]),
                xytext=(cl_e[arrow_start_idx], cl_n[arrow_start_idx]),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))

    ax.set_aspect("equal")
    ax.set_xlabel("east (m) — relative to start/finish")
    ax.set_ylabel("north (m) — relative to start/finish")
    ax.set_title("Portland International Raceway — corner zones\n"
                 "(thick colored arcs = [start_m, end_m] per corner; "
                 "circle = apex_m; square = secondary apex; "
                 "x/+ = your Maps pins; red star = start/finish)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
