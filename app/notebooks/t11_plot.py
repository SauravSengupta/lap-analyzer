"""Plot T11 GPS lines for two laps, colored by speed / lat-G / brake."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from pathlib import Path

T11_START = 2529.0
T11_END = 2681.0
T11_REF_APEX = 2600.0
PRE = 200.0
POST = 80.0

root = Path(r"D:\Projects\lap-analyzer\data\sessions\ridge")
NOTEBOOKS = Path(__file__).parent

LAPS = [
    ("BEST FAST  20240518-111554 L2  (lap 2:03.0  trail-brake)", "20240518-111554", 2),
    ("MAY-1      20260501-101355 L3  (lap 2:04.5  brake-then-turn)", "20260501-101355", 3),
]


def load_t11_segment(session_id: str, lap: int) -> pd.DataFrame:
    samples = pd.read_parquet(root / session_id / "samples.parquet")
    seg = samples[
        (samples["lap"] == lap)
        & (samples["dist_lap_m"] >= T11_START - PRE)
        & (samples["dist_lap_m"] <= T11_END + POST)
    ].copy().reset_index(drop=True)
    return seg


def plot_colored(ax, seg: pd.DataFrame, c, cmap, vmin, vmax, lw=4):
    x = seg["long"].values
    y = seg["lat"].values
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=Normalize(vmin, vmax), linewidth=lw)
    lc.set_array(c[:-1])
    ax.add_collection(lc)
    return lc


def overlay_markers(ax, seg: pd.DataFrame):
    in_corner = seg[(seg["dist_lap_m"] >= T11_START) & (seg["dist_lap_m"] <= T11_END)]
    smin = in_corner.loc[in_corner["speed_mph"].idxmin()]
    lpk = in_corner.loc[in_corner["lat_g"].abs().idxmax()]
    cs = seg.iloc[(seg["dist_lap_m"] - T11_START).abs().idxmin()]
    ce = seg.iloc[(seg["dist_lap_m"] - T11_END).abs().idxmin()]
    ca = seg.iloc[(seg["dist_lap_m"] - T11_REF_APEX).abs().idxmin()]
    ax.scatter([cs["long"]], [cs["lat"]], s=110, marker="X", c="white", edgecolors="black", linewidths=1.5, zorder=5)
    ax.scatter([ce["long"]], [ce["lat"]], s=110, marker="X", c="lightgray", edgecolors="black", linewidths=1.5, zorder=5)
    ax.scatter([ca["long"]], [ca["lat"]], s=180, marker="*", c="white", edgecolors="black", linewidths=1.5, zorder=5)
    ax.scatter([smin["long"]], [smin["lat"]], s=140, marker="o", facecolors="none", edgecolors="cyan", linewidths=2.5, zorder=6)
    ax.scatter([lpk["long"]], [lpk["lat"]], s=140, marker="s", facecolors="none", edgecolors="magenta", linewidths=2.5, zorder=6)


segments = [(label, load_t11_segment(sid, lap)) for label, sid, lap in LAPS]
all_speed = pd.concat([s["speed_mph"] for _, s in segments])
all_latg_abs = pd.concat([s["lat_g"].abs() for _, s in segments])
spd_min, spd_max = all_speed.min(), all_speed.max()
latg_max = all_latg_abs.max()

# Compute shared bounding box for all axes
all_long = pd.concat([s["long"] for _, s in segments])
all_lat = pd.concat([s["lat"] for _, s in segments])
xlim = (all_long.min() - 0.0001, all_long.max() + 0.0001)
ylim = (all_lat.min() - 0.0001, all_lat.max() + 0.0001)

fig, axes = plt.subplots(2, 4, figsize=(22, 12))
fig.suptitle("T11 line comparison — two laps × four signals", fontsize=15, y=0.995)

for row, (label, seg) in enumerate(segments):
    # Speed
    lc = plot_colored(axes[row, 0], seg, seg["speed_mph"].values, "RdYlGn", spd_min, spd_max)
    overlay_markers(axes[row, 0], seg)
    axes[row, 0].set_title(f"{label}\nspeed_mph", fontsize=10)
    plt.colorbar(lc, ax=axes[row, 0], label="mph", shrink=0.8)

    # Lat-G
    lc = plot_colored(axes[row, 1], seg, seg["lat_g"].abs().values, "viridis", 0, latg_max)
    overlay_markers(axes[row, 1], seg)
    axes[row, 1].set_title(f"{label}\n|lat_g|", fontsize=10)
    plt.colorbar(lc, ax=axes[row, 1], label="g", shrink=0.8)

    # Brake
    lc = plot_colored(axes[row, 2], seg, seg["brake"].astype(float).values, "RdYlBu_r", 0, 1)
    overlay_markers(axes[row, 2], seg)
    axes[row, 2].set_title(f"{label}\nbrake (red=on)", fontsize=10)
    plt.colorbar(lc, ax=axes[row, 2], label="brake", shrink=0.8)

    # Throttle
    lc = plot_colored(axes[row, 3], seg, (seg["throttle_norm"] * 100).values, "RdYlGn", 0, 100)
    overlay_markers(axes[row, 3], seg)
    axes[row, 3].set_title(f"{label}\nthrottle %", fontsize=10)
    plt.colorbar(lc, ax=axes[row, 3], label="%", shrink=0.8)

for ax in axes.flat:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude", fontsize=8)
    ax.set_ylabel("latitude", fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)

# Legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="X", markerfacecolor="white", markeredgecolor="black", color="w", markersize=10, label="T11 entry"),
    Line2D([0], [0], marker="X", markerfacecolor="lightgray", markeredgecolor="black", color="w", markersize=10, label="T11 exit"),
    Line2D([0], [0], marker="*", markerfacecolor="white", markeredgecolor="black", color="w", markersize=14, label="canonical apex (2600m)"),
    Line2D([0], [0], marker="o", markerfacecolor="none", markeredgecolor="cyan", color="w", markersize=12, markeredgewidth=2.5, label="speed-min (this lap)"),
    Line2D([0], [0], marker="s", markerfacecolor="none", markeredgecolor="magenta", color="w", markersize=12, markeredgewidth=2.5, label="lat-G peak (this lap)"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.005))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
out = NOTEBOOKS / "t11_compare.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"wrote {out}")
