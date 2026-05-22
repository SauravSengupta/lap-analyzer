"""Track-map view of corner zones — current boxes vs proposed — for review
before the corpus rebuild. Reads proposed_corner_bounds.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lap_analyzer.config import corpus_dir, tracks_dir

TRACK = "ridge"
M_LAT = 111_132.0
M_LON = 111_132.0 * np.cos(np.radians(47.255))

td = json.loads((tracks_dir() / f"{TRACK}.json").read_text(encoding="utf-8"))
corners = sorted(td["corners"], key=lambda c: c["apex_m"])
ids = [c["id"] for c in corners]
apex = {c["id"]: float(c["apex_m"]) for c in corners}
current = {c["id"]: (float(c["start_m"]), float(c["end_m"])) for c in td["corners"]}

prop = json.loads((Path(__file__).parent / "proposed_corner_bounds.json")
                  .read_text(encoding="utf-8"))
proposed = {cid: (b["start_m"], b["end_m"]) for cid, b in prop["bounds"].items()}

cl = pd.read_parquet(corpus_dir() / f"{TRACK}_centerline.parquet").sort_values("track_dist_m")
lat0, lon0 = cl["lat"].mean(), cl["long"].mean()
cl_x = (cl["long"].to_numpy() - lon0) * M_LON
cl_y = (cl["lat"].to_numpy() - lat0) * M_LAT
cl_d = cl["track_dist_m"].to_numpy()

cmap = plt.get_cmap("tab20")
colors = {cid: cmap(i % 20) for i, cid in enumerate(ids)}


mx = 0.04 * (cl_x.max() - cl_x.min())
my = 0.06 * (cl_y.max() - cl_y.min())
XLIM = (cl_x.min() - mx, cl_x.max() + mx)
YLIM = (cl_y.min() - my, cl_y.max() + my)


def draw(ax, bounds, title):
    ax.plot(cl_x, cl_y, color="0.82", lw=1.4, zorder=1)  # full track, faint
    for cid in ids:
        s, e = bounds[cid]
        m = (cl_d >= s) & (cl_d <= e)
        ax.plot(cl_x[m], cl_y[m], color=colors[cid], lw=7, alpha=0.85,
                solid_capstyle="round", zorder=2)
        j = int(np.argmin(np.abs(cl_d - apex[cid])))
        ax.annotate(cid, (cl_x[j], cl_y[j]), fontsize=10, fontweight="bold",
                    ha="center", va="center", zorder=3,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec=colors[cid], lw=1.4))
    ax.set_aspect("equal")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_title(title, fontsize=14, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, (a1, a2) = plt.subplots(2, 1, figsize=(19, 13))
draw(a1, current, "CURRENT corner boxes")
draw(a2, proposed, "PROPOSED zones — WOT-to-WOT complexes, tiled")
fig.suptitle("Ridge Motorsports Park — corner zones: current vs proposed  "
             "(grey = un-boxed straight)", fontsize=15, y=0.99)
plt.tight_layout()
out = Path(__file__).parent / "corner_zones.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"wrote {out}")
