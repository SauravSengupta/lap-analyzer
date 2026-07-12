"""Unified GPS-trust trajectory layer — Stage 1: the per-track corridor.

The corridor is the "asphalt ribbon": per-10m-bin p2/p98 of clean-lap signed
lateral offset from the centerline (the Mode-4 discriminator), plus the σ=10m
smoothed centerline field (positions + tangents, GPS-noise immune) and the
corpus clean-lap signed curvature (κ from |lat_g|·g/v², GPS-free — never from
centerline XY, per design R3). Every trajectory consumer reads one corridor.

Design: docs/superpowers/specs/2026-07-11-unified-gps-trust-trajectory-design.md
Plan:   docs/superpowers/plans/2026-07-11-unified-gps-trust-implementation.md §4.1

Later stages add estimate_trajectory() and section_timing() to this module.

CLI:  python -m lap_analyzer.trajectory build-corridor ridge
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .analysis import load_centerline
from .centerline import M_PER_DEG_LAT
from .config import corpus_dir, sessions_dir
from .gates import TrackFrame, _smoothed_tangent

# --- provenance-carrying constants (design R9) ------------------------------
CORRIDOR_BIN_M = 10.0          # 2026-07-11, canonical 10m corridor bin (design §4.1)
ENV_PAD_M = 2.0               # 2026-07-11, ±2m antenna asymmetry pad (right-mount, project_gps_antenna_position)
ENV_CAP_M = 20.0             # 2026-07-11, hard ±20m envelope cap (design §2 R4.1)
ENV_MIN_LAPS = 30            # 2026-07-11, min clean-lap votes/bin before trusting p2/p98 (prototype)
KAPPA_MIN_LAPS = 20         # 2026-07-11, min votes/bin for a corpus κ median (consistency_check.py)
KAPPA_DEFAULT = 0.0         # sparse-bin κ fallback (straight)
ENV_DEFAULT_LO = -15.0      # 2026-07-11, cold-bin envelope fallback (prototype)
ENV_DEFAULT_HI = 15.0
SMOOTH_SIGMA_M = 10.0        # 2026-07-11, σ=10m position/tangent smoothing (design R3)
SMOOTH_HALF_WINDOW_M = 20.0  # 2026-07-11, ±20m smoothing window (>2 snake wavelengths, R3)
CLEAN_MEDIAN_DELTA_M = 100.0  # 2026-07-11, |median(track_dist−dist_lap)| clean-lap gate (prototype)
CLEAN_MAX_DELTA_M = 40.0     # 2026-07-11, max |offset| clean-lap gate (prototype/quality STANDARD)
MIN_LAP_SAMPLES = 100       # 2026-07-11, min samples for a usable lap (prototype)
MIN_SPEED_MPS = 8.0         # 2026-07-11, floor for κ = lat_g·g/v² (avoid v→0 blowup)
MPH_TO_MPS = 0.44704
CALIB_VERSION = "corridor-v1"  # bump when the corridor schema/algorithm changes


@dataclass(frozen=True)
class Corridor:
    """Per-track corridor artifact: data/corpus/{track}_corridor.parquet."""

    s_bin: np.ndarray          # 10m bin centers
    e_lo: np.ndarray           # clean-lap p2 lateral − pad, capped at −ENV_CAP_M
    e_hi: np.ndarray           # clean-lap p98 lateral + pad, capped at +ENV_CAP_M
    tx: np.ndarray             # σ=10m-smoothed centerline tangent x (unit), signs offsets only
    ty: np.ndarray             # σ=10m-smoothed centerline tangent y (unit)
    gx: np.ndarray             # σ=10m-smoothed centerline position x (TrackFrame metres)
    gy: np.ndarray             # σ=10m-smoothed centerline position y
    kappa_signed: np.ndarray   # corpus clean-lap median lat_g·g/v² per bin (signed, +=right)
    meta: dict = field(default_factory=dict)  # calib_version, input hash, n_laps, build date

    def bin_index(self, track_dist_m: np.ndarray) -> np.ndarray:
        """Map track_dist_m to corridor bin indices (clipped to range)."""
        return np.clip(np.round(np.asarray(track_dist_m) / CORRIDOR_BIN_M).astype(int),
                       0, len(self.s_bin) - 1)

    def lateral_offset(self, x: np.ndarray, y: np.ndarray, track_dist_m: np.ndarray) -> np.ndarray:
        """Signed lateral offset (metres, + = left of travel) of frame-XY points
        from the smoothed centerline. `x,y` must already be drift-corrected
        (design R5) and in the same TrackFrame as gx/gy."""
        b = self.bin_index(track_dist_m)
        return -self.ty[b] * (x - self.gx[b]) + self.tx[b] * (y - self.gy[b])


# --- smoothed centerline field ----------------------------------------------

def _smoothed_field(cd: np.ndarray, cx: np.ndarray, cy: np.ndarray, grid: np.ndarray,
                    sigma_m: float = SMOOTH_SIGMA_M, half_window_m: float = SMOOTH_HALF_WINDOW_M):
    """σ=10m Gaussian-smoothed positions (weighted mean) and unit tangents (from
    gates._smoothed_tangent — the SAME fit that orients gates, design R3) on `grid`."""
    gx = np.empty(len(grid))
    gy = np.empty(len(grid))
    tx = np.empty(len(grid))
    ty = np.empty(len(grid))
    for i, s in enumerate(grid):
        win = np.abs(cd - s) <= half_window_m
        if win.any():
            w = np.exp(-0.5 * ((cd[win] - s) / sigma_m) ** 2)
            sw = w.sum()
            gx[i] = np.sum(w * cx[win]) / sw
            gy[i] = np.sum(w * cy[win]) / sw
        else:
            gx[i], gy[i] = cx[int(np.argmin(np.abs(cd - s)))], cy[int(np.argmin(np.abs(cd - s)))]
        tan = _smoothed_tangent(cd, cx, cy, float(s), sigma_m, half_window_m)
        if tan is None:
            tx[i], ty[i] = 1.0, 0.0
        else:
            tx[i], ty[i] = tan
    return gx, gy, tx, ty


# --- corpus iteration -------------------------------------------------------

_CORRIDOR_COLUMNS = ["lap", "track_dist_m", "dist_lap_m", "lat", "long", "lat_g",
                     "speed_mph", "gps_drift_lat_m", "gps_drift_lon_m"]


def _drift_corrected_xy(frame: TrackFrame, g: pd.DataFrame, m_per_deg_lon: float):
    """Frame-XY of a lap's GPS path with the per-lap drift vector removed (R5)."""
    lat = g["lat"].to_numpy() - g["gps_drift_lat_m"].to_numpy() / M_PER_DEG_LAT
    lon = g["long"].to_numpy() - g["gps_drift_lon_m"].to_numpy() / m_per_deg_lon
    return frame.to_xy(lat, lon)


def _iter_clean_laps(track: str):
    """Yield (session_id, lap, DataFrame) for every clean flying lap in the corpus.

    Clean = enough samples, sane dist_lap_m rescale, no large GPS excursion — the
    prototype's envelope-population gate. Missing drift columns default to 0."""
    sdir = sessions_dir(track)
    for sd in sorted(sdir.iterdir()):
        sp = sd / "samples.parquet"
        if not sp.exists():
            continue
        try:
            import pyarrow.parquet as pq
            have = set(pq.read_schema(sp).names)
            cols = [c for c in _CORRIDOR_COLUMNS if c in have]
            if not {"lap", "track_dist_m", "dist_lap_m", "lat", "long"} <= have:
                continue
            df = pd.read_parquet(sp, columns=cols)
        except Exception:
            continue
        if df["track_dist_m"].isna().all():
            continue
        for c in _CORRIDOR_COLUMNS:
            if c not in df.columns:
                df[c] = 0.0 if c.startswith("gps_drift") else np.nan
        for lap_n, g in df.groupby("lap"):
            td = g["track_dist_m"].to_numpy()
            dl = g["dist_lap_m"].to_numpy()
            if len(td) < MIN_LAP_SAMPLES:
                continue
            delta = td - dl
            if abs(np.nanmedian(delta)) > CLEAN_MEDIAN_DELTA_M or np.nanmax(np.abs(delta)) > CLEAN_MAX_DELTA_M:
                continue
            yield sd.name, int(lap_n), g


def _per_lap_bin_medians(offset: np.ndarray, bins: np.ndarray, n_bins: int) -> dict:
    """One vote per lap per bin: median offset in each bin the lap visits."""
    return pd.DataFrame({"b": bins, "e": offset}).groupby("b")["e"].median().to_dict()


def build_corridor(track: str, save: bool = True) -> Corridor:
    """Build the per-track corridor from the clean-lap corpus (two-pass EM-trim).

    Pass 1 fits a raw p2/p98 envelope; pass 2 recomputes it after dropping the
    individual fixes pass 1 rejects, purging Mode-4 contamination that would
    otherwise widen the ribbon (design §Architecture; convergence tested).
    """
    centerline = load_centerline(track)
    frame = TrackFrame.from_centerline(centerline)
    m_per_deg_lon = frame.m_per_deg_lon
    cd = centerline["track_dist_m"].to_numpy()
    cx, cy = frame.to_xy(centerline["lat"].to_numpy(), centerline["long"].to_numpy())
    L = float(cd.max())
    grid = np.arange(0.0, L, CORRIDOR_BIN_M)
    n_bins = len(grid)
    gx, gy, tx, ty = _smoothed_field(cd, cx, cy, grid)
    corr = Corridor(s_bin=grid, e_lo=np.full(n_bins, ENV_DEFAULT_LO),
                    e_hi=np.full(n_bins, ENV_DEFAULT_HI), tx=tx, ty=ty, gx=gx, gy=gy,
                    kappa_signed=np.full(n_bins, KAPPA_DEFAULT))

    # Gather each clean lap's per-bin offset votes + κ votes once.
    lap_votes = []          # list of dict{bin: median_offset}
    kappa_acc = {i: [] for i in range(n_bins)}
    n_laps = 0
    for _sid, _lap, g in _iter_clean_laps(track):
        td = g["track_dist_m"].to_numpy()
        x, y = _drift_corrected_xy(frame, g, m_per_deg_lon)
        offset = corr.lateral_offset(x, y, td)
        bins = corr.bin_index(td)
        lap_votes.append((bins, offset))
        # κ = lat_g·g/v² per bin (GPS-free; NaN-safe — OBD-dropout laps have NaN speed)
        v = g["speed_mph"].to_numpy() * MPH_TO_MPS
        latg = g["lat_g"].to_numpy()
        good = np.isfinite(v) & (v > MIN_SPEED_MPS) & np.isfinite(latg)
        if good.sum() >= MIN_LAP_SAMPLES:
            k = latg[good] * 9.81 / v[good] ** 2
            kb = corr.bin_index(td[good])
            for bi, kv in pd.DataFrame({"b": kb, "k": k}).groupby("b")["k"].median().items():
                if np.isfinite(kv):
                    kappa_acc[int(bi)].append(kv)
        n_laps += 1

    e_lo, e_hi = _fit_envelope(lap_votes, n_bins, reject=None)          # pass 1
    e_lo, e_hi = _fit_envelope(lap_votes, n_bins, reject=(e_lo, e_hi))  # pass 2 (EM-trim)

    kappa = np.array([np.nanmedian(kappa_acc[i]) if len(kappa_acc[i]) >= KAPPA_MIN_LAPS
                      else KAPPA_DEFAULT for i in range(n_bins)])
    kappa = np.where(np.isfinite(kappa), kappa, KAPPA_DEFAULT)

    meta = {
        "calib_version": CALIB_VERSION,
        "input_hash": _centerline_hash(centerline),
        "n_laps": n_laps,
        "build_date": date.today().isoformat(),
        "track": track,
    }
    corridor = Corridor(s_bin=grid, e_lo=e_lo, e_hi=e_hi, tx=tx, ty=ty, gx=gx, gy=gy,
                        kappa_signed=kappa, meta=meta)
    if save:
        _save_corridor(track, corridor)
    return corridor


def _fit_envelope(lap_votes, n_bins, reject):
    """p2/p98 (+pad, ±cap) of per-lap-per-bin offset votes. If `reject=(lo,hi)`,
    drop votes outside that envelope first (EM-trim pass 2)."""
    acc = {i: [] for i in range(n_bins)}
    for bins, offset in lap_votes:
        keep = offset
        keepbins = bins
        if reject is not None:
            lo, hi = reject
            m = (offset >= lo[bins]) & (offset <= hi[bins])
            keep = offset[m]
            keepbins = bins[m]
        if len(keep) == 0:
            continue
        for bi, ev in pd.DataFrame({"b": keepbins, "e": keep}).groupby("b")["e"].median().items():
            acc[int(bi)].append(ev)
    e_lo = np.full(n_bins, ENV_DEFAULT_LO)
    e_hi = np.full(n_bins, ENV_DEFAULT_HI)
    for i in range(n_bins):
        if len(acc[i]) >= ENV_MIN_LAPS:
            p2 = np.percentile(acc[i], 2) - ENV_PAD_M
            p98 = np.percentile(acc[i], 98) + ENV_PAD_M
            e_lo[i] = max(p2, -ENV_CAP_M)
            e_hi[i] = min(p98, ENV_CAP_M)
    return e_lo, e_hi


def _centerline_hash(centerline: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(centerline[["track_dist_m", "lat", "long"]].to_numpy().tobytes())
    return h.hexdigest()[:16]


# --- persistence ------------------------------------------------------------

def _corridor_path(track: str):
    return corpus_dir() / f"{track}_corridor.parquet"


def _save_corridor(track: str, corridor: Corridor) -> None:
    df = pd.DataFrame({
        "s_bin": corridor.s_bin, "e_lo": corridor.e_lo, "e_hi": corridor.e_hi,
        "tx": corridor.tx, "ty": corridor.ty, "gx": corridor.gx, "gy": corridor.gy,
        "kappa_signed": corridor.kappa_signed,
    })
    df.attrs["meta"] = json.dumps(corridor.meta)
    path = _corridor_path(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stash meta in a sidecar (parquet attrs aren't reliably round-tripped).
    df.to_parquet(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(corridor.meta), encoding="utf-8")


def load_corridor(track: str) -> Corridor:
    path = _corridor_path(track)
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("calib_version") not in (None, CALIB_VERSION):
        raise ValueError(
            f"corridor {track} calib_version {meta.get('calib_version')!r} != {CALIB_VERSION!r}; rebuild it")
    return Corridor(
        s_bin=df["s_bin"].to_numpy(), e_lo=df["e_lo"].to_numpy(), e_hi=df["e_hi"].to_numpy(),
        tx=df["tx"].to_numpy(), ty=df["ty"].to_numpy(), gx=df["gx"].to_numpy(),
        gy=df["gy"].to_numpy(), kappa_signed=df["kappa_signed"].to_numpy(), meta=meta)


# --- CLI --------------------------------------------------------------------

def _main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m lap_analyzer.trajectory")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-corridor", help="build the per-track corridor artifact")
    b.add_argument("track")
    args = p.parse_args(argv)
    if args.cmd == "build-corridor":
        c = build_corridor(args.track)
        finite = np.isfinite(c.e_lo)
        print(f"corridor[{args.track}]: {len(c.s_bin)} bins from {c.meta['n_laps']} clean laps")
        print(f"  envelope median width {np.median((c.e_hi - c.e_lo)[finite]):.1f}m; "
              f"kappa nonzero bins {(c.kappa_signed != 0).sum()}")
        print(f"  saved -> {_corridor_path(args.track)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
