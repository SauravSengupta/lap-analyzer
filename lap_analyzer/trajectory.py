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
        """Map track_dist_m to corridor bin indices (clipped to range).

        Non-finite positions (NaN/inf) map to bin 0 rather than an undefined int
        cast — callers must filter such samples out of any downstream use."""
        td = np.asarray(track_dist_m, dtype=float)
        idx = np.round(np.where(np.isfinite(td), td, 0.0) / CORRIDOR_BIN_M).astype(int)
        return np.clip(idx, 0, len(self.s_bin) - 1)

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


def default_corridor(centerline: pd.DataFrame, frame: TrackFrame | None = None) -> Corridor:
    """A corridor with no corpus behind it: the σ=10m smoothed centerline field, a
    flat ±ENV_DEFAULT envelope, and zero κ. The design's cold-start path (a new
    track with <30 clean laps) and what span_time uses when no per-track corridor is
    supplied — a straight synthetic lap has κ=0 and needs no ribbon to be timed."""
    if frame is None:
        frame = TrackFrame.from_centerline(centerline)
    cd = centerline["track_dist_m"].to_numpy(dtype=float)
    cx, cy = frame.to_xy(centerline["lat"].to_numpy(), centerline["long"].to_numpy())
    grid = np.arange(0.0, float(cd.max()), CORRIDOR_BIN_M)
    n = len(grid)
    gx, gy, tx, ty = _smoothed_field(cd, cx, cy, grid)
    return Corridor(s_bin=grid, e_lo=np.full(n, ENV_DEFAULT_LO), e_hi=np.full(n, ENV_DEFAULT_HI),
                    tx=tx, ty=ty, gx=gx, gy=gy, kappa_signed=np.full(n, KAPPA_DEFAULT),
                    meta={"calib_version": CALIB_VERSION, "cold_start": True})


def load_corridor(track: str) -> Corridor:
    path = _corridor_path(track)
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("calib_version") not in (None, CALIB_VERSION):
        raise ValueError(
            f"corridor {track} calib_version {meta.get('calib_version')!r} != {CALIB_VERSION!r}; "
            f"rebuild it: python -m lap_analyzer.trajectory build-corridor {track}")
    stored_hash = meta.get("input_hash")
    if stored_hash is not None and stored_hash != _centerline_hash(load_centerline(track)):
        raise ValueError(
            f"corridor {track} was built from a different centerline (input_hash "
            f"{stored_hash!r} stale); rebuild it: python -m lap_analyzer.trajectory "
            f"build-corridor {track}")
    return Corridor(
        s_bin=df["s_bin"].to_numpy(), e_lo=df["e_lo"].to_numpy(), e_hi=df["e_hi"].to_numpy(),
        tx=df["tx"].to_numpy(), ty=df["ty"].to_numpy(), gx=df["gx"].to_numpy(),
        gy=df["gy"].to_numpy(), kappa_signed=df["kappa_signed"].to_numpy(), meta=meta)


# --- estimator (Stage 2) ----------------------------------------------------
# Provenance-carrying constants (design R9); values from the validated prototype
# (shat_prototype.py) + plan §6 signal-constants block.
FRESH_FIX_M = 0.01           # 2026-07-11, |Δtrack_dist|>this = a fresh GPS fix (prototype)
SNAP_MASK_M = 200.0          # 2026-07-11, |δ − lap median|>this = kd-tree wrong-segment snap (design brief 8.6%/600-2700m)
RESCALE_INVALID_M = 150.0    # 2026-07-11, |median δ|>this → status rescale_invalid (session first/last lap ~-700m)
HAMPEL_WINDOW = 31           # 2026-07-11, Hampel teleport window (prototype)
HAMPEL_NSIGMA = 5.0          # 2026-07-11, Hampel k·MAD (prototype)
TELEPORT_BACKSTOP_M = 15.0   # 2026-07-11, Hampel absolute floor for the residual (prototype uses max(15, 5·MAD))
EVIDENCE_BACKSTOP_M = 60.0   # 2026-07-11, absolute teleport backstop on |δ − ev median| (design brief 50-200m)
ENV_SOFT_SCALE_M = 5.0       # 2026-07-11, w_env = exp(−(excess/this)²) (design R4.1)
KNOT_SPACING_M = 10.0        # 2026-07-11, δ̂ knots every 10m of dist_lap_m (prototype)
BANDWIDTH_M = 60.0           # 2026-07-11, tricube bandwidth; curvature-adaptive is §9.2/PR1-T5 (prototype)
SIGMA_MEAS_M = 3.0           # 2026-07-11, per-fix measurement σ floor (plan §6)
SIGMA_PRIOR_END_M = 2.0      # 2026-07-11, δ=0 prior σ at lap ends (rescale pins them) (plan §6)
SIGMA_PRIOR_MID_M = 12.0     # 2026-07-11, δ=0 prior σ mid-lap ≡ measured ±0.55s fallback noise (plan §6)
PRIOR_TAPER_M = 200.0        # 2026-07-11, ends→mid prior taper length (prototype)
SLOPE_BOUND_KAPPA = 20.0     # 2026-07-11, slope_bound = min(κ·this+0.10, 0.80) m/m (prototype/plan step 4)
SLOPE_BOUND_FLOOR = 0.10     # 2026-07-11, straight-line offset-slope floor (plan §6 p95=0.111)
SLOPE_BOUND_CAP = 0.80       # 2026-07-11, max plausible offset slope (prototype)
SIGMA_FIT_CAP_M = 15.0       # 2026-07-11, cap on SE_fit before combining (prototype)
SIGMA_FLOOR_OBD_M = 1.0      # 2026-07-11, σ floor with OBD backbone (design §Uncertainty)
SIGMA_FLOOR_GPS_M = 3.0      # 2026-07-11, σ floor on GPS-only backbone (design §Architecture)

# --- section-timing constants (Stage 2, item 5) -----------------------------
# ρ for the correlation-aware section variance Var(T)=[σ_A²+σ_B²−2ρσ_Aσ_B]/(v_A v_B)
# (judge-mandated; independence is wrong-signed for Mode 4). Fitted 2026-07-12 as
# the Fisher-z weighted-mean offset autocorrelation δ=track_dist−dist_lap between
# each corner's entry/exit gates over the 95 clean corpus laps (= 0.454); the
# common-mode Mode-1 offset is positively correlated (T7 0.88, T14 0.79) so section
# times sharpen, while Mode-4 corners go negative when the constant is removed
# (T8 −0.50, T11 −0.21) — that widening is carried by the driven-band/consistency
# σ-nets and validated per-mode by the R7 battery.
RHO_SECTION = 0.45
# driven-band net: band = (∫|κ| ds over [a,b])·6 + 7 m. The heading integral of the
# corridor κ reproduces the plan §6 band_m table (heading_rad·6+7) within ~1m per
# corner (verified 2026-07-12 vs corner_bounds_verdict.csv hdg_deg / baseline
# heading_rad). NOTE: the §6 note's "max(…, clean p99.5+2 from all_rel_devs)" branch
# does NOT bind — all_rel_devs is the contaminated *reliable* set (p99.5|dev| 35–67m),
# so using it would widen the band 2–3× and let the T8 fakes (dev −31…−37m) pass,
# defeating the net (a §9.7-forbidden widening). §6 band_m == heading formula; used.
DRIVEN_BAND_HDG_SCALE = 6.0
DRIVEN_BAND_FLOOR_M = 7.0
# line-length consistency net: resid = driven − [(b−a) + Σ κ_signed·e_left·Δs];
# clean std 6.6m (design §Prototype / item 5). Inflate σ beyond ±2σ.
CONSISTENCY_STD_M = 6.6
SPEED_CONSIST_TOL_S = 0.2     # 2026-07-11, speed-consistency slack (design tripwire)
TIER_A_S = 0.10              # 2026-07-11, rankable σ_t ceiling (PROVISIONAL until R7, §9.1)
TIER_B_S = 0.30             # 2026-07-11, shown-shaded σ_t ceiling (PROVISIONAL until R7, §9.1)


@dataclass
class Trajectory:
    """Per-lap along-track estimate on the canonical ruler; s_hat monotone."""

    t: np.ndarray               # time-sorted sample times
    s_hat: np.ndarray           # car position on the ruler = maximum.accumulate(dl + delta_hat)
    sigma_m: np.ndarray         # per-sample 1σ (metres) interpolated from the knot σ
    delta_hat: np.ndarray       # per-sample δ̂ (track_dist − dist_lap correction)
    dl: np.ndarray              # per-sample dist_lap_m (odometer); the knot σ grid is in dl space
    v: np.ndarray               # per-sample speed (m/s), for section σ_t = σ_s/v
    evidence: np.ndarray        # bool: sample contributed accepted GPS evidence
    status: str                 # 'ok' | 'rescale_invalid' | 'gps_backbone'
    knots: np.ndarray           # dist_lap_m knot grid
    knot_sigma: np.ndarray      # σ at each knot (metres)
    e_lat: np.ndarray | None = None  # drift-corrected signed lateral offset per sample
    checks: list = field(default_factory=list)  # structured audit records (R12)

    def time_at(self, s: float):
        """(t, sigma_t) at ruler position s, or None if s is outside the lap's
        s_hat range — a scalar ruler position crosses the monotone s_hat once, and
        out-of-range means no crossing (the caller emits no_coverage, never a
        fabricated clamped time)."""
        if not (self.s_hat[0] <= s <= self.s_hat[-1]):
            return None
        t = float(np.interp(s, self.s_hat, self.t))
        return t, self.sigma_at(s) / max(self.v_at(t), MIN_SPEED_MPS)

    def sigma_at(self, s: float) -> float:
        """σ (metres) at ruler position s. δ̂ is not constant, so invert s→dl
        through s_hat first, then read the knot σ (which lives in dl space)."""
        dl_at_s = float(np.interp(s, self.s_hat, self.dl))
        return float(np.interp(dl_at_s, self.knots, self.knot_sigma))

    def v_at(self, t: float) -> float:
        return float(np.interp(t, self.t, self.v))


def _check(name, value, threshold, ok):
    return {"name": name, "value": round(float(value), 3),
            "threshold": round(float(threshold), 3), "pass": bool(ok)}


def estimate_trajectory(lap_samples: pd.DataFrame, corridor: Corridor,
                        frame: TrackFrame) -> Trajectory:
    """Robust corridor-weighted, slope-bounded δ̂ smooth + honest per-sample σ.

    Ports shat_prototype.estimate onto the Corridor. Steps (design §Architecture):
    evidence extraction → corridor weighting → robust knot fit → slope-bound
    projection → δ=0 prior blend → σ terms → s_hat = maximum.accumulate(dl + δ̂).
    """
    g = lap_samples.sort_values("t", kind="stable")
    t = g["t"].to_numpy(dtype=float)
    dl = g["dist_lap_m"].to_numpy(dtype=float)
    td = g["track_dist_m"].to_numpy(dtype=float)
    checks = []

    obd = g["speed_mph"].to_numpy(dtype=float)
    has_obd = bool(np.isfinite(obd).any())
    v = (obd if has_obd else g["speed_mph_gps"].to_numpy(dtype=float)) * MPH_TO_MPS
    v = np.where(np.isfinite(v), v, MIN_SPEED_MPS)

    # Drift-corrected signed lateral offset for EVERY sample (R5) — the consistency
    # net reads it even on prior-only laps; NaN where the lap carries no position.
    e_lat_all = _lap_lateral_offset(g, corridor, frame, td)
    # GPS-only laps (no OBD): dist_lap_m is already integrated from speed_mph_gps by
    # normalize, so it is a valid backbone — but the lap rides on GPS alone, so it
    # carries status='gps_backbone' and the wider GPS σ floor throughout (plan §4.1).
    backbone_status = "gps_backbone" if not has_obd else "ok"
    sigma_floor = SIGMA_FLOOR_GPS_M if not has_obd else SIGMA_FLOOR_OBD_M

    delta = td - dl
    med_delta = float(np.nanmedian(delta))
    n = len(td)

    # --- rescale sanity: a bad dist_lap_m rescale (session first/last lap) is prior-only
    if abs(med_delta) > RESCALE_INVALID_M or not np.isfinite(med_delta):
        checks.append(_check("rescale_median_delta", med_delta, RESCALE_INVALID_M, False))
        return _prior_only_trajectory(t, dl, v, "rescale_invalid", corridor, checks,
                                      e_lat=e_lat_all)

    # --- Step 1: evidence extraction (fresh → snap-mask → Hampel) ---
    fresh = np.ones(n, bool)
    fresh[1:] = np.abs(np.diff(td)) > FRESH_FIX_M
    ok = fresh & np.isfinite(delta) & (np.abs(delta - med_delta) < SNAP_MASK_M)
    d_ok = pd.Series(np.where(ok, delta, np.nan))
    med = d_ok.rolling(HAMPEL_WINDOW, center=True, min_periods=5).median()
    mad = (d_ok - med).abs().rolling(HAMPEL_WINDOW, center=True, min_periods=5).median() * 1.4826
    keep = (np.abs(delta - med.to_numpy()) <=
            np.maximum(TELEPORT_BACKSTOP_M, HAMPEL_NSIGMA * mad.fillna(SIGMA_MEAS_M).to_numpy()))
    ev = ok & keep
    if ev.any():
        ev &= np.abs(delta - np.median(delta[ev])) < EVIDENCE_BACKSTOP_M
    checks.append(_check("evidence_fraction", ev.mean(), 0.0, ev.any()))

    # No usable GPS evidence → prior-only (OBD/GPS backbone; nothing dropped)
    if not ev.any():
        return _prior_only_trajectory(t, dl, v, backbone_status, corridor, checks,
                                      sigma_floor=sigma_floor, e_lat=e_lat_all)

    # --- Step 2: corridor weighting (the Mode-4 discriminator) ---
    e_lat = e_lat_all
    b = corridor.bin_index(td)
    excess = np.maximum(0.0, np.maximum(corridor.e_lo[b] - e_lat, e_lat - corridor.e_hi[b]))
    w_env = np.exp(-((excess / ENV_SOFT_SCALE_M) ** 2))
    checks.append(_check("w_env_mean_evidence", float(w_env[ev].mean()), 0.0, True))

    # --- Steps 3-6: knot fit → slope bound → prior blend → σ ---
    knots = np.arange(dl.min(), dl.max(), KNOT_SPACING_M)
    if len(knots) < 2:
        return _prior_only_trajectory(t, dl, v, backbone_status, corridor, checks,
                                      sigma_floor=sigma_floor)
    dh, se, neff = _knot_fit(dl[ev], delta[ev], w_env[ev], knots)
    slope_bound = _slope_bound(knots, dl, v, g["lat_g"].to_numpy(dtype=float), corridor, td)
    dh, clip_mag = _rate_limit(dh, slope_bound, knots)
    sig = _knot_sigma(dh, se, clip_mag, knots, dl[ev], slope_bound)
    dh, sig = _prior_blend(dh, sig, knots)
    sig = np.maximum(sig, sigma_floor)

    d_smp = np.interp(dl, knots, dh)
    s_hat = np.maximum.accumulate(dl + d_smp)
    return Trajectory(t=t, s_hat=s_hat, sigma_m=np.interp(dl, knots, sig), delta_hat=d_smp,
                      dl=dl, v=v, evidence=ev, status=backbone_status, knots=knots,
                      knot_sigma=sig, e_lat=e_lat_all, checks=checks)


def _prior_only_trajectory(t, dl, v, status, corridor, checks,
                           sigma_floor=SIGMA_FLOOR_OBD_M, e_lat=None):
    """δ̂ ≡ 0 (OBD/GPS backbone), σ = the tapered δ=0 prior — reproduces the
    OBD-anchored fallback in the zero-GPS-evidence limit (design R6)."""
    dl = dl.astype(float)
    knots = np.arange(dl.min(), dl.max(), KNOT_SPACING_M) if dl.max() > dl.min() else dl[:1]
    sig = np.maximum(_prior_sigma(knots), sigma_floor)
    s_hat = np.maximum.accumulate(dl)
    return Trajectory(t=t, s_hat=s_hat, sigma_m=np.interp(dl, knots, sig),
                      delta_hat=np.zeros(len(dl)), dl=dl, v=v,
                      evidence=np.zeros(len(dl), bool), status=status, knots=knots,
                      knot_sigma=sig, e_lat=e_lat, checks=checks)


def _lap_lateral_offset(g: pd.DataFrame, corridor: Corridor, frame: TrackFrame,
                        td: np.ndarray) -> np.ndarray:
    """Per-sample drift-corrected signed lateral offset from the smoothed centerline
    (metres, + = left of travel), or all-NaN when the lap has no position columns."""
    if "lat" not in g or "long" not in g:
        return np.full(len(td), np.nan)
    lat = g["lat"].to_numpy(dtype=float)
    lon = g["long"].to_numpy(dtype=float)
    if "gps_drift_lat_m" in g:  # drift-correct (R5) when the columns are present
        lat = lat - g["gps_drift_lat_m"].to_numpy(dtype=float) / M_PER_DEG_LAT
        lon = lon - g["gps_drift_lon_m"].to_numpy(dtype=float) / frame.m_per_deg_lon
    x, y = frame.to_xy(lat, lon)
    return corridor.lateral_offset(x, y, td)


def _knot_fit(di, zi, wi, knots):
    """Corridor-weighted tricube local-linear fit with 2 IRLS Huber passes."""
    dh = np.full(len(knots), np.nan)
    se = np.full(len(knots), np.inf)
    neff = np.zeros(len(knots))
    for k, s0 in enumerate(knots):
        u = np.abs(di - s0) / BANDWIDTH_M
        m = u < 1
        if m.sum() < 3:
            continue
        x = di[m] - s0
        z = zi[m]
        w = wi[m] * (1 - u[m] ** 3) ** 3
        zm = 0.0
        r = z
        for _ in range(2):
            W = w.sum()
            if W < 1e-9:
                break
            xm = np.sum(w * x) / W
            zm = np.sum(w * z) / W
            sxx = np.sum(w * (x - xm) ** 2)
            slope = np.sum(w * (x - xm) * (z - zm)) / sxx if sxx > 1e-9 else 0.0
            r = z - (zm + slope * (x - xm))
            mad_r = 1.4826 * np.median(np.abs(r - np.median(r))) + 0.5
            w = wi[m] * (1 - u[m] ** 3) ** 3 * np.minimum(1.0, 3 * mad_r / np.maximum(np.abs(r), 1e-9))
        W = w.sum()
        if W < 1e-6:
            continue
        dh[k] = zm
        neff[k] = W ** 2 / np.sum(w ** 2)
        res_sd = max(1.4826 * np.median(np.abs(r - np.median(r))), SIGMA_MEAS_M)
        se[k] = res_sd / np.sqrt(max(neff[k], 1e-9))
    return dh, se, neff


def _slope_bound(knots, dl, v, latg, corridor, td):
    """Per-knot offset-slope bound from local |lat_g| curvature (prototype step)."""
    kappa = np.zeros(len(knots))
    for k, s0 in enumerate(knots):
        m = np.abs(dl - s0) < 30
        if m.any():
            vv = np.maximum(v[m], MIN_SPEED_MPS)
            kappa[k] = np.nanmedian(np.abs(latg[m]) * 9.81 / vv ** 2)
    kappa = np.where(np.isfinite(kappa), kappa, 0.0)
    return np.minimum(kappa * SLOPE_BOUND_KAPPA + SLOPE_BOUND_FLOOR, SLOPE_BOUND_CAP)


def _rate_limit(dh, slope_bound, knots):
    """Forward+backward slope-bound projection; clip magnitude → a σ term."""
    valid = ~np.isnan(dh)
    if valid.sum() < 2:
        return np.zeros(len(knots)), np.zeros(len(knots))
    dh = np.interp(knots, knots[valid], dh[valid])
    step = KNOT_SPACING_M
    fwd = dh.copy()
    for k in range(1, len(knots)):
        lim = slope_bound[k] * step
        fwd[k] = np.clip(fwd[k], fwd[k - 1] - lim, fwd[k - 1] + lim)
    bwd = dh.copy()
    for k in range(len(knots) - 2, -1, -1):
        lim = slope_bound[k] * step
        bwd[k] = np.clip(bwd[k], bwd[k + 1] - lim, bwd[k + 1] + lim)
    merged = 0.5 * (fwd + bwd)
    return merged, np.abs(dh - merged)


def _knot_sigma(dh, se, clip_mag, knots, di, slope_bound):
    """σ² = SE_fit² + σ_gap² + σ_clip², σ_gap = slope_bound × distance-to-evidence."""
    gap = np.full(len(knots), 1e4)
    if len(di):
        ds = np.sort(di)
        j = np.searchsorted(ds, knots)
        left = np.where(j > 0, knots - ds[np.maximum(j - 1, 0)], 1e4)
        right = np.where(j < len(ds), ds[np.minimum(j, len(ds) - 1)] - knots, 1e4)
        gap = np.minimum(np.abs(left), np.abs(right))
    sig_gap = slope_bound * gap
    return np.sqrt(np.minimum(se, SIGMA_FIT_CAP_M) ** 2 + sig_gap ** 2 + clip_mag ** 2)


def _prior_sigma(knots):
    if len(knots) < 2:
        return np.full(len(knots), SIGMA_PRIOR_MID_M)
    frac = np.minimum(knots - knots[0], knots[-1] - knots) / PRIOR_TAPER_M
    return np.clip(SIGMA_PRIOR_END_M + (SIGMA_PRIOR_MID_M - SIGMA_PRIOR_END_M) * frac,
                   SIGMA_PRIOR_END_M, SIGMA_PRIOR_MID_M)


def _prior_blend(dh, sig, knots):
    """Precision-weighted blend toward the δ=0 prior (one good fix outweighs it 16:1)."""
    sig_prior = _prior_sigma(knots)
    prec_ev = 1.0 / np.maximum(sig, 0.5) ** 2
    prec_pr = 1.0 / sig_prior ** 2
    dh_b = (dh * prec_ev) / (prec_ev + prec_pr)   # prior mean is 0
    return dh_b, np.sqrt(1.0 / (prec_ev + prec_pr))


# --- section timing (Stage 2, item 5) ---------------------------------------


@dataclass
class SectionTiming:
    """One (lap, section) timing row — ALWAYS emitted (design R10). Value and
    confidence come from the SAME trajectory pass (R1); the three σ-nets inflate
    σ, they never reject."""

    time_s: float               # t_b − t_a (NaN if no_coverage)
    sigma_s: float              # correlation-aware 1σ in seconds, after the σ-nets
    t_a: float
    t_b: float
    driven_m: float             # odometer distance between the posterior crossings
    status: str                 # 'ok' | 'no_coverage' | 'rescale_invalid'
    tier: str                   # 'A' | 'B' | 'C' (derived from sigma_s)
    rank_eligible: bool         # tier == 'A' and status == 'ok'
    checks: list = field(default_factory=list)  # structured audit records (R12)


def _tier(sigma_s: float, cap_b: bool = False) -> str:
    if not np.isfinite(sigma_s):
        return "C"
    if sigma_s <= TIER_A_S and not cap_b:
        return "A"
    if sigma_s <= TIER_B_S:
        return "B"
    return "C"


def _section_heading_rad(corridor: Corridor, a: float, b: float) -> float:
    """GPS-free section heading = ∫|κ| ds over [a,b] from the corpus κ (R3)."""
    m = (corridor.s_bin >= a) & (corridor.s_bin <= b)
    return float(np.sum(np.abs(corridor.kappa_signed[m])) * CORRIDOR_BIN_M)


def _consistency_resid(traj: Trajectory, corridor: Corridor, a: float, b: float,
                       driven: float) -> float:
    """resid = driven − [(b−a) + Σ κ_signed·e_left·Δs]: does the lap's own lateral
    line explain its odometer shortening? NaN when the lap has no lateral offset."""
    e = traj.e_lat
    if e is None:
        return np.nan
    e = np.asarray(e, dtype=float)
    s = traj.s_hat
    m = (s >= a) & (s <= b) & np.isfinite(e)
    if m.sum() < 5:
        return np.nan
    bidx = corridor.bin_index(s[m])
    prof = pd.Series(e[m]).groupby(bidx).median()
    lo, hi = int(a // CORRIDOR_BIN_M), int(b // CORRIDOR_BIN_M)
    bins_in = np.arange(lo, hi + 1)
    e_bins = pd.Series([prof.get(bi, np.nan) for bi in bins_in]).interpolate(
        limit_direction="both").to_numpy()
    k_bins = corridor.kappa_signed[np.clip(bins_in, 0, len(corridor.s_bin) - 1)]
    predicted = (b - a) + float(np.nansum(k_bins * e_bins) * CORRIDOR_BIN_M)
    return driven - predicted


def section_timing(traj: Trajectory, dist_a: float, dist_b: float,
                   corridor: Corridor) -> SectionTiming:
    """Time the section [dist_a, dist_b] on one lap's estimate; ALWAYS emit a row.

    The value (t at the monotone s_hat crossings) and its σ come from the same pass
    (R1). σ is correlation-aware, then inflated (never rejected) by the driven-band
    and line-length-consistency nets; the speed-consistency tripwire is diagnostic.
    Design: docs/…/trajectory-design.md §Section timing; SPEC tests/SPEC.md.
    """
    s, t = traj.s_hat, traj.t
    checks: list = []

    # --- coverage (R10): a scalar ruler position must fall inside monotone s_hat ---
    covered = (s[0] <= dist_a <= s[-1]) and (s[0] <= dist_b <= s[-1]) and dist_b > dist_a
    checks.append(_check("coverage", float(covered), 1.0, covered))
    if not covered:
        return SectionTiming(np.nan, np.nan, np.nan, np.nan, np.nan,
                             "no_coverage", "C", False, checks)

    t_a = float(np.interp(dist_a, s, t))
    t_b = float(np.interp(dist_b, s, t))
    time_s = t_b - t_a
    v_a = max(traj.v_at(t_a), MIN_SPEED_MPS)
    v_b = max(traj.v_at(t_b), MIN_SPEED_MPS)
    vbar = 0.5 * (v_a + v_b)
    sig_a = traj.sigma_at(dist_a)
    sig_b = traj.sigma_at(dist_b)

    # --- correlation-aware base variance (seconds²) ---
    var_base = max(0.0, (sig_a ** 2 + sig_b ** 2 - 2 * RHO_SECTION * sig_a * sig_b)
                   / (v_a * v_b))

    # odometer distance between the POSTERIOR crossings (R4.2)
    driven = float(np.interp(t_b, t, traj.dl) - np.interp(t_a, t, traj.dl))

    # --- net 1: driven-band (σ inflation, never a rejector) ---
    band = _section_heading_rad(corridor, dist_a, dist_b) * DRIVEN_BAND_HDG_SCALE + DRIVEN_BAND_FLOOR_M
    dev = driven - (dist_b - dist_a)
    excess_band = max(0.0, abs(dev) - band)
    sig_band = excess_band / vbar
    checks.append(_check("driven_band_dev_m", dev, band, abs(dev) <= band))

    # --- net 2: line-length consistency (σ inflation) ---
    resid = _consistency_resid(traj, corridor, dist_a, dist_b, driven)
    if np.isfinite(resid):
        excess_resid = max(0.0, abs(resid) - 2 * CONSISTENCY_STD_M)
        sig_resid = excess_resid / vbar
        checks.append(_check("consistency_resid_m", resid, 2 * CONSISTENCY_STD_M,
                             abs(resid) <= 2 * CONSISTENCY_STD_M))
    else:
        sig_resid = 0.0
        checks.append(_check("consistency_resid_m", np.nan, 2 * CONSISTENCY_STD_M, True))

    sigma_s = float(np.sqrt(var_base + sig_band ** 2 + sig_resid ** 2))

    # --- net 3: speed-consistency tripwire (DIAGNOSTIC only, never inflates) ---
    obd_time = float(np.interp(dist_b, traj.dl, t) - np.interp(dist_a, traj.dl, t))
    delta_a = float(np.interp(dist_a, s, traj.delta_hat))
    delta_b = float(np.interp(dist_b, s, traj.delta_hat))
    tol = (abs(delta_a) + abs(delta_b)) / vbar + SPEED_CONSIST_TOL_S
    checks.append(_check("speed_consistency_s", time_s - obd_time, tol,
                         abs(time_s - obd_time) <= tol))

    status = "rescale_invalid" if traj.status == "rescale_invalid" else "ok"
    tier = _tier(sigma_s, cap_b=(traj.status == "gps_backbone"))
    rank_eligible = (tier == "A") and (status == "ok")
    return SectionTiming(time_s, sigma_s, t_a, t_b, driven, status, tier,
                         rank_eligible, checks)


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
