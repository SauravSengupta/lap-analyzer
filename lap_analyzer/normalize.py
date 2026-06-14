from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import data_root
from .schemas import SessionMeta


def load_session_notes(track: str) -> dict:
    notes_path = data_root() / "notes" / f"{track}.json"
    if notes_path.exists():
        return json.loads(notes_path.read_text())
    return {}


def reference_session_ids(track: str) -> set[str]:
    """Session IDs flagged `"reference": true` in data/notes/<track>.json.

    Reference sessions (e.g. an instructor driving the user's car) are fully
    normalized and labeled, but kept out of the corpus and every corpus-wide
    percentile stat — so their laps never move the user's pace deciles or
    per-corner reference medians. They remain on disk for side-by-side
    comparison."""
    return {sid for sid, n in load_session_notes(track).items() if n.get("reference")}


COLUMN_RENAME = {
    "Time": "t",
    "UTC Time": "utc",
    "Lap": "lap",
    "Sector": "sector",
    "Latitude": "lat",
    "Longitude": "long",
    "Altitude (m)": "altitude_m",
    "Speed (MPH)": "speed_mph_gps",
    "Heading": "heading",
    "Accuracy (m)": "gps_accuracy_m",
    "Accel X": "long_g",
    "Accel Y": "lat_g",
    "Accel Z": "vert_g",
    "Brake (calculated)": "brake",
    "Engine Speed (RPM) *OBD": "rpm",
    "Vehicle Speed (mph) *OBD": "speed_mph",
    "Throttle Position (%) *OBD": "throttle_raw",
    "Engine Coolant Temp (F) *OBD": "coolant_f",
    "Intake Air Temp (F) *OBD": "iat_f",
    "Intake Manifold Pressure (PSI) *OBD": "manifold_psi",
}

OBD_CHANNELS = ["rpm", "speed_mph", "throttle_raw", "coolant_f", "iat_f", "manifold_psi"]

OUTPUT_COLUMNS = [
    "session_id", "t", "lap",
    "dist_m", "dist_lap_m",
    "speed_mph", "speed_mph_gps",
    "throttle_norm", "brake", "rpm",
    "lat_g", "long_g",
    "coolant_f", "iat_f",
    "lat", "long", "altitude_m", "gps_accuracy_m",
]

SESSION_ID_RE = re.compile(r"Log-(\d{8}-\d{6})")


def session_id_from_filename(path: Path) -> str:
    m = SESSION_ID_RE.search(Path(path).name)
    if not m:
        raise ValueError(f"Cannot derive session_id from filename: {path}")
    return m.group(1)


def parse_metadata_header(path: Path) -> dict:
    meta: dict = {"raw_csv_path": str(path), "split_points": []}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if body.startswith("Vehicle:"):
                meta["vehicle"] = body.split(":", 1)[1].strip()
            elif body.startswith("End Point:"):
                coords = _parse_coords(body.split(":", 1)[1])
                if coords:
                    meta["start_finish"] = coords
            else:
                m = re.match(r"Split Point (\d+):\s*(.+)", body)
                if m and (coords := _parse_coords(m.group(2))):
                    meta["split_points"].append({"index": int(m.group(1)), **coords})
    return meta


def _parse_coords(s: str) -> dict | None:
    s = s.split("@")[0].strip().rstrip(",")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 2:
        return None
    try:
        return {"lat": float(parts[0]), "long": float(parts[1])}
    except ValueError:
        return None


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#", low_memory=False)
    return df.rename(columns=COLUMN_RENAME)


def normalize_dataframe(
    raw: pd.DataFrame,
    session_id: str,
    canonical_lap_length_m: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()

    # On this phone/orientation, Accel Y is the lateral channel (Accel X is longitudinal).
    # Raw Accel Y is positive on left turns; negate so canonical lat_g is positive = right.
    # Raw Accel X is positive under braking; negate so canonical long_g is positive = accel.
    df["lat_g"] = -df["lat_g"]
    df["long_g"] = -df["long_g"]

    # OBD channels only update on OBD ticks; forward-fill so every row has the most recent reading.
    # For GPS-only (OBD-dropout) sessions these columns are all-NaN and stay NaN.
    df[OBD_CHANNELS] = df[OBD_CHANNELS].ffill().bfill()

    # Per-session throttle max is the "true 100%" for this car/sensor (Porsche pedals top out ~90% raw).
    # GPS-only sessions have no throttle channel -> throttle_norm is NaN everywhere (distinct from
    # the all-zero-throttle case, which is 0.0).
    throttle_max = float(df["throttle_raw"].max())
    if np.isnan(throttle_max):
        df["throttle_norm"] = np.nan
    elif throttle_max > 0:
        df["throttle_norm"] = df["throttle_raw"] / throttle_max
    else:
        df["throttle_norm"] = 0.0

    # Distance-from-start: trapezoidal integration of speed (mph -> m/s). Use OBD speed when
    # present, else GPS speed (GPS-only sessions) so dist_m is still a real monotonic ruler.
    speed_col = "speed_mph" if df["speed_mph"].notna().any() else "speed_mph_gps"
    speed_ms = df[speed_col].fillna(0).to_numpy() * 0.44704
    t = df["t"].to_numpy()
    dist = np.zeros(len(df))
    if len(df) > 1:
        dt = np.diff(t)
        dist[1:] = np.cumsum(0.5 * (speed_ms[:-1] + speed_ms[1:]) * dt)
    df["dist_m"] = dist
    df["dist_lap_m"] = df["dist_m"] - df.groupby("lap")["dist_m"].transform("first")

    # Per-lap dist_lap_m rescaling: integration error makes total lap distance vary 0.5-1.5%
    # across nominally-identical laps. Rescale each lap to the track's canonical length so
    # "dist_lap_m = X" means the same physical point across laps.
    if canonical_lap_length_m is not None:
        lap_max = df.groupby("lap")["dist_lap_m"].transform("max")
        scale = np.where(lap_max > 0, canonical_lap_length_m / lap_max, 1.0)
        df["dist_lap_m"] = df["dist_lap_m"] * scale

    df["session_id"] = session_id
    df["lap"] = df["lap"].astype("int32")
    df["brake"] = df["brake"].fillna(0).astype("int8")
    # rpm is int32 when present; GPS-only sessions keep it as float NaN (no OBD).
    if df["rpm"].notna().any():
        df["rpm"] = df["rpm"].fillna(0).astype("int32")

    return df[OUTPUT_COLUMNS], {"throttle_max_observed": throttle_max}


def compute_lap_times(df: pd.DataFrame) -> pd.DataFrame:
    starts = df.groupby("lap")["t"].first()
    last_t = df.groupby("lap")["t"].last()
    laps_sorted = sorted(starts.index.tolist())
    rows = []
    for i, lap in enumerate(laps_sorted):
        if i + 1 < len(laps_sorted):
            lap_time = float(starts[laps_sorted[i + 1]] - starts[lap])
        else:
            # Final lap is incomplete (in-lap); use end-of-data minus start.
            lap_time = float(last_t[lap] - starts[lap])
        rows.append({"lap": int(lap), "lap_time_s": round(lap_time, 3)})
    return pd.DataFrame(rows)


def lap_summary(df: pd.DataFrame, lap_times: pd.DataFrame) -> pd.DataFrame:
    aug = df.assign(
        abs_lat=df["lat_g"].abs(),
        neg_long=-df["long_g"],
        at_wot=(df["throttle_norm"] >= 0.95).astype(int),
        braking=(df["brake"] == 1).astype(int),
    )
    summary = aug.groupby("lap").agg(
        lap_dist_m=("dist_lap_m", "max"),
        max_speed_mph=("speed_mph", "max"),
        avg_speed_mph=("speed_mph", "mean"),
        max_rpm=("rpm", "max"),
        max_lat_g=("abs_lat", "max"),
        max_accel_g=("long_g", "max"),
        max_decel_g=("neg_long", "max"),
        pct_wot=("at_wot", "mean"),
        avg_throttle=("throttle_norm", "mean"),
        pct_braking=("braking", "mean"),
        coolant_min_f=("coolant_f", "min"),
        coolant_max_f=("coolant_f", "max"),
        iat_min_f=("iat_f", "min"),
        iat_max_f=("iat_f", "max"),
    ).reset_index().merge(lap_times, on="lap")

    summary["session_id"] = df["session_id"].iloc[0]

    # Lap-level cleanness is shape only (warmup/cooldown). Quality filtering belongs in the
    # corner-transit table — traffic on one corner doesn't poison the rest of the lap.
    laps_sorted = sorted(summary["lap"].tolist())
    first_lap, last_lap = laps_sorted[0], laps_sorted[-1]

    def reason(row):
        if row["lap"] == first_lap:
            return "warmup"
        if row["lap"] == last_lap:
            return "cooldown"
        return ""

    summary["clean_reason"] = summary.apply(reason, axis=1)
    summary["is_clean"] = summary["clean_reason"] == ""

    cols = [
        "session_id", "lap", "lap_time_s", "lap_dist_m",
        "max_speed_mph", "avg_speed_mph", "max_rpm",
        "max_lat_g", "max_accel_g", "max_decel_g",
        "pct_wot", "avg_throttle", "pct_braking",
        "coolant_min_f", "coolant_max_f", "iat_min_f", "iat_max_f",
        "is_clean", "clean_reason",
    ]
    return summary[cols]


def build_session_meta(
    session_id: str,
    track: str,
    raw_meta: dict,
    derived: dict,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    first_utc: float,
    has_obd: bool = True,
) -> SessionMeta:
    flying = summary[summary["is_clean"]]
    if len(flying):
        idx = flying["lap_time_s"].idxmin()
        best_lap = int(flying.loc[idx, "lap"])
        best_lap_time = float(flying.loc[idx, "lap_time_s"])
    else:
        best_lap, best_lap_time = None, None

    duration = float(df["t"].iloc[-1] - df["t"].iloc[0])
    sample_rate = (len(df) - 1) / duration if duration > 0 else 0.0
    has_coolant = df["coolant_f"].notna().any()
    has_iat = df["iat_f"].notna().any()
    throttle_max = derived["throttle_max_observed"]

    return SessionMeta(
        session_id=session_id,
        track=track,
        vehicle=raw_meta.get("vehicle", "unknown"),
        date_utc=datetime.fromtimestamp(first_utc, tz=timezone.utc),
        n_laps=int(summary["lap"].nunique()),
        n_clean_laps=int(summary["is_clean"].sum()),
        sample_rate_hz=round(sample_rate, 2),
        duration_s=round(duration, 2),
        best_lap=best_lap,
        best_lap_time_s=best_lap_time,
        has_obd=has_obd,
        throttle_max_observed=round(throttle_max, 3) if not np.isnan(throttle_max) else None,
        speed_max_obd_mph=float(df["speed_mph"].max()) if has_obd else None,
        rpm_max=int(df["rpm"].max()) if has_obd else None,
        coolant_min_f=float(df["coolant_f"].min()) if has_coolant else None,
        coolant_max_f=float(df["coolant_f"].max()) if has_coolant else None,
        # First-sample IAT before the engine bay heats up — rough ambient proxy.
        iat_first_f=float(df["iat_f"].iloc[0]) if has_iat else None,
        iat_max_f=float(df["iat_f"].max()) if has_iat else None,
        trackaddict_start_finish=raw_meta.get("start_finish"),
        trackaddict_split_points=raw_meta.get("split_points", []),
        raw_csv_path=raw_meta["raw_csv_path"],
    )


def _canonical_lap_length(track: str) -> float | None:
    """Read tracks/<track>.json's lap_length_internal_m if present."""
    from .config import tracks_dir
    path = tracks_dir() / f"{track}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    val = data.get("lap_length_internal_m")
    return float(val) if val else None


def normalize_session(csv_path: Path, track: str, out_dir: Path) -> SessionMeta:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)

    session_id = session_id_from_filename(csv_path)
    raw_meta = parse_metadata_header(csv_path)
    raw = read_csv(csv_path)

    # OBD-dropout sessions (~12% of logs) are ingested GPS-only rather than rejected:
    # create the missing OBD channels as all-NaN so the rest of the pipeline runs
    # unchanged, and flag has_obd=False so downstream stats can filter them out.
    missing = [c for c in OBD_CHANNELS if c not in raw.columns]
    has_obd = not missing
    for c in missing:
        raw[c] = np.nan

    first_utc = float(raw["utc"].iloc[0])

    canonical_length = _canonical_lap_length(track)
    df, derived = normalize_dataframe(raw, session_id, canonical_lap_length_m=canonical_length)
    lap_times = compute_lap_times(df)
    summary = lap_summary(df, lap_times)
    meta = build_session_meta(
        session_id, track, raw_meta, derived, df, summary, first_utc, has_obd=has_obd
    )

    session_dir = out_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(session_dir / "samples.parquet", engine="pyarrow", index=False)
    summary.to_csv(session_dir / "laps.csv", index=False)
    (session_dir / "meta.json").write_text(meta.model_dump_json(indent=2))
    return meta
