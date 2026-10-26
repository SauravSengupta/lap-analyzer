"""Derive the per-corner accel-speed references by their rule (W3 amendment + session
accelerometer calibration, user decision 2026-10-25).

Rule: over the GPS-only emulation (validation_checks.gps_only_emulation) of the
CALIBRATION-split sessions that have clean OBD laps, per (track, corner):
  every lap of a session (not rescale_invalid) feeds the session's accelerometer fit
  (trajectory.fit_accel_params on raw resid − offset_c vs section T and ∫lat_g dt) —
  exactly what production fits;
  only status=='ok' sections of clean laps feed the references:
  offset_mps = median(raw resid − session correction)     (rounded to 0.01 m/s)
  tol_mps    = p99 |corrected − offset|, rounded UP to 0.1 m/s
  n          = number of finite corrected residuals; corners with n < MIN_N get NO entry.
Offsets and session fits depend on each other, so they are alternated to a fixed point:
fit every session against the current offsets, median-centre the session parameters per
track (the track-wide part belongs to the per-corner offsets), recompute the offsets;
stop when no offset moves by more than CONVERGE_MPS. A session too short to fit
(trajectory.ACCEL_FIT_MIN_SECTIONS) contributes no references — production cannot verify
it either. Raw residuals are read from section_timing(...).checks called WITHOUT a
reference (the production code path). Prints the per-corner table; with --write it
updates tracks/{track}.json's `accel_speed_check` block in place (every other byte of the
file is left as it is). Local; needs data/.

  $env:PYTHONPATH = "."; python scripts/derive_accel_speed_tol.py [--write]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validation_checks as vc

from lap_analyzer.config import sessions_dir, tracks_dir
from lap_analyzer.trajectory import (
    AccelCalibration,
    accel_correction,
    accel_ref_for,
    accel_section_features,
    estimate_trajectory,
    fit_accel_calibration,
    fit_accel_params,
    section_timing,
)

TRACKS = ("pir", "ridge")
CHECK = "accel_speed_resid_mps"
BLOCK_KEY = "accel_speed_check"
MIN_N = 20                                    # 2026-10-24, fewer clean sections → no reference
CONVERGE_MPS = 0.005                          # 2026-10-25, half the 0.01 offset rounding
MAX_ITERS = 100
PROBE = ("pir", "20260926-111509", 2, "T1")   # confirmed fake (final W3 review); sanity only


def p99_ceil(values) -> float:
    """p99 of |values| (non-finite ignored), rounded UP to the 0.1 grid. A value already
    on the grid is not bumped (the 1e-9 guard absorbs float noise in p·10)."""
    a = np.abs(np.asarray(values, dtype=float))
    a = a[np.isfinite(a)]
    return math.ceil(float(np.percentile(a, 99)) * 10.0 - 1e-9) / 10.0


def corner_refs(df: pd.DataFrame, min_n: int = MIN_N):
    """Per-(track, corner) reference from a frame with track, corner_id, resid columns.
    Returns ({(track, corner): {offset_mps, tol_mps, n}}, [(track, corner, n) dropped
    for n < min_n]). Non-finite residuals are ignored."""
    refs: dict = {}
    dropped: list = []
    for (tr, cid), sub in df.groupby(["track", "corner_id"], sort=True):
        x = sub["resid"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) < min_n:
            dropped.append((tr, cid, len(x)))
            continue
        med = float(np.median(x))
        refs[(tr, cid)] = {"offset_mps": round(med, 2), "tol_mps": p99_ceil(x - med),
                           "n": len(x)}
    return refs, dropped


def splice_accel_block(text: str, block: dict) -> str:
    """The track JSON text with its top-level `accel_speed_check` set to `block`: the
    existing value is replaced in place, or the key is appended as the last top-level
    member. Every other byte (escapes, ordering, spacing) is left untouched."""
    blk = json.dumps(block, indent=2, ensure_ascii=False).replace("\n", "\n  ")
    data = json.loads(text)
    if BLOCK_KEY in data:
        key = f'"{BLOCK_KEY}"'
        k = text.index(key)
        v0 = text.index(":", k + len(key)) + 1
        while text[v0] in " \t\r\n":
            v0 += 1
        _, v1 = json.JSONDecoder().raw_decode(text, v0)
        return text[:v0] + blk + text[v1:]
    end = text.rstrip().rfind("}")
    body = text[:end].rstrip()
    return f'{body},\n  "{BLOCK_KEY}": {blk}\n' + text[end:]


def section_resid(traj, a: float, b: float, corridor) -> tuple[str, float]:
    """(section status, RAW accel_speed_resid_mps) as section_timing records it."""
    st = section_timing(traj, a, b, corridor)          # no reference → nothing centred
    rec = next(c for c in st.checks if c["name"] == CHECK)
    return st.status, float(rec["raw"])


COLS = ["track", "session_id", "lap", "corner_id", "resid", "T", "ilat", "clean", "fit"]


def collect(ctx, laps) -> pd.DataFrame:
    """One row per (lap, corner) section of EVERY lap of each session in `laps` (GPS-only
    emulation): `fit` = the section can feed the session fit (lap not rescale_invalid,
    covered, finite features); `clean` = the lap is one of `laps` and the section is
    status ok (feeds the references)."""
    clean = {(sid, lap) for sid, lap, _ in laps}
    rows = []
    for sid in sorted({sid for sid, _ in clean}):
        s = vc._read_traj_samples(sessions_dir(ctx.track) / sid / "samples.parquet")
        if s is None:
            continue
        for lap in sorted(set(s["lap"])):
            g = vc._lap_frame(s, lap)
            if len(g) < 2:
                continue
            traj = estimate_trajectory(vc.gps_only_emulation(g), ctx.corridor, ctx.frame)
            for cid, (a, b) in ctx.bounds.items():
                ft = accel_section_features(traj, a, b)
                if ft is None:
                    continue
                status, r = section_resid(traj, a, b, ctx.corridor)
                fit = traj.status != "rescale_invalid" and bool(np.all(np.isfinite(ft)))
                rows.append({"track": ctx.track, "session_id": sid, "lap": int(lap), "corner_id": cid,
                             "resid": r, "T": ft[1], "ilat": ft[2],
                             "clean": (sid, int(lap)) in clean and status == "ok", "fit": fit})
    return pd.DataFrame(rows, columns=COLS)


def _offsets(df: pd.DataFrame, col: str) -> dict:
    sub = df[df["clean"] & np.isfinite(df[col])]
    return {k: float(v) for k, v in sub.groupby(["track", "corner_id"])[col].median().items()}


def joint_refs(df: pd.DataFrame, min_n: int = MIN_N):
    """The alternating fit (module docstring). df has COLS. Returns (refs as corner_refs
    returns them, dropped, {(track, session_id): AccelCalibration | None}, iterations)."""
    df = df.copy()
    off = _offsets(df, "resid")
    cal: dict = {}
    for it in range(1, MAX_ITERS + 1):
        o = np.array([off.get((t, c), np.nan) for t, c in zip(df["track"], df["corner_id"])])
        raw = {}
        for key, idx in df[df["fit"]].groupby(["track", "session_id"]).groups.items():
            d = df.loc[idx]
            raw[key] = fit_accel_params(d["resid"].to_numpy() - o[df.index.get_indexer(idx)],
                                        d["T"], d["ilat"])
        cal = {}
        for tr in {k[0] for k in raw}:
            fitted = [v for k, v in raw.items() if k[0] == tr and v is not None]
            mb = float(np.median([v.b_g for v in fitted])) if fitted else 0.0
            mp = float(np.median([v.psi_rad for v in fitted])) if fitted else 0.0
            for k, v in raw.items():
                if k[0] == tr:
                    cal[k] = None if v is None else AccelCalibration(v.b_g - mb, v.psi_rad - mp, v.n_sections)
        corr = np.array([accel_correction(cal[(t, s)], T, L) if cal.get((t, s)) is not None else np.nan
                         for t, s, T, L in zip(df["track"], df["session_id"], df["T"], df["ilat"])])
        df["corrected"] = df["resid"].to_numpy() - corr
        new = _offsets(df, "corrected")
        moved = max((abs(new[k] - off.get(k, np.inf)) for k in new), default=0.0)
        off = new
        if moved <= CONVERGE_MPS:
            break
    sub = df[df["clean"]].rename(columns={"resid": "raw", "corrected": "resid"})
    refs, dropped = corner_refs(sub, min_n)
    return refs, dropped, cal, it


def _with_block(track_def: dict, refs, track: str) -> dict:
    """A copy of the track definition whose accel_speed_check holds the refs just derived."""
    return {**track_def, BLOCK_KEY: {"corners": {cid: v for (t, cid), v in refs.items() if t == track}}}


def probe(ctxs, refs) -> None:
    """The confirmed fake on its REAL lap, judged as production would: the session's
    calibration fitted over all its real laps against the references just derived."""
    track, sid, lap, cid = PROBE
    sp = sessions_dir(track) / sid / "samples.parquet"
    if not sp.exists():
        print(f"probe {track} {sid} L{lap} {cid}: session not under data/ — skipped")
        return
    s = vc._read_traj_samples(sp)
    if s is None or len(vc._lap_frame(s, lap)) < 2:
        print(f"probe {track} {sid} L{lap} {cid}: lap missing — skipped")
        return
    ctx = ctxs[track]
    tdef = _with_block(ctx.track_def or {}, refs, track)
    trajs = {int(n): estimate_trajectory(vc._lap_frame(s, n), ctx.corridor, ctx.frame)
             for n in sorted(set(s["lap"])) if len(vc._lap_frame(s, n)) >= 2}
    cal = fit_accel_calibration(list(trajs.values()), tdef)
    a, b = ctx.bounds[cid]
    st = section_timing(trajs[lap], a, b, ctx.corridor, accel_ref=accel_ref_for(tdef, cid), accel_cal=cal)
    rec = next(c for c in st.checks if c["name"] == CHECK)
    print(f"probe {track} {sid} L{lap} {cid} (real lap, status={trajs[lap].status}, cal={cal}): "
          f"raw={rec['raw']} m/s, correction={rec['correction']}, value={rec['value']} vs "
          f"tol={rec['threshold']} → pass={rec['pass']}, tier={st.tier}, rank={st.rank_eligible}, "
          f"time={st.time_s:.3f}s")


def write_tracks(refs) -> None:
    today = datetime.now(UTC).date()
    for tr in TRACKS:
        corners = {cid: v for (t, cid), v in refs.items() if t == tr}
        block = {"provenance": f"{today}, per-corner median + p99 |centred| of session-calibrated "
                               f"{CHECK} on GPS-only emulation of clean calibration OBD laps "
                               f"(W3 amendment + per-session accelerometer calibration)",
                 "corners": corners}
        p = tracks_dir() / f"{tr}.json"
        raw = p.read_bytes().decode("utf-8")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")
        out = splice_accel_block(text, block)
        back = json.loads(out)
        if any(accel_ref_for(back, cid) is None for cid in corners):
            raise RuntimeError(f"{p}: spliced block does not read back — not written")
        p.write_bytes((out.replace("\n", "\r\n") if crlf else out).encode("utf-8"))
        print(f"wrote {len(corners)} corner references → {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="update tracks/{track}.json accel_speed_check")
    args = ap.parse_args(argv)
    sessions = pd.concat([vc.list_sessions(t) for t in TRACKS], ignore_index=True)
    split = vc.session_split(sessions)
    ctxs = {tr: vc.TrackCtx.load(tr) for tr in TRACKS}
    parts = []
    for tr, ctx in ctxs.items():
        cal = [s for (t, s), v in split.items() if t == tr and v == "cal"]
        parts.append(collect(ctx, list(vc.clean_laps(ctx, cal))))
    df = pd.concat(parts, ignore_index=True)
    clean = df[df["clean"]]
    n_fin = int(np.isfinite(clean["resid"].to_numpy(dtype=float)).sum())
    if not n_fin:
        print("No evaluable sections — STOP and discuss.")
        return 1
    print(f"clean sections (status ok, finite raw resid) = {n_fin}; fit sections = {int(df['fit'].sum())}")
    refs, dropped, cals, iters = joint_refs(df)
    n_none = sum(v is None for v in cals.values())
    print(f"alternation converged in {iters} iterations; {len(cals)} sessions, {n_none} too short to fit")
    for (tr, sid), v in sorted(cals.items()):
        if v is not None:
            print(f"  {tr:6s} {sid}: b={v.b_g:+.4f} g, psi={np.degrees(v.psi_rad):+.2f} deg, n={v.n_sections}")
    print(f"{'track':6s} {'corner':6s} {'n':>5s} {'offset':>8s} {'tol':>6s} {'tol_was':>8s}")
    for (tr, cid), v in refs.items():
        was = ctxs[tr].accel_ref(cid)
        print(f"{tr:6s} {cid:6s} {v['n']:5d} {v['offset_mps']:+8.2f} {v['tol_mps']:6.1f} "
              f"{(was[1] if was else float('nan')):8.1f}")
    for tr, cid, n in dropped:
        print(f"WARNING: {tr} {cid}: only {n} finite clean sections (< {MIN_N}) — no reference, "
              f"GPS-only laps cannot reach tier A there")
    probe(ctxs, refs)
    if args.write:
        write_tracks(refs)
    else:
        print("(dry run — pass --write to update tracks/{track}.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
