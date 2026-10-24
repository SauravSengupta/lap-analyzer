"""Derive the per-corner accel-speed references by their rule (W3 amendment).

Rule (user decision 2026-10-24, replacing the pooled 2026-10-22 rule): over every
status=='ok' section of the GPS-only emulation (validation_checks.gps_only_emulation)
of clean CALIBRATION-split OBD laps, per (track, corner):
  offset_mps = median(resid)                         (rounded to 0.01 m/s)
  tol_mps    = p99 |resid − offset|, rounded UP to 0.1 m/s
  n          = number of finite residuals; corners with n < MIN_N get NO entry.
The residual is the RAW accel_speed_resid_mps read from section_timing(...).checks
(called WITHOUT a reference, so nothing is centred) — the production code path, so the
rule measures exactly what production checks. Prints the per-corner table; with
--write it updates tracks/{track}.json's `accel_speed_check` block in place (every
other byte of the file is left as it is). Local; needs data/.

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
from lap_analyzer.trajectory import accel_ref_for, estimate_trajectory, section_timing

TRACKS = ("pir", "ridge")
CHECK = "accel_speed_resid_mps"
BLOCK_KEY = "accel_speed_check"
MIN_N = 20                                    # 2026-10-24, fewer clean sections → no reference
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


def collect(ctx, laps) -> pd.DataFrame:
    rows = []
    for sid, lap, g in laps:
        traj = estimate_trajectory(vc.gps_only_emulation(g), ctx.corridor, ctx.frame)
        for cid, (a, b) in ctx.bounds.items():
            status, r = section_resid(traj, a, b, ctx.corridor)
            if status == "ok":
                rows.append({"track": ctx.track, "session_id": sid, "lap": lap, "corner_id": cid, "resid": r})
    return pd.DataFrame(rows, columns=["track", "session_id", "lap", "corner_id", "resid"])


def probe(ctxs, refs) -> None:
    """The confirmed fake on its REAL lap, judged against the reference just derived."""
    track, sid, lap, cid = PROBE
    sp = sessions_dir(track) / sid / "samples.parquet"
    if not sp.exists():
        print(f"probe {track} {sid} L{lap} {cid}: session not under data/ — skipped")
        return
    s = vc._read_traj_samples(sp)
    g = vc._lap_frame(s, lap) if s is not None else None
    if g is None or len(g) < 2:
        print(f"probe {track} {sid} L{lap} {cid}: lap missing — skipped")
        return
    ctx = ctxs[track]
    traj = estimate_trajectory(g, ctx.corridor, ctx.frame)   # the real lap, not emulated
    a, b = ctx.bounds[cid]
    ent = refs.get((track, cid))
    ref = (ent["offset_mps"], ent["tol_mps"]) if ent else None
    st = section_timing(traj, a, b, ctx.corridor, accel_ref=ref)
    rec = next(c for c in st.checks if c["name"] == CHECK)
    print(f"probe {track} {sid} L{lap} {cid} (real lap, status={traj.status}): raw={rec['raw']} m/s, "
          f"centred={rec['value']} vs tol={rec['threshold']} (ref {ref}) → pass={rec['pass']}, "
          f"tier={st.tier}, rank={st.rank_eligible}, time={st.time_s:.3f}s")


def write_tracks(refs) -> None:
    today = datetime.now(UTC).date()
    for tr in TRACKS:
        corners = {cid: v for (t, cid), v in refs.items() if t == tr}
        block = {"provenance": f"{today}, per-corner median + p99 |centred| of {CHECK} on GPS-only "
                               f"emulation of clean calibration OBD laps (W3 amendment)",
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
        parts.append(collect(ctx, vc.clean_laps(ctx, cal)))
    df = pd.concat(parts, ignore_index=True)
    n_fin = int(np.isfinite(df["resid"].to_numpy(dtype=float)).sum())
    if not n_fin:
        print("No evaluable sections — STOP and discuss.")
        return 1
    print(f"n sections (status ok, finite resid) = {n_fin}  (non-finite dropped: {len(df) - n_fin})")
    refs, dropped = corner_refs(df)
    print(f"{'track':6s} {'corner':6s} {'n':>5s} {'offset':>8s} {'tol':>6s}")
    for (tr, cid), v in refs.items():
        print(f"{tr:6s} {cid:6s} {v['n']:5d} {v['offset_mps']:+8.2f} {v['tol_mps']:6.1f}")
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
