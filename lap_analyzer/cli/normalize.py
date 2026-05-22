from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import raw_dir, sessions_dir
from ..normalize import MissingOBDError, load_session_notes, normalize_session, session_id_from_filename


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Normalize TrackAddict CSV(s) to parquet + lap CSV + meta JSON.")
    p.add_argument("csv", nargs="?", help="Path to a TrackAddict CSV. Omit with --all to batch.")
    p.add_argument("--track", required=True, help="Track slug (e.g. 'ridge')")
    p.add_argument("--all", action="store_true", help="Process every CSV in data/raw/<track>/")
    p.add_argument("--out", help="Override output directory")
    p.add_argument("--force", action="store_true", help="Re-normalize even if outputs exist")
    args = p.parse_args(argv)

    out = Path(args.out) if args.out else sessions_dir(args.track)

    if args.all:
        if args.csv:
            p.error("Pass either a CSV path or --all, not both")
        targets = sorted(raw_dir(args.track).glob("*.csv"))
    elif args.csv:
        targets = [Path(args.csv)]
    else:
        p.error("Specify a CSV path or --all")

    if not targets:
        print("No CSVs found.", file=sys.stderr)
        return 1

    notes = load_session_notes(args.track)
    skipped = noobd = excluded = failed = 0
    for csv in targets:
        try:
            sid = session_id_from_filename(csv)
        except ValueError as e:
            print(f"skip  {csv.name}: {e}")
            failed += 1
            continue
        if sid in notes and "exclude" in notes[sid]:
            print(f"excl  {sid} ({notes[sid]['exclude']})")
            # Remove any prior normalized output so corpus stays clean.
            session_dir = out / sid
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
            excluded += 1
            continue
        if (out / sid / "samples.parquet").exists() and not args.force:
            print(f"skip  {sid} (already normalized)")
            skipped += 1
            continue
        try:
            meta = normalize_session(csv, args.track, out)
        except MissingOBDError as e:
            print(f"noobd {sid}: {e}")
            noobd += 1
            continue
        except Exception as e:
            print(f"FAIL  {sid}: {type(e).__name__}: {e}")
            failed += 1
            continue
        best = f"{meta.best_lap_time_s:.3f}" if meta.best_lap_time_s else "—"
        iat = f"{meta.iat_first_f:.0f}F" if meta.iat_first_f else "—"
        print(f"ok    {sid}  laps={meta.n_laps} clean={meta.n_clean_laps} best={best} iat={iat}")

    processed = len(targets) - skipped - noobd - excluded - failed
    print(f"\n{processed} processed, {skipped} skipped, {excluded} excluded, {noobd} no-OBD, {failed} failed.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
