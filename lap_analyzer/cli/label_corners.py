from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import sessions_dir
from ..labeler import build_reference_index, label_session, load_track


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Label samples with corner_id and produce per-corner-transit tables.")
    p.add_argument("--track", required=True, help="Track id (e.g. 'ridge'); reads tracks/<id>.json")
    p.add_argument("--session", help="Process only this session id; default = all sessions for the track")
    args = p.parse_args(argv)

    track = load_track(args.track)
    print(f"Loaded {track.name}: {len(track.corners)} corners")

    session_root = sessions_dir(args.track)
    if not session_root.exists():
        print(f"No sessions directory at {session_root}", file=sys.stderr)
        return 1

    ref = build_reference_index(track)
    print(f"Reference lap: {track.reference_session_id} L{track.reference_lap}  ({len(ref.ref_dist)} samples)")

    if args.session:
        targets = [session_root / args.session]
    else:
        # Skip dirs starting with underscore (e.g. _synthetic_centerline) — they
        # are reference fixtures, not recordings to be labeled.
        targets = sorted(d for d in session_root.iterdir() if d.is_dir() and not d.name.startswith("_"))

    total_transits = 0
    n_sessions = 0
    for sd in targets:
        if not (sd / "samples.parquet").exists():
            continue
        try:
            n_samples, n_transits = label_session(sd, track, ref=ref)
        except Exception as e:
            print(f"  {sd.name}  FAILED: {e}", file=sys.stderr)
            continue
        n_sessions += 1
        total_transits += n_transits
        print(f"  {sd.name}  samples={n_samples}  transits={n_transits}")

    print(f"\n{n_sessions} sessions, {total_transits} corner-transits total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
