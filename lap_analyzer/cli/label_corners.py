from __future__ import annotations

import argparse
import sys

from ..config import sessions_dir
from ..labeler import (
    build_reference_index,
    build_session_corners_file,
    label_session,
    label_session_samples,
    load_track,
)


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

    # Single-session: label + build corners in one call (self-loads the corridor,
    # which is corpus-wide, from the persisted/on-the-fly build).
    if args.session:
        sd = session_root / args.session
        if not (sd / "samples.parquet").exists():
            print(f"No samples.parquet under {sd}", file=sys.stderr)
            return 1
        n_samples, n_transits = label_session(sd, track, ref=ref)
        print(f"  {sd.name}  samples={n_samples}  transits={n_transits}")
        return 0

    # All-sessions rebuild is TWO PHASES so the corridor (built from labeled
    # samples) slots between them (design PR-4): (1) label every session's samples,
    # (2) build/load the corridor + frame ONCE, (3) build every session's corners on
    # the trajectory s_hat. Skip underscore dirs (e.g. _synthetic_centerline).
    targets = sorted(d for d in session_root.iterdir()
                     if d.is_dir() and not d.name.startswith("_")
                     and (d / "samples.parquet").exists())

    print(f"\nPhase 1/2: labeling samples in {len(targets)} sessions")
    labeled = []
    for sd in targets:
        try:
            n_samples = label_session_samples(sd, track, ref=ref)
        except Exception as e:
            print(f"  {sd.name}  LABEL FAILED: {e}", file=sys.stderr)
            continue
        labeled.append(sd)
        print(f"  {sd.name}  samples={n_samples}")

    # Corridor + frame, built once now that every session's samples carry track_dist_m.
    from ..analysis import _load_or_build_corridor, load_centerline
    from ..gates import TrackFrame
    corridor = _load_or_build_corridor(args.track)
    frame = TrackFrame.from_centerline(load_centerline(args.track))
    print(f"\nCorridor ready ({len(corridor.s_bin)} bins); "
          f"Phase 2/2: building corners on s_hat")

    total_transits = 0
    n_sessions = 0
    for sd in labeled:
        try:
            n_transits = build_session_corners_file(sd, track, corridor=corridor, frame=frame)
        except Exception as e:
            print(f"  {sd.name}  CORNERS FAILED: {e}", file=sys.stderr)
            continue
        n_sessions += 1
        total_transits += n_transits
        print(f"  {sd.name}  transits={n_transits}")

    print(f"\n{n_sessions} sessions, {total_transits} corner-transits total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
