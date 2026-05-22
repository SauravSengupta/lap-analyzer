from __future__ import annotations

import argparse
import sys

from ..centerline import install_centerline
from ..config import corpus_dir, sessions_dir, tracks_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Install a built centerline as the labeler's _synthetic_centerline reference session.")
    p.add_argument("--track", required=True, help="Track slug (e.g. 'ridge')")
    args = p.parse_args(argv)

    track_json = tracks_dir() / f"{args.track}.json"
    if not track_json.exists():
        print(f"No track definition found: {track_json}", file=sys.stderr)
        return 1
    centerline_path = corpus_dir() / f"{args.track}_centerline.parquet"
    if not centerline_path.exists():
        print(f"No centerline found: {centerline_path}", file=sys.stderr)
        print(f"  Run: python -m lap_analyzer.cli.build_centerline --track {args.track}", file=sys.stderr)
        return 1
    synth_dir = sessions_dir(args.track) / "_synthetic_centerline"

    install_centerline(track=args.track, centerline_path=centerline_path, synth_dir=synth_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
