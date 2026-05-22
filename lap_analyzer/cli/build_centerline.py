from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..centerline import build_centerline
from ..config import corpus_dir, data_root, sessions_dir, tracks_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Build synthetic centerline polyline from drift-corrected clean laps.")
    p.add_argument("--track", required=True, help="Track slug (e.g. 'ridge')")
    p.add_argument("--out", help="Override output parquet path")
    args = p.parse_args(argv)

    track_json = tracks_dir() / f"{args.track}.json"
    if not track_json.exists():
        print(f"No track definition found: {track_json}", file=sys.stderr)
        return 1
    notes = data_root() / "notes" / f"{args.track}.json"
    if not notes.exists():
        print(f"No notes file found: {notes}", file=sys.stderr)
        return 1
    track_def = json.loads(track_json.read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else corpus_dir() / f"{args.track}_centerline.parquet"

    # Corner [start_m, end_m] spans are "protected" from spread/fold rejection so
    # real corner line-variation is preserved; only straights get cleaned.
    protected = [(float(c["start_m"]), float(c["end_m"])) for c in track_def.get("corners", [])]

    build_centerline(
        sessions_dir=sessions_dir(args.track),
        notes_path=notes,
        lap_length_m=float(track_def["lap_length_internal_m"]),
        track_lat_deg=float(track_def["start_finish"]["lat"]),
        out_path=out,
        protected_ranges=protected,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
