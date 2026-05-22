from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from ..config import corpus_dir, sessions_dir
from ..corners import extract_session_candidates


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detect corner candidates across all normalized sessions for a track.")
    p.add_argument("--track", required=True)
    p.add_argument("--out", help="Override corpus output path (default: data/corpus/<track>_candidates.csv)")
    args = p.parse_args(argv)

    session_root = sessions_dir(args.track)
    if not session_root.exists():
        print(f"No sessions directory at {session_root}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else corpus_dir() / f"{args.track}_candidates.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for sd in sorted(session_root.iterdir()):
        if not sd.is_dir() or not (sd / "samples.parquet").exists():
            continue
        df = extract_session_candidates(sd)
        df.to_csv(sd / "candidates.csv", index=False)
        all_dfs.append(df)
        print(f"  {sd.name}  candidates={len(df)}")

    if not all_dfs:
        print("No sessions found.", file=sys.stderr)
        return 1

    corpus = pd.concat(all_dfs, ignore_index=True)
    corpus.to_csv(out_path, index=False)
    print(f"\n{len(all_dfs)} sessions, {len(corpus)} candidates total")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
