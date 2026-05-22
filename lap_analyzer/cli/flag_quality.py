from __future__ import annotations

import argparse
import sys

from ..quality import write_quality


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compute corpus-wide quality flags and write back to each session's corners.parquet.")
    p.add_argument("--track", required=True)
    args = p.parse_args(argv)
    result = write_quality(args.track)
    print(f"Updated {result['sessions_updated']} sessions, {result['n_transits']} transit rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
