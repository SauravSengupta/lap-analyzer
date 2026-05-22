from __future__ import annotations

import argparse
import sys

from ..corpus import build_corpus


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Concat per-session corners.parquet into one corpus parquet.")
    p.add_argument("--track", required=True)
    args = p.parse_args(argv)
    r = build_corpus(args.track)
    print(f"Wrote {r['out_path']}: {r['n_sessions']} sessions, {r['n_laps']} laps, {r['n_transits']} transits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
