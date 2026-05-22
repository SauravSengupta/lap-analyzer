"""Component 4 — corpus builder.

Concatenates every session's `corners.parquet` for a track into a single
`data/corpus/<track>_corners.parquet` for sub-second cross-session queries.
"""
from __future__ import annotations

import pandas as pd

from .config import corpus_dir, sessions_dir
from .normalize import reference_session_ids


def build_corpus(track: str) -> dict:
    """Concat all per-session corners.parquet into one corpus parquet. Returns counts dict.

    Reference sessions (see data/notes/<track>.json) are excluded — they stay on
    disk for comparison but never enter the corpus."""
    root = sessions_dir(track)
    ref = reference_session_ids(track)
    paths = [p for p in sorted(root.rglob("corners.parquet"))
             if p.parent.name not in ref]
    if not paths:
        raise FileNotFoundError(f"no corners.parquet under {root}")
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    out_dir = corpus_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{track}_corners.parquet"
    df.to_parquet(out_path, index=False)

    return {
        "out_path": str(out_path),
        "n_sessions": len(paths),
        "n_transits": len(df),
        "n_laps": df[["session_id", "lap"]].drop_duplicates().shape[0],
    }
