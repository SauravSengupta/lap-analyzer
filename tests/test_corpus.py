"""Clean-room tests for lap_analyzer.corpus.

Source of truth: tests/SPEC.md (## corpus.build_corpus) + docs/PIPELINE.md
(build_corpus CLI / corpus parquet). NO implementation under lap_analyzer/ was
read while writing these.

INTEGRATION tier. NOTE: build_corpus WRITES to corpus_dir(). Running it under the
committed sample bundle would overwrite the committed sample corpus, so these
tests:
  (a) point DATA_ROOT at a COPY of the relevant sessions/<track> tree in tmp_path
      before calling, and
  (b) for the no-input edge case, point DATA_ROOT at an empty tmp_path and assert
      FileNotFoundError.
We never write under data/samples/.

Coverage:
  - FileNotFoundError when no corners.parquet exists
  - return dict keys (out_path, n_sessions, n_transits, n_laps)
  - out_path file is actually written under the (tmp) corpus dir
  - n_laps == distinct (session_id, lap) pairs
  - reference sessions excluded from n_sessions
  - determinism
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from lap_analyzer.corpus import build_corpus
from lap_analyzer.config import sessions_dir, corpus_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_ROOT = REPO_ROOT / "data" / "samples"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def copied_ridge_root(tmp_path, monkeypatch):
    """Copy the committed Ridge sessions tree into a tmp DATA_ROOT.

    Returns the tmp data root. build_corpus will read sessions/ridge/*/corners.parquet
    and write data/corpus/ridge_corners.parquet under this tmp root, never touching
    data/samples/.
    """
    src = SAMPLES_ROOT / "sessions" / "ridge"
    dst = tmp_path / "sessions" / "ridge"
    shutil.copytree(src, dst)
    # Copy notes so reference/exclude annotations are honored.
    notes_src = SAMPLES_ROOT / "notes"
    if notes_src.exists():
        shutil.copytree(notes_src, tmp_path / "notes")
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Edge case: no input
# ---------------------------------------------------------------------------

def test_build_corpus_no_corners_raises_filenotfound(tmp_path, monkeypatch):
    # SPEC: corpus.build_corpus — raises FileNotFoundError when no corners.parquet
    #       exists under the track's sessions dir (fresh empty DATA_ROOT).
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        build_corpus("ridge")


# ---------------------------------------------------------------------------
# Return contract
# ---------------------------------------------------------------------------

def test_build_corpus_returns_documented_keys(copied_ridge_root):
    # SPEC: corpus.build_corpus — returns a dict with keys out_path, n_sessions,
    #       n_transits, n_laps.
    out = build_corpus("ridge")
    assert isinstance(out, dict)
    assert set(out.keys()) >= {"out_path", "n_sessions", "n_transits", "n_laps"}


def test_build_corpus_counts_are_positive(copied_ridge_root):
    # SPEC: corpus.build_corpus — concatenates real sessions; with 3 sample
    #       sessions the counts are positive (sanity).
    out = build_corpus("ridge")
    assert out["n_sessions"] >= 1
    assert out["n_transits"] >= 1
    assert out["n_laps"] >= 1


def test_build_corpus_writes_out_path(copied_ridge_root):
    # SPEC: corpus.build_corpus / PIPELINE — writes data/corpus/<track>_corners.parquet.
    out = build_corpus("ridge")
    out_path = Path(out["out_path"])
    assert out_path.exists()
    # It lives under the tmp corpus dir, not under data/samples/.
    assert corpus_dir() in out_path.parents or out_path.parent == corpus_dir()
    df = pd.read_parquet(out_path)
    assert len(df) == out["n_transits"]


# ---------------------------------------------------------------------------
# Counting invariants
# ---------------------------------------------------------------------------

def test_n_laps_is_distinct_session_lap_pairs(copied_ridge_root):
    # SPEC: corpus.build_corpus — n_laps = distinct (session_id, lap) pairs.
    out = build_corpus("ridge")
    df = pd.read_parquet(out["out_path"])
    distinct = df.drop_duplicates(subset=["session_id", "lap"]).shape[0]
    assert out["n_laps"] == distinct


def test_n_transits_matches_row_count(copied_ridge_root):
    # SPEC: corpus.build_corpus — n_transits is the concatenated row count (one
    #       row per clean lap x corner).
    out = build_corpus("ridge")
    df = pd.read_parquet(out["out_path"])
    assert out["n_transits"] == len(df)


def test_n_sessions_counts_nonreference_session_dirs(copied_ridge_root):
    # SPEC: corpus.build_corpus — reference sessions are excluded; n_sessions counts
    #       non-reference session dirs that have a corners.parquet.
    out = build_corpus("ridge")
    sess_root = sessions_dir("ridge")
    dirs_with_corners = [
        d for d in sess_root.iterdir()
        if d.is_dir()
        and not d.name.startswith("_")
        and (d / "corners.parquet").exists()
    ]
    # The reference session (20260516-114331) is not on disk in the bundle, so the
    # count of on-disk non-reference dirs with corners is the upper bound; it must
    # equal n_sessions (no reference dir present to subtract).
    assert out["n_sessions"] == len(dirs_with_corners)


def test_reference_session_excluded_from_corpus(copied_ridge_root):
    # SPEC: corpus.build_corpus — reference sessions (notes "reference": true) are
    #       excluded from the corpus. The Ridge notes mark 20260516-114331 reference.
    out = build_corpus("ridge")
    df = pd.read_parquet(out["out_path"])
    assert "20260516-114331" not in set(df["session_id"].astype(str))


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_build_corpus_is_deterministic(copied_ridge_root):
    # SPEC: ARCHITECTURE design principle 1 — same input -> identical output.
    out1 = build_corpus("ridge")
    df1 = pd.read_parquet(out1["out_path"]).reset_index(drop=True)
    out2 = build_corpus("ridge")
    df2 = pd.read_parquet(out2["out_path"]).reset_index(drop=True)
    assert out1["n_transits"] == out2["n_transits"]
    assert out1["n_laps"] == out2["n_laps"]
    assert out1["n_sessions"] == out2["n_sessions"]
    pd.testing.assert_frame_equal(df1, df2)
