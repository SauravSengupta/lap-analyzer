"""Clean-room tests for lap_analyzer.config.

Source of truth: tests/SPEC.md "## config". Behavior is derived from the spec
only — no inspection of lap_analyzer source.
"""
from __future__ import annotations

from pathlib import Path


from lap_analyzer.config import (
    corpus_dir,
    data_root,
    raw_dir,
    sessions_dir,
    tracks_dir,
)


# SPEC: config — all path helpers return pathlib.Path
def test_helpers_return_path(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert isinstance(data_root(), Path)
    assert isinstance(raw_dir("ridge"), Path)
    assert isinstance(sessions_dir("ridge"), Path)
    assert isinstance(corpus_dir(), Path)
    assert isinstance(tracks_dir(), Path)


# SPEC: config — all returned paths are absolute
def test_all_paths_absolute(monkeypatch):
    # default DATA_ROOT (relative "data") must still resolve to an absolute path
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert data_root().is_absolute()
    assert raw_dir("ridge").is_absolute()
    assert sessions_dir("ridge").is_absolute()
    assert corpus_dir().is_absolute()
    assert tracks_dir().is_absolute()


# SPEC: config — default DATA_ROOT is "data" relative to repo root
def test_default_data_root_is_data(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    assert data_root().name == "data"


# SPEC: config — absolute DATA_ROOT is honored as-is (wins over repo-root join)
def test_absolute_data_root_honored(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    dr = data_root()
    assert dr.is_absolute()
    # honored as-is: the result is the given absolute path, not joined under the repo
    assert dr.resolve() == tmp_path.resolve()


# SPEC: config — data_root() reads the env var on every call (no caching)
def test_data_root_reads_env_each_call(monkeypatch, tmp_path):
    first = tmp_path / "alpha"
    second = tmp_path / "beta"
    monkeypatch.setenv("DATA_ROOT", str(first))
    assert data_root() == first
    monkeypatch.setenv("DATA_ROOT", str(second))
    assert data_root() == second  # changed without reload -> no caching


# SPEC: config — raw_dir(t) = data_root()/raw/<t>
def test_raw_dir_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert raw_dir("ridge") == data_root() / "raw" / "ridge"


# SPEC: config — sessions_dir(t) = data_root()/sessions/<t>
def test_sessions_dir_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert sessions_dir("pir") == data_root() / "sessions" / "pir"


# SPEC: config — corpus_dir() = data_root()/corpus
def test_corpus_dir_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert corpus_dir() == data_root() / "corpus"


# SPEC: config — track slug is interpolated into raw_dir/sessions_dir
def test_track_slug_interpolated(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert raw_dir("ridge").name == "ridge"
    assert sessions_dir("ridge").name == "ridge"
    assert raw_dir("pir").name == "pir"


# SPEC: config — tracks_dir() is independent of DATA_ROOT (always repo-root /tracks)
def test_tracks_dir_independent_of_data_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    under_tmp = tracks_dir()
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "elsewhere"))
    under_other = tracks_dir()
    assert under_tmp == under_other  # DATA_ROOT change does not move tracks_dir
    assert under_tmp.name == "tracks"
    # tracks_dir is not under either DATA_ROOT
    assert tmp_path not in under_tmp.parents


# SPEC: config — data_root() resolves DATA_ROOT relative to repo root when relative
def test_relative_data_root_joined_to_repo_root(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "data")
    dr = data_root()
    assert dr.is_absolute()
    assert dr.name == "data"
    # the repo root contains a tracks/ sibling
    assert (dr.parent / "tracks") == tracks_dir()
