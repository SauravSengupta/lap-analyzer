"""Harness sanity check — not a clean-room test; verifies fixtures wire up."""
import importlib

import pandas as pd


def test_package_imports():
    assert importlib.import_module("lap_analyzer") is not None


def test_sample_bundle_present(sample_data_root):
    corpus = sample_data_root / "corpus" / "ridge_corners.parquet"
    assert corpus.exists(), "committed sample corpus should exist"
    df = pd.read_parquet(corpus)
    assert len(df) > 0


def test_data_root_honors_env(sample_data_root):
    from lap_analyzer.config import data_root
    assert data_root() == sample_data_root
