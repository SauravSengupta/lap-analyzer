import os
from pathlib import Path

from dotenv import load_dotenv

APP_ROOT = Path(__file__).resolve().parents[1]  # lap_analyzer/ -> repo root
load_dotenv(APP_ROOT / ".env")


def data_root() -> Path:
    return (APP_ROOT / os.environ.get("DATA_ROOT", "data")).resolve()


def raw_dir(track: str) -> Path:
    return data_root() / "raw" / track


def sessions_dir(track: str) -> Path:
    return data_root() / "sessions" / track


def corpus_dir() -> Path:
    return data_root() / "corpus"


def tracks_dir() -> Path:
    return APP_ROOT / "tracks"
