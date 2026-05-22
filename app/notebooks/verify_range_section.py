r"""Verification for the composite corner-range section functions.

Run: app\.venv\Scripts\python.exe app/notebooks/verify_range_section.py
The project has no pytest harness; this script is the test harness.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_STUB = {
    "lap_length_internal_m": 3800.0,
    "corners": [
        {"id": "T8", "start_m": 1830.5, "end_m": 2036.5, "apex_m": 1876.5},
        {"id": "T9", "start_m": 2093.5, "end_m": 2296.5, "apex_m": 2168.5},
        {"id": "T10", "start_m": 2340.5, "end_m": 2501.5, "apex_m": 2425.5},
        {"id": "T11", "start_m": 2574.5, "end_m": 2727.5, "apex_m": 2643.5},
    ],
}


def check_section_range_bounds() -> None:
    from lap_analyzer.analysis import section_bounds, section_range_bounds

    # from == to must equal the single-corner section_bounds entry.
    sb = section_bounds(_STUB)
    assert section_range_bounds(_STUB, "T9", "T9") == sb["T9"], (
        section_range_bounds(_STUB, "T9", "T9"), sb["T9"])

    # A range T8..T10: a = T8.start-50; b = min(T11.start, T10.end+250).
    a, b = section_range_bounds(_STUB, "T8", "T10")
    assert a == 1830.5 - 50.0, a
    assert b == min(2574.5, 2501.5 + 250.0), b

    # Last corner (no next corner): b capped by lap length.
    a, b = section_range_bounds(_STUB, "T11", "T11")
    assert b == min(3800.0, 2727.5 + 250.0), b

    # to before from is rejected.
    try:
        section_range_bounds(_STUB, "T10", "T8")
        raise AssertionError("expected ValueError for reversed range")
    except ValueError:
        pass
    print("check_section_range_bounds OK")


def check_range_section_times() -> None:
    """Smoke test against the real Ridge corpus (IO function — no synthetic input)."""
    import json
    from lap_analyzer.analysis import range_section_times
    from lap_analyzer.config import tracks_dir

    td = json.loads((tracks_dir() / "ridge.json").read_text(encoding="utf-8"))
    df = range_section_times("ridge", td, "T8", "T10")
    assert list(df.columns) == ["session_id", "lap", "section_time_s",
                                "obd_discrepancy_m"], df.columns
    assert len(df) > 50, f"expected many timed laps, got {len(df)}"
    assert (df["section_time_s"] > 0).all(), "section times must be positive"
    assert df["section_time_s"].between(5, 60).mean() > 0.9, (
        "T8-T10 section times should mostly be 5-60s; got "
        f"{df['section_time_s'].describe()}")
    # from == to must agree with the per-corner section_times for that corner.
    from lap_analyzer.analysis import section_times
    one = range_section_times("ridge", td, "T9", "T9").set_index(["session_id", "lap"])
    per = section_times("ridge", td)
    per_t9 = per[per["corner_id"] == "T9"].set_index(["session_id", "lap"])
    common = one.index.intersection(per_t9.index)
    assert len(common) > 50, f"too few overlapping laps: {len(common)}"
    diff = (one.loc[common, "section_time_s"] - per_t9.loc[common, "section_time_s"]).abs()
    assert diff.max() < 0.01, f"from==to disagrees with section_times: max diff {diff.max()}"
    print("check_range_section_times OK")


def main() -> None:
    check_section_range_bounds()
    check_range_section_times()


if __name__ == "__main__":
    main()
