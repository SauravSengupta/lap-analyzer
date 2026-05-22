r"""Verification for gear derivation + T8 downshift classification.

Run: app\.venv\Scripts\python.exe app/notebooks/verify_gear_classification.py
The project has no pytest harness; this script is the test harness. Each
check_* function asserts on synthetic input; main() also prints a corpus
eyeball at the end (added in Task 5).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def check_find_gear_bands() -> None:
    from lap_analyzer.analysis import find_gear_bands

    rng = np.random.default_rng(0)
    ratios = np.concatenate([
        rng.normal(54.0, 1.0, 3000),
        rng.normal(99.0, 1.2, 3000),
        rng.normal(180.0, 2.0, 3000),
    ])
    bands = find_gear_bands(ratios)
    assert len(bands) == 3, f"expected 3 bands, got {bands}"
    assert abs(bands[0] - 54.0) < 2.0, bands
    assert abs(bands[1] - 99.0) < 2.0, bands
    assert abs(bands[2] - 180.0) < 3.0, bands
    print("check_find_gear_bands OK", bands)


def _synthetic_samples(ratios: list[float], speed: float = 80.0,
                       dt: float = 0.046) -> pd.DataFrame:
    """Samples df with constant speed; rpm chosen to hit each target ratio.

    ratios entries may be None to mark an OBD dropout (rpm = 0).
    """
    n = len(ratios)
    rpm = np.array([0.0 if r is None else r * speed for r in ratios])
    return pd.DataFrame({
        "t": np.arange(n) * dt,
        "rpm": rpm,
        "speed_mph": np.full(n, speed),
        "track_dist_m": np.arange(n) * 2.0,
        "dist_lap_m": np.arange(n) * 2.0,
    })


def check_derive_gear() -> None:
    from lap_analyzer.analysis import derive_gear

    bands = np.array([54.0, 99.0, 180.0])  # gear idx: 54->2, 99->1, 180->0

    # A 3-sample blip at ratio 99 inside a long ratio-54 stint is below dwell
    # and must be smoothed away -> gear stays 2 (tallest) throughout.
    blip = [54.0] * 100 + [99.0] * 3 + [54.0] * 97
    g = derive_gear(_synthetic_samples(blip), bands)
    confirmed = g.dropna()
    assert (confirmed == 2).all(), f"blip not smoothed: {sorted(confirmed.unique())}"

    # A real, sustained downshift 54 -> 180 must register: gear 2 then 0.
    shift = [54.0] * 100 + [180.0] * 100
    g2 = derive_gear(_synthetic_samples(shift), bands).to_numpy()
    assert g2[20] == 2, g2[20]
    assert g2[-1] == 0, g2[-1]

    # OBD dropout (rpm == 0) -> NaN gear.
    drop = [54.0] * 50 + [None] * 50 + [54.0] * 50
    g3 = derive_gear(_synthetic_samples(drop), bands).to_numpy()
    assert g3[49] == 2, g3[49]       # last sample before dropout still gear 2
    assert np.isnan(g3[75]), g3[75]  # mid-dropout is NaN
    assert g3[125] == 2, g3[125]     # first sample after dropout back to gear 2
    print("check_derive_gear OK")


def check_span_time() -> None:
    from lap_analyzer.analysis import span_time

    # track_dist advances 40 m/s; dist_lap_m mirrors it (OBD agrees with GPS).
    dist = np.arange(0.0, 3000.0, 2.0)
    s = pd.DataFrame({"t": dist / 40.0, "track_dist_m": dist, "dist_lap_m": dist})
    got = span_time(s, 1830.5, 2574.5)
    assert got is not None and abs(got - (744.0 / 40.0)) < 0.05, got

    # GPS glitch: dist_lap_m (OBD) says the car traveled far less than the span.
    s_glitch = s.copy()
    s_glitch["dist_lap_m"] = s_glitch["dist_lap_m"] * 0.5
    assert span_time(s_glitch, 1830.5, 2574.5) is None

    # Bound never reached -> None.
    assert span_time(s, 1830.5, 9999.0) is None
    print("check_span_time OK")


_TRACK_DEF_STUB = {
    "corners": [
        {"id": "T8", "start_m": 1830.5, "end_m": 2036.5, "apex_m": 1876.5},
        {"id": "T9", "start_m": 2093.5, "end_m": 2296.5, "apex_m": 2168.5},
        {"id": "T10", "start_m": 2340.5, "end_m": 2501.5, "apex_m": 2425.5},
        {"id": "T11", "start_m": 2574.5, "end_m": 2727.5, "apex_m": 2643.5},
    ]
}


def _samples_over_span(ratio_fn, lo: float = 1760.0, hi: float = 2620.0,
                       speed: float = 80.0, dt: float = 0.046) -> pd.DataFrame:
    """Samples spanning [lo, hi] in track_dist_m; ratio_fn(dist) -> ratio."""
    dist = np.arange(lo, hi, 2.0)
    ratios = np.array([ratio_fn(d) for d in dist])
    return pd.DataFrame({
        "t": np.arange(len(dist)) * dt,
        "rpm": ratios * speed,
        "speed_mph": np.full(len(dist), speed),
        "track_dist_m": dist,
        "dist_lap_m": dist,
    })


def check_classify_t8_section() -> None:
    from lap_analyzer.analysis import classify_t8_section

    bands = np.array([54.0, 99.0, 180.0])

    # Downshift inside the T8 entry zone [1780.5, 1876.5]: ratio 54 -> 180 at 1850.
    ds = _samples_over_span(lambda d: 180.0 if d >= 1850.0 else 54.0)
    r = classify_t8_section(ds, _TRACK_DEF_STUB, bands)
    assert r["label"] == "downshift", r

    # No downshift anywhere: constant ratio.
    nd = _samples_over_span(lambda d: 54.0)
    r = classify_t8_section(nd, _TRACK_DEF_STUB, bands)
    assert r["label"] == "no-downshift", r

    # Downshift outside the T8 zone but inside the window (~2400 m, traffic).
    tr = _samples_over_span(lambda d: 180.0 if d >= 2400.0 else 54.0)
    r = classify_t8_section(tr, _TRACK_DEF_STUB, bands)
    assert r["label"] == "excluded" and r["excluded_reason"] == "traffic", r

    # No OBD across the whole window -> excluded/no-obd.
    no_obd = _samples_over_span(lambda d: 54.0)
    no_obd["rpm"] = 0.0
    r = classify_t8_section(no_obd, _TRACK_DEF_STUB, bands)
    assert r["label"] == "excluded" and r["excluded_reason"] == "no-obd", r
    print("check_classify_t8_section OK")


def eyeball_corpus() -> None:
    """Print real gear bands + a ratio histogram + classification counts."""
    from lap_analyzer.analysis import (
        classify_t8_section, gear_bands, lap_index, load_corpus, load_samples,
    )
    from lap_analyzer.config import tracks_dir
    import json

    track = "ridge"
    bands = gear_bands(track)
    print("\n=== gear bands (ratio centers) ===")
    print(np.round(bands, 1))

    td = json.loads((tracks_dir() / f"{track}.json").read_text(encoding="utf-8"))

    laps = lap_index(load_corpus(track), track=track)
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    examples: dict[str, list] = {"downshift": [], "no-downshift": []}
    for r in laps.itertuples(index=False):
        try:
            s = load_samples(track, r.session_id, int(r.lap))
        except Exception:
            continue
        res = classify_t8_section(s, td, bands)
        counts[res["label"]] = counts.get(res["label"], 0) + 1
        if res["excluded_reason"]:
            reasons[res["excluded_reason"]] = reasons.get(res["excluded_reason"], 0) + 1
        if res["label"] in examples and len(examples[res["label"]]) < 5:
            examples[res["label"]].append(
                (r.session_id, int(r.lap), res["downshift_dist_m"], res["t8_apex_gear"])
            )

    print("\n=== classification counts ===")
    for k, v in sorted(counts.items()):
        print(f"  {k:14s} {v}")
    print("  excluded reasons:", reasons)
    print("\n=== example laps ===")
    for label, rows in examples.items():
        print(f"  {label}:")
        for sid, lap, loc, ag in rows:
            loc_s = f"{loc:.0f}m" if loc is not None else "—"
            print(f"    {sid} L{lap}  downshift@{loc_s}  t8_apex_gear={ag}")


def main() -> None:
    check_find_gear_bands()
    check_derive_gear()
    check_span_time()
    check_classify_t8_section()
    eyeball_corpus()


if __name__ == "__main__":
    main()
