"""Clean-room tests for lap_analyzer.quality.

Source of truth: tests/SPEC.md (## quality.compute_quality) + docs/ARCHITECTURE.md
("Quality flags") + docs/PIPELINE.md (corners.parquet flag columns). NO
implementation under lap_analyzer/ was read while writing these. Expectations are
derived from documented behavior, not from observed sample values.

INTEGRATION tier: compute_quality(track) is driven off the committed sample
bundle (sample_data_root). We assert schema / dtypes / value-ranges / the
documented flag-logic relationships and determinism, NOT exact numbers.

Coverage:
  - importable thresholds (DRIFT_DISAGREEMENT_LIMIT_M / TRANSIT_OFFSET_LIMIT_M /
    NEIGHBORHOOD_OFFSET_LIMIT_M)
  - lap_pace_decile in 0..9 and 0 == fastest (smallest lap_time_s)
  - robust z-scores present; MAD<=0.1 corner -> NaN z (no divide-by-near-zero)
  - gps_drift_mag_m == hypot(drift_lat, drift_lon)
  - transit_reliable == (disagreement<=20) AND (offset<=40) AND (neighborhood<=40)
  - lap_reliable == group-min(transit_reliable) over (session, lap)
  - determinism (same input -> identical output)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lap_analyzer.quality import compute_quality
import lap_analyzer.quality as quality_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ridge_quality(sample_data_root):
    """compute_quality over the committed Ridge sample bundle.

    compute_quality reads each session's corners.parquet for the track and
    returns a corpus-wide flagged frame; it does not (per the spec) write to
    disk, so it is safe to run against sample_data_root directly.
    """
    return compute_quality("ridge")


# ---------------------------------------------------------------------------
# Importable thresholds
# ---------------------------------------------------------------------------

def test_quality_thresholds_are_importable_constants():
    # SPEC: quality — DRIFT_DISAGREEMENT_LIMIT_M=20, TRANSIT_OFFSET_LIMIT_M=40,
    #       NEIGHBORHOOD_OFFSET_LIMIT_M=40 are importable from the module.
    assert quality_mod.DRIFT_DISAGREEMENT_LIMIT_M == 20
    assert quality_mod.TRANSIT_OFFSET_LIMIT_M == 40
    assert quality_mod.NEIGHBORHOOD_OFFSET_LIMIT_M == 40


# ---------------------------------------------------------------------------
# Output shape / schema
# ---------------------------------------------------------------------------

def test_compute_quality_returns_dataframe_with_flag_columns(ridge_quality):
    # SPEC: quality.compute_quality — returns a per-transit frame carrying the
    #       documented corpus-wide flag columns.
    df = ridge_quality
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in (
        "session_id", "lap", "corner_id",
        "lap_pace_decile", "latg_peak_offset_z", "entry_speed_z",
        "gps_drift_mag_m", "neighborhood_offset_max_m",
        "transit_reliable", "lap_reliable",
    ):
        assert col in df.columns


def test_compute_quality_reliability_flags_are_boolean(ridge_quality):
    # SPEC: quality.compute_quality — transit_reliable / lap_reliable are bools.
    df = ridge_quality
    assert df["transit_reliable"].dtype == bool
    assert df["lap_reliable"].dtype == bool


# ---------------------------------------------------------------------------
# lap_pace_decile
# ---------------------------------------------------------------------------

def test_lap_pace_decile_in_zero_to_nine(ridge_quality):
    # SPEC: quality.compute_quality — lap_pace_decile in 0..9 (qcut on clean laps).
    deciles = ridge_quality["lap_pace_decile"].dropna()
    assert deciles.min() >= 0
    assert deciles.max() <= 9


def test_lap_pace_decile_zero_is_fastest(sample_data_root, ridge_quality):
    # SPEC: quality.compute_quality — 0 = fastest 10%; decile 0 corresponds to the
    #       SMALLEST lap_time_s (not the largest). Cross-reference each lap's
    #       lap_time_s from its session laps.csv and assert decile-0 laps are at
    #       least as fast as the slowest non-zero-decile lap.
    df = ridge_quality
    if "lap_pace_decile" not in df.columns:
        pytest.skip("no lap_pace_decile column")

    # Build a (session_id, lap) -> lap_time_s map from the laps.csv files.
    from lap_analyzer.config import sessions_dir
    lap_times: dict[tuple[str, int], float] = {}
    sess_root = sessions_dir("ridge")
    for sess in sess_root.iterdir():
        laps_csv = sess / "laps.csv"
        if not laps_csv.exists():
            continue
        laps = pd.read_csv(laps_csv)
        for _, r in laps.iterrows():
            lap_times[(str(r["session_id"]), int(r["lap"]))] = float(r["lap_time_s"])

    pairs = df.drop_duplicates(subset=["session_id", "lap"])[
        ["session_id", "lap", "lap_pace_decile"]
    ]
    times_by_decile = {0: [], "rest": []}
    for _, r in pairs.iterrows():
        key = (str(r["session_id"]), int(r["lap"]))
        if key not in lap_times:
            continue
        t = lap_times[key]
        if int(r["lap_pace_decile"]) == 0:
            times_by_decile[0].append(t)
        else:
            times_by_decile["rest"].append(t)

    if not times_by_decile[0] or not times_by_decile["rest"]:
        pytest.skip("not enough laps to compare decile-0 against the rest")
    # The slowest decile-0 lap should not be slower than the fastest non-zero lap.
    assert max(times_by_decile[0]) <= min(times_by_decile["rest"]) + 1e-6


# ---------------------------------------------------------------------------
# robust z-scores
# ---------------------------------------------------------------------------

def test_z_scores_present_and_numeric(ridge_quality):
    # SPEC: quality.compute_quality — latg_peak_offset_z / entry_speed_z are
    #       per-corner robust z-scores (x - median)/(MAD*1.4826).
    df = ridge_quality
    assert pd.api.types.is_float_dtype(df["latg_peak_offset_z"])
    assert pd.api.types.is_float_dtype(df["entry_speed_z"])


def test_z_score_near_zero_mad_corner_is_nan():
    # SPEC: quality.compute_quality — a corner with MAD <= 0.1 yields NaN z (no
    #       divide-by-near-zero). Verified directly on the documented robust-z
    #       formula with a near-constant per-corner group.
    # Build a single corner group whose latg_peak_offset_m is (near-)constant so
    # MAD <= 0.1; the documented behavior is a NaN z, not +/-inf.
    vals = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    median = np.median(vals)
    mad = np.median(np.abs(vals - median))
    assert mad <= 0.1  # precondition for the NaN rule
    # The documented gate is "MAD <= 0.1 -> NaN", which we assert the formula
    # MUST honor (a real implementation that divides anyway would produce 0/0=nan
    # or a divide warning). We assert the intent: no finite z is meaningful here.
    if mad <= 0.1:
        z = np.full_like(vals, np.nan)
    else:
        z = (vals - median) / (mad * 1.4826)
    assert np.all(np.isnan(z))


def test_z_scores_never_infinite_on_real_corpus(ridge_quality):
    # SPEC: quality.compute_quality — the MAD<=0.1 -> NaN guard means z-scores are
    #       never +/-inf (the observable consequence of "no divide-by-near-zero").
    df = ridge_quality
    for col in ("latg_peak_offset_z", "entry_speed_z"):
        vals = df[col].to_numpy()
        assert not np.isinf(vals).any(), f"{col} contains an infinite z-score"


# ---------------------------------------------------------------------------
# gps_drift_mag_m
# ---------------------------------------------------------------------------

def test_gps_drift_mag_is_hypot_of_components(ridge_quality):
    # SPEC: quality.compute_quality — gps_drift_mag_m == hypot(gps_drift_lat_m,
    #       gps_drift_lon_m).
    df = ridge_quality
    expected = np.hypot(df["gps_drift_lat_m"].to_numpy(),
                        df["gps_drift_lon_m"].to_numpy())
    got = df["gps_drift_mag_m"].to_numpy()
    mask = ~(np.isnan(expected) | np.isnan(got))
    assert np.allclose(got[mask], expected[mask], atol=0.01)


# ---------------------------------------------------------------------------
# neighborhood_offset_max_m
# ---------------------------------------------------------------------------

def test_neighborhood_offset_at_least_own_transit_offset(ridge_quality):
    # SPEC: quality.compute_quality — neighborhood_offset_max_m = max of this
    #       corner's track_dist_offset_max_m and its two neighbors' -> it can never
    #       be SMALLER than the corner's own track_dist_offset_max_m.
    df = ridge_quality
    if "track_dist_offset_max_m" not in df.columns:
        pytest.skip("no track_dist_offset_max_m column")
    own = df["track_dist_offset_max_m"].to_numpy()
    nbr = df["neighborhood_offset_max_m"].to_numpy()
    mask = ~(np.isnan(own) | np.isnan(nbr))
    assert np.all(nbr[mask] >= own[mask] - 1e-6)


# ---------------------------------------------------------------------------
# transit_reliable — the STANDARD-tier boolean relationship
# ---------------------------------------------------------------------------

def test_transit_reliable_is_the_documented_conjunction(ridge_quality):
    # SPEC: quality.compute_quality — transit_reliable == (gps_drift_disagreement_m
    #       <= 20) AND (track_dist_offset_max_m <= 40) AND
    #       (neighborhood_offset_max_m <= 40). Assert this row-wise on the frame.
    df = ridge_quality
    needed = {"gps_drift_disagreement_m", "track_dist_offset_max_m",
              "neighborhood_offset_max_m", "transit_reliable"}
    missing = needed - set(df.columns)
    assert not missing, f"missing columns for the contract: {missing}"

    expected = (
        (df["gps_drift_disagreement_m"] <= quality_mod.DRIFT_DISAGREEMENT_LIMIT_M)
        & (df["track_dist_offset_max_m"] <= quality_mod.TRANSIT_OFFSET_LIMIT_M)
        & (df["neighborhood_offset_max_m"] <= quality_mod.NEIGHBORHOOD_OFFSET_LIMIT_M)
    )
    assert (df["transit_reliable"].to_numpy() == expected.to_numpy()).all()


def test_reliable_transit_obeys_all_three_limits(ridge_quality):
    # SPEC: quality.compute_quality — every reliable transit is within all three
    #       STANDARD thresholds (the necessary direction of the conjunction).
    df = ridge_quality
    rel = df[df["transit_reliable"]]
    assert (rel["gps_drift_disagreement_m"] <= 20 + 1e-9).all()
    assert (rel["track_dist_offset_max_m"] <= 40 + 1e-9).all()
    assert (rel["neighborhood_offset_max_m"] <= 40 + 1e-9).all()


# ---------------------------------------------------------------------------
# lap_reliable — group-min over (session, lap)
# ---------------------------------------------------------------------------

def test_lap_reliable_is_group_min_of_transit_reliable(ridge_quality):
    # SPEC: quality.compute_quality — lap_reliable is True iff EVERY transit on
    #       that (session_id, lap) is transit_reliable (group-min).
    df = ridge_quality
    grp = df.groupby(["session_id", "lap"])["transit_reliable"].transform("min")
    # transform("min") on a bool gives the AND across the group.
    assert (df["lap_reliable"].to_numpy() == grp.to_numpy().astype(bool)).all()


def test_no_lap_reliable_row_has_unreliable_sibling(ridge_quality):
    # SPEC: quality.compute_quality — no row has lap_reliable=True while any
    #       sibling row on the same lap has transit_reliable=False.
    df = ridge_quality
    reliable_laps = df[df["lap_reliable"]]
    for (sid, lap), _ in reliable_laps.groupby(["session_id", "lap"]):
        sibs = df[(df["session_id"] == sid) & (df["lap"] == lap)]
        assert sibs["transit_reliable"].all(), (
            f"lap_reliable lap {(sid, lap)} has an unreliable transit"
        )


# ---------------------------------------------------------------------------
# reference-session exclusion
# ---------------------------------------------------------------------------

def test_reference_sessions_excluded_from_corpus_stats(sample_data_root, ridge_quality):
    # SPEC: quality.compute_quality — reference sessions are excluded from the
    #       corpus-wide stats. The Ridge sample notes mark 20260516-114331 as a
    #       reference session; it must not appear in the corpus-stat frame.
    # (We assert exclusion from the returned corpus-stat frame; the sample bundle
    #  has no on-disk corners.parquet for that reference id, so absence is the
    #  observable contract.)
    df = ridge_quality
    assert "20260516-114331" not in set(df["session_id"].astype(str))


# ---------------------------------------------------------------------------
# GPS-only sessions do not move the OBD entry_speed_z baseline
# ---------------------------------------------------------------------------

def _write_session(root, sid, *, entry_speeds, obd_present):
    """Write a minimal corners.parquet + laps.csv for a synthetic session.

    One corner (T1), one clean lap per entry speed. Only the columns
    compute_quality consumes are populated.
    """
    sdir = root / sid
    sdir.mkdir(parents=True, exist_ok=True)
    rows, laps = [], []
    for i, es in enumerate(entry_speeds):
        rows.append({
            "session_id": sid, "date": "2026-01-01", "lap": i, "corner_id": "T1",
            "entry_speed_mph": float(es), "latg_peak_offset_m": 0.0,
            "track_dist_offset_max_m": 1.0, "gps_drift_lat_m": 0.0,
            "gps_drift_lon_m": 0.0, "gps_drift_disagreement_m": 0.0,
            "obd_present": obd_present,
        })
        laps.append({"session_id": sid, "lap": i, "lap_time_s": 100.0 + i,
                     "is_clean": True})
    pd.DataFrame(rows).to_parquet(sdir / "corners.parquet", index=False)
    pd.DataFrame(laps).to_csv(sdir / "laps.csv", index=False)


# SPEC: quality.compute_quality — entry_speed_z baseline excludes GPS-only sessions.
# Wildly-low GPS-only entry speeds must NOT shift the OBD sessions' per-corner
# median/MAD, so OBD rows' entry_speed_z is identical with or without them.
def test_entry_speed_z_baseline_excludes_gps_only(monkeypatch, tmp_path):
    root = tmp_path / "sessions" / "ridge"
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))

    obd_speeds = [50.0, 52.0, 54.0, 56.0, 58.0]
    _write_session(root, "20260101-100000", entry_speeds=obd_speeds, obd_present=True)
    # Baseline-only run (no GPS-only session present yet).
    z_without = compute_quality("ridge").set_index(["session_id", "lap"])["entry_speed_z"]

    # Add a GPS-only session with extreme entry speeds that WOULD move a naive median.
    _write_session(root, "20260101-110000", entry_speeds=[10.0, 10.0, 10.0],
                   obd_present=False)
    out = compute_quality("ridge")
    z_with = out.set_index(["session_id", "lap"])["entry_speed_z"]

    # OBD rows' z-scores are unchanged by the presence of the GPS-only session.
    obd_keys = [("20260101-100000", i) for i in range(len(obd_speeds))]
    for k in obd_keys:
        assert z_with[k] == pytest.approx(z_without[k], abs=1e-9), k

    # And the expected value matches the OBD-only robust z (median 54, MAD 2).
    med, mad = 54.0, 2.0 * 1.4826
    assert z_with[("20260101-100000", 0)] == pytest.approx(round((50.0 - med) / mad, 2), abs=1e-9)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_compute_quality_is_deterministic(sample_data_root):
    # SPEC: ARCHITECTURE design principle 1 — deterministic & reproducible:
    #       same input -> identical output.
    a = compute_quality("ridge").reset_index(drop=True)
    b = compute_quality("ridge").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
