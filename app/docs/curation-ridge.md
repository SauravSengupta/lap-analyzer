# Ridge Motorsports Park — Track Map Curation Handoff

> **NOTE (2026-05-10):** This doc describes the **original curation methodology** — cluster the auto-detected lat-G peaks from `data/corpus/ridge_candidates.csv` and use the cluster medians to populate `tracks/ridge.json`. This is no longer the active workflow. The current architecture uses **Google Maps satellite-derived apex pins** as the geometric source of truth; `apex_m` is set to each pin's centerline position rather than to cluster medians. See [`/README.md`](../../README.md) for current architecture and the "How to add a new track" section there for the bootstrap recipe. This doc is kept as historical context for how `tracks/ridge.json` first got populated (Component 2.5 — `corners.py` — still exists and can still seed initial candidate clusters, but is now optional rather than load-bearing).

## Your job

Turn `data/corpus/ridge_candidates.csv` (auto-detected corner peaks) into `app/tracks/ridge.json` (curated corner definitions). The pipeline detected 6,080 candidate peaks across 285 flying laps from 47 sessions — most are real corners, some are multi-apex doubles, some are noise. Cluster them, identify Ridge's 16 corners, write the JSON.

## Inputs you'll get

1. **`data/corpus/ridge_candidates.csv`** — one row per detected peak per lap (the dataset to cluster)
2. **The official Ridge track map** — image showing 16 numbered corners, CCW direction
3. **Optionally**: a TrackAddict screenshot showing speed-vs-distance and Accel-Y-vs-distance for one lap, useful for visualizing where corners fall

## What you know about Ridge

- Counter-clockwise, 2.47 mi (3,975 m) real
- 16 numbered corners, T1 through T16
- The pipeline's lap-distance is OBD-integrated and reads ~3,962 m — essentially real-world accurate (within ~0.3%)
- Front straight is between T16 and T1; start/finish is on this straight
- Roughly **50/50 left/right turn split** in candidate detection — Ridge has lots of direction changes through chicanes and the inner loop. Don't assume "CCW = mostly lefts."
- Notable features per the user: a carousel/Ridge Complex around T6 (multi-apex), an inner loop (T6–T9 area, multiple direction changes), the slow back section around T13–T15 (slowest corner of the lap, 26-28 mph)

## CSV columns

| column | meaning |
|---|---|
| `session_id`, `date`, `lap`, `lap_time_s` | identity / filtering |
| `candidate_idx` | sequential within the lap |
| `peak_dist_m` | distance-from-lap-start at lat-G peak — **primary clustering key** |
| `peak_lat_g` | signed lat-G at peak. **Positive = right turn, negative = left** (automotive standard, already canonicalized) |
| `direction` | "right" / "left", matches sign of peak_lat_g |
| `entry_dist_m` / `exit_dist_m` | where lat-G crossed ±0.2g going in / out |
| `duration_s` | exit_idx t minus entry_idx t |
| `min_speed_mph` / `min_speed_dist_m` | speed minimum within the corner window — usually the actual apex |
| `brake_on_offset_m`, `throttle_lift_offset_m`, `throttle_return_offset_m` | input timing relative to entry_dist_m, null if didn't occur in window |
| `apex_lat`, `apex_long` | GPS coords at speed minimum, useful for geographic cluster checks |

## Output schema (`app/tracks/ridge.json`)

```json
{
  "track_id": "ridge",
  "name": "Ridge Motorsports Park",
  "configuration": "full",
  "official_length_m": 3975,
  "direction": "counter-clockwise",
  "start_finish": {"lat": 47.254699, "long": -123.192475},
  "lap_length_internal_m": 3962,
  "corners": [
    {
      "id": "T1",
      "name": null,
      "start_m": 350.0,
      "end_m": 540.0,
      "apex_m": 482.0,
      "type": "left",
      "notes": null
    },
    {
      "id": "T6",
      "name": "Carousel",
      "start_m": 1700.0,
      "end_m": 2050.0,
      "apex_m": 1800.0,
      "secondary_apex_m": 1980.0,
      "type": "left",
      "notes": "Multi-apex sweeper"
    }
  ]
}
```

- `start_m` / `end_m`: cluster median of `entry_dist_m` / `exit_dist_m`
- `apex_m`: cluster median of `min_speed_dist_m` (more reliable than peak_dist_m for true apex)
- `type`: "left" or "right" — should be the consistent mode of the cluster
- `secondary_apex_m`: only for multi-apex corners (T6 carousel and similar)
- Numbering: T1 is the first corner after start/finish, T2 is the next, etc., in order of `peak_dist_m`

## Quirks to watch for

1. **Multi-apex corners fire 2–3 candidates per lap.** The carousel and the inner loop generate adjacent peaks within 100–200m of each other on the same lap. Merge these into one corner with a `secondary_apex_m`. Don't number them as separate corners.

2. **Brief direction kinks within a larger corner show as opposite-direction candidates.** A long left sweeper can briefly trip a small right-G peak between two left apexes. If a cluster has mixed directions across laps with one direction dominating, treat it as a single corner of the dominant direction.

3. **Some real Ridge corners are below the 0.4g detection threshold and aren't in the CSV.** If the track map has a corner where the candidates have a gap, the corner is real but undetected — estimate `start_m` / `end_m` visually and put it in the JSON anyway.

4. **GPS glitches show as sparse single-lap candidates.** A cluster with only 1–3 candidates from 1–2 sessions, far from any other cluster, is almost certainly a glitch. Drop.

## Suggested workflow

1. Load the CSV, group by `peak_dist_m` with a 30–50m bucket to find initial clusters.
2. For each cluster summarize: number of laps it fires on, direction consistency, median duration, median min_speed.
3. Drop sparse clusters (<10 laps from <3 sessions).
4. Walk through the remaining clusters in distance order with the user, naming each as T1, T2, … T16. Use the track map to disambiguate: "this cluster at 480m, left, min speed 73 mph — that's T1 right?"
5. Identify multi-apex pairs (close peaks, same direction, same speed minimum) and merge.
6. Estimate `start_m` / `end_m` / `apex_m` from cluster medians.
7. For corners that are on the track map but not in the candidates (sub-threshold sweepers), get their approximate position from the user and add manually.
8. Write `app/tracks/ridge.json` and verify: 16 corners, monotonically increasing `start_m`, no overlaps, T1 starts ~300–500m into the lap, T16 ends well before 3962m so there's a front straight back to start.

## After this is done

The next pipeline step is `label_corners.py` (not yet built). It uses `tracks/ridge.json` to label each sample with a corner_id and produce the per-corner-per-lap stats CSV that drops into ChatGPT/Claude for analysis. The track JSON is what makes that work, so accuracy here matters — but every field can be edited later if something turns out wrong.
