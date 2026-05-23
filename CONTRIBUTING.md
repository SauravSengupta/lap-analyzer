# Contributing

This is a personal project, published in the hope it's useful to other
track-day drivers. PRs are welcome but not actively solicited — I built it for my
own data and my own car.

## Adding your own track

The most likely reason you're here. Follow [docs/NEW-TRACK.md](docs/NEW-TRACK.md)
— it walks through the full bootstrap, including the lessons learned adding a
second track (Portland International Raceway) to a codebase that had been
Ridge-only. If you hit a snag, open an issue with:

- the raw TrackAddict CSV header (the `#` comment block — scrub anything you
  don't want public),
- the apex-pin JSON you produced,
- and the exact command + error.

## Heads-up on assumptions

The pipeline was built around one driver, one car (a Porsche 981 Cayman), and
one logging setup (TrackAddict + OBD + external GPS). A few things are tuned to
that and may need adjusting for your data — they're documented where they live:

- The **accelerometer axis mapping** (this phone logs lateral on `Accel Y`, not
  `Accel X`) — see [docs/PIPELINE.md](docs/PIPELINE.md).
- **OBD speed is canonical** and uncorrected for tire size — see
  [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
- TrackAddict CSV is the only supported input format.

## Tests

The suite lives in `tests/` and runs with pytest:

```
pip install -e ".[dev]"   # pytest + ruff
python -m pytest
```

Tests assert **documented design intent** — the contracts and invariants in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/PIPELINE.md](docs/PIPELINE.md) — rather than snapshotting whatever the code
currently returns. [`tests/SPEC.md`](tests/SPEC.md) is the behavioral spec the
suite was derived from; if you change a contract, update the spec and the test
together. Pure functions are unit-tested on synthetic inputs; IO-heavy stages
(labeler, quality, corpus, centerline) are exercised against the committed
`data/samples/` bundle. Known data limitations (e.g. GPS-compressed section
times) are marked `xfail`, not asserted as correct.

## Conventions

- Python 3.11+, formatted with `ruff` (see `pyproject.toml`).
- Run the pipeline from the repo root with `PYTHONPATH=.` (see
  [docs/PIPELINE.md](docs/PIPELINE.md)).
- No auto-generated coaching prose — the pipeline emits metrics and structured
  data; interpretation stays with the human.
