"""Writing a recorded history out, so it can be read somewhere that is not this window (#39).

The viewer is a diagnostic instrument (§3.3), and the questions it cannot answer are the ones that
want a spreadsheet or a notebook: how a trait's distribution moved over ten thousand ticks, how two
seeds differ, whether a run is worth keeping. Those are not viewer features; they are *this file*
plus whatever the reader already uses.

**I/O lives in a client, and only in a client.** `core/` may not perform it at all (§4), and
`metrics/` deliberately does not either — it produces serialisable values and stops, because on the
shared machine §3.1 is heading for, the process that records a series is not the process that
writes a file. Putting the write here keeps that split honest while the two still share an address
space.

Two formats, because they answer different questions and neither subsumes the other:

- **CSV** is what a spreadsheet and a notebook both open without being told anything. It is the
  format for "plot this", and it is flat, so the per-gene columns are widened out by name.
- **JSON** is the sample as recorded, nested and lossless. It is the format for "feed this back to
  something", and it is exactly what `Sample.as_dict()` returns — the same shape that would come
  off a socket, which is what makes it worth having beside the flat one.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence


def flatten(sample) -> dict:
    """One sample as a flat mapping, per-gene lists widened into named columns.

    `expressed_mean` is a list whose meaning is positional, and a positional column in a CSV is a
    column nobody can read six months later — worse, it silently means a different gene the first
    time the vocabulary is widened, which §2.3 makes an additive and expected event. Naming each
    one `speed_mean`, `speed_spread` is what makes the file survive that.
    """
    row = {}
    for name, value in sample.as_dict().items():
        if name == "gene_names":
            continue
        if name in ("expressed_mean", "expressed_spread"):
            suffix = name.removeprefix("expressed_")
            row.update(
                {f"{gene}_{suffix}": v for gene, v in zip(sample.gene_names, value, strict=True)}
            )
        else:
            row[name] = value
    return row


def write_csv(samples: Sequence, path: Path) -> Path:
    """Write `samples` oldest-first as CSV, and return the path written.

    Raises on an empty history rather than writing a headerless file, because a zero-byte export
    reads as "nothing happened" when what happened is that nobody had recorded anything yet (§8.7).
    """
    if not len(samples):
        raise ValueError(f"nothing to write to {path}: the history holds no samples yet")
    rows = [flatten(sample) for sample in samples]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(samples: Sequence, path: Path) -> Path:
    """Write `samples` oldest-first as JSON, nested exactly as recorded, and return the path."""
    if not len(samples):
        raise ValueError(f"nothing to write to {path}: the history holds no samples yet")
    path.write_text(
        json.dumps([sample.as_dict() for sample in samples], indent=2), encoding="utf-8"
    )
    return path


def export_history(history, directory: Path, stem: str) -> tuple[Path, Path]:
    """Write both formats under `directory`, named `<stem>.csv` and `<stem>.json`.

    Both rather than a choice, because the cost of writing the second is nothing against the cost
    of discovering during analysis that the run was exported in the wrong one — and a viewer
    keypress is a bad place to ask a question.

    `directory` is created if it does not exist: an export that fails because a folder is missing
    is an export the user has to do twice.
    """
    directory.mkdir(parents=True, exist_ok=True)
    return (
        write_csv(history.samples, directory / f"{stem}.csv"),
        write_json(history.samples, directory / f"{stem}.json"),
    )
