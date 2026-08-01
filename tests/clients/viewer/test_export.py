"""Writing a history out, so it can be read somewhere that is not the viewer window (#39)."""

import csv
import json

import pytest

from clients.viewer.demo_world import build_demo_world
from clients.viewer.export import export_history, flatten, write_csv, write_json


def history(ticks=60):
    world = build_demo_world(seed=1, n_entities=60)
    world.loop.advance(ticks)
    return world.loop.metrics


class TestFlatten:
    def test_per_gene_lists_become_named_columns(self):
        """A positional column is one nobody can read six months later — and worse, it silently
        means a *different* gene the first time the vocabulary is widened, which §2.3 makes an
        additive and expected event."""
        row = flatten(history().samples[-1])

        assert "speed_mean" in row and "speed_spread" in row
        assert "expressed_mean" not in row

    def test_the_gene_name_list_is_dropped_rather_than_repeated_on_every_row(self):
        """It is the same tuple in every sample, and a column repeating the whole vocabulary once
        per row is noise that makes the file harder to open, not easier to read."""
        assert "gene_names" not in flatten(history().samples[-1])

    def test_every_scalar_field_survives(self):
        row = flatten(history().samples[-1])

        for name in ("tick", "living", "gestating", "conceptions", "deaths", "median_energy"):
            assert name in row


class TestWriting:
    def test_csv_round_trips_through_a_reader(self, tmp_path):
        recorded = history()

        written = write_csv(recorded.samples, tmp_path / "run.csv")

        with written.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len(recorded.samples)
        assert [int(row["tick"]) for row in rows] == recorded.series("tick")

    def test_json_keeps_the_sample_shape_as_recorded(self):
        """The nested form is what would come off a socket (§3.1), which is what makes it worth
        having beside the flat one rather than instead of it."""
        recorded = history()

        restored = json.loads(json.dumps([s.as_dict() for s in recorded.samples]))

        assert restored[0]["gene_names"] == list(recorded.samples[0].gene_names)
        assert len(restored[-1]["expressed_mean"]) == len(recorded.samples[-1].gene_names)

    def test_both_files_are_written_and_the_directory_is_created(self, tmp_path):
        """An export that fails because a folder is missing is an export the user has to do twice."""
        target = tmp_path / "not" / "yet" / "there"

        csv_path, json_path = export_history(history(), target, "run")

        assert csv_path.exists() and json_path.exists()
        assert csv_path.name == "run.csv" and json_path.name == "run.json"

    @pytest.mark.parametrize("writer", [write_csv, write_json])
    def test_an_empty_history_is_refused_rather_than_written_as_an_empty_file(
        self, writer, tmp_path
    ):
        """A zero-byte export reads as "nothing happened", when what happened is that nobody had
        recorded anything yet (§8.7)."""
        with pytest.raises(ValueError, match="no samples yet"):
            writer([], tmp_path / "empty.csv")
