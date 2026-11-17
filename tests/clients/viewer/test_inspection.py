"""Click an animal and see what it is (#195).

Test-first (§8.1): picking is arithmetic and the panel is a pure function of a report, so both
contracts were writable before the implementation. What the report *holds* is `Session.inspect`'s,
and is tested with the api.

Asserted on *what the panel says*, never on its exact layout — a test pinning the character
positions of a diagnostic readout is a maintenance cost wearing the costume of safety (§8.1).
"""

import numpy as np
import pytest

import api
from clients.viewer.render import describe_entity, pick_entity, screen_to_world

SCREEN = (900, 900)


def session():
    return api.create("demo", seed=3, founders=40)


def first_id(s):
    return int(s.frame().ids.to_numpy()[0])


def text(s, entity_id):
    return "\n".join(describe_entity(entity_id, s.inspect(entity_id)))


class TestScreenToWorld:
    def test_it_inverts_world_to_screen(self):
        from clients.viewer.render import world_to_screen

        x = np.array([0.0, 37.5, 75.0], dtype=np.float64)
        y = np.array([75.0, 10.0, 0.0], dtype=np.float64)
        px, py = world_to_screen(x, y, 75.0, 75.0, *SCREEN)

        back_x, back_y = screen_to_world(px, py, 75.0, 75.0, *SCREEN)

        assert back_x == pytest.approx(x, abs=0.2)
        assert back_y == pytest.approx(y, abs=0.2)

    def test_a_degenerate_world_maps_to_the_origin(self):
        """A zero-extent world is a config error rather than something to divide by (§8.7 prefers
        a raised error, but `world_to_screen` already returns zeros here and the two must agree)."""
        x, y = screen_to_world(np.array([10.0]), np.array([10.0]), 0.0, 0.0, *SCREEN)

        assert x == pytest.approx([0.0])
        assert y == pytest.approx([0.0])


class TestPicking:
    def positions(self):
        return (
            np.array([10.0, 20.0, 30.0]),
            np.array([10.0, 20.0, 30.0]),
            np.array([0, 1, 2], dtype=np.int64),
        )

    def test_it_finds_the_nearest_entity(self):
        x, y, rows = self.positions()

        assert pick_entity(19.0, 21.0, x, y, rows, radius=5.0) == 1

    def test_it_finds_nothing_on_empty_ground(self):
        """Clicking nowhere must clear the panel rather than leave the last animal selected."""
        x, y, rows = self.positions()

        assert pick_entity(100.0, 100.0, x, y, rows, radius=5.0) is None

    def test_the_radius_is_respected(self):
        x, y, rows = self.positions()

        assert pick_entity(14.0, 10.0, x, y, rows, radius=5.0) == 0
        assert pick_entity(16.0, 10.0, x, y, rows, radius=5.0) is None

    def test_an_empty_world_picks_nothing(self):
        empty = np.array([], dtype=np.float64)

        assert pick_entity(0.0, 0.0, empty, empty, np.array([], dtype=np.int64), 5.0) is None

    def test_it_returns_the_id_it_was_given_not_a_position_index(self):
        """The index into this frame's arrays means nothing once the next frame arrives in a
        different order. The id still names the same animal."""
        x = np.array([10.0, 20.0])
        y = np.array([10.0, 20.0])

        assert pick_entity(20.0, 20.0, x, y, np.array([7, 41], dtype=np.int64), 5.0) == 41


class TestThePanel:
    def test_it_reports_speed_against_top_speed_and_urge_against_pace(self):
        """The two readings momentum and haste are only visible through (#203, #204). A position
        cannot say whether an animal is being held below its own pace by agility or by an empty
        pool, and "why did this population crash" is what the panel exists to answer (§3.3).
        """
        s = session()
        s.advance(5)
        entity = first_id(s)

        line = next(
            line for line in describe_entity(entity, s.inspect(entity)) if line.startswith("speed ")
        )

        assert "top" in line
        assert "urge" in line and "pace" in line

    def test_it_names_the_entity_and_its_species(self):
        s = session()
        entity = first_id(s)

        readout = text(s, entity)

        assert "species" in readout.lower()
        assert str(entity) in readout
        assert str(s.inspect(entity).species) in readout

    def test_it_reports_energy_and_what_the_animal_costs_to_keep(self):
        """`Ecology.upkeep` is exposed separately from `drain` precisely so a viewer can ask what
        an animal costs without spending it."""
        s = session()

        readout = text(s, first_id(s)).lower()

        assert "energy" in readout
        assert "upkeep" in readout

    def test_it_shows_every_drive_with_its_share_of_the_decision(self):
        """"62% of that heading was hunger" is strictly more informative than a winner's name —
        `Behaviour.breakdown`'s own words, and the reason it returns contributions."""
        s = session()
        s.advance(2)
        entity = first_id(s)

        readout = text(s, entity).lower()

        for share in s.inspect(entity).drives:
            assert share.name in readout

    def test_it_shows_genes_with_both_the_stored_and_expressed_value(self):
        """The expression mode is invisible otherwise: a magnitude gene folding across zero and a
        unit-interval gene squashing are only legible when both numbers are on screen."""
        s = session()
        entity = first_id(s)
        lines = describe_entity(entity, s.inspect(entity))

        assert any("diet_animal_derived" in line for line in lines)
        assert any("size" in line for line in lines)
        gene_line = next(line for line in lines if "diet_animal_derived" in line)
        assert len(gene_line.split()) >= 3

    def test_a_gestating_animal_is_named_as_one(self):
        """A negative age is the gestation clock (#20), not a corrupt row, and a panel that showed
        "age -14" would read as a bug."""
        s = session()
        entity = first_id(s)
        row = int(np.flatnonzero(s._world.store.row_ids() == entity)[0])
        s._world.store.age[row] = -14

        readout = text(s, entity).lower()

        assert "gestating" in readout
        assert "14" in readout

    def test_a_born_animal_is_not_called_gestating(self):
        s = session()
        entity = first_id(s)
        row = int(np.flatnonzero(s._world.store.row_ids() == entity)[0])
        s._world.store.age[row] = 300

        readout = text(s, entity).lower()

        assert "gestating" not in readout
        assert "300" in readout

    def test_a_dead_animal_says_so_rather_than_describing_a_corpse(self):
        """`release` leaves a row's columns untouched (#119), so a freed row still reads like an
        animal. By id, the selected animal is gone, and the panel says that."""
        s = session()
        entity = first_id(s)
        s._world.store.release(np.array([entity], dtype=np.int64))

        readout = text(s, entity).lower()

        assert "gone" in readout
        assert "energy" not in readout

    def test_every_line_is_a_string(self):
        s = session()
        entity = first_id(s)

        assert all(isinstance(line, str) for line in describe_entity(entity, s.inspect(entity)))
