"""What crosses the boundary is a value (#35): proven by walking every return, not by assertion.

The walk is the "every returned type is proven serialisable" half of #35's done-when. It does not
encode anything, because no transport exists yet (gate 1 on #35 is open) and a codec with no
caller is §8.2's speculative generality. What it pins is the property any codec would need: every
leaf is a JSON scalar or a `Column`'s bytes, and nothing is a view, a core object or a callable.
"""

import dataclasses
import json

import numpy as np
import pytest

import api
from api import Column, CullRequest, FenceRequest, SpeciesNameRequest
from tests.api.test_session import session_on

_SCALARS = (type(None), bool, int, float, str)


def assert_plain(value, path="value"):
    """Fail on anything a socket could not carry as JSON plus binary columns."""
    if isinstance(value, _SCALARS):
        return
    if isinstance(value, Column):
        assert type(value.data) is bytes, f"{path}.data is {type(value.data).__name__}, not bytes"
        assert isinstance(value.dtype, str), f"{path}.dtype is not a string"
        assert all(type(n) is int for n in value.shape), f"{path}.shape holds a non-int"
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_plain(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str), f"{path} has a non-string key {key!r}"
            assert_plain(item, f"{path}[{key!r}]")
        return
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        assert value.__dataclass_params__.frozen, f"{path} is a mutable {type(value).__name__}"
        for field in dataclasses.fields(value):
            assert_plain(getattr(value, field.name), f"{path}.{field.name}")
        return
    raise AssertionError(f"{path} is a {type(value).__module__}.{type(value).__name__}")


def json_round_trip(value):
    """JSON with columns as base64 text: the cheapest transport there could be, so a value that
    survives it survives any of them."""
    import base64

    def default(item):
        if isinstance(item, bytes):
            return base64.b64encode(item).decode()
        if dataclasses.is_dataclass(item):
            return {f.name: getattr(item, f.name) for f in dataclasses.fields(item)}
        raise TypeError(type(item).__name__)

    return json.loads(json.dumps(value, default=default))


class TestColumn:
    @pytest.mark.parametrize(
        "array",
        [
            np.arange(5, dtype=np.float32),
            np.arange(6, dtype=np.int64).reshape(2, 3),
            np.array([True, False, True]),
            np.array([1.5, -2.25], dtype=">f8"),
        ],
    )
    def test_it_reads_back_the_array_it_was_given(self, array):
        assert np.array_equal(Column.of(array).to_numpy(), array)

    def test_it_is_always_little_endian_so_any_reader_decodes_the_same_numbers(self):
        column = Column.of(np.array([1.0, 2.0], dtype=">f4"))

        assert column.dtype == "<f4"

    def test_it_is_a_copy_so_the_world_writing_next_tick_cannot_reach_it(self):
        """The rule the type exists for: a view would let a client hold a frame that changes
        under it, and a socket could never carry it anyway."""
        source = np.zeros(3, dtype=np.float32)
        column = Column.of(source)

        source[:] = 7.0

        assert np.array_equal(column.to_numpy(), np.zeros(3))

    def test_a_client_cannot_write_into_what_it_was_handed(self):
        with pytest.raises(ValueError):
            Column.of(np.zeros(3, dtype=np.float32)).to_numpy()[0] = 1.0

    def test_a_non_contiguous_slice_is_packed_rather_than_misread(self):
        grid = np.arange(12, dtype=np.float32).reshape(3, 4)

        assert np.array_equal(Column.of(grid[:, 1]).to_numpy(), grid[:, 1])


@pytest.fixture(scope="module")
def session():
    session = session_on(founders=20, seed=2)
    session.advance(25)
    return session


class TestEverythingReturnedIsPlain:
    """Every public read, walked, so a new method returning something live fails by name."""

    def calls(self, session):
        info = session.describe()
        some_id = int(session.frame().ids.to_numpy()[0])
        playing = api.create("demo", seed=2, founders=20, scenario="first-split")
        playing.advance(25)
        return {
            "worlds": api.worlds(),
            "scenarios": api.scenarios(),
            "status": session.status(),
            "describe": info,
            "terrain": session.terrain(),
            "frame": session.frame(),
            "barriers": session.barriers(),
            "history": session.history(),
            "inspect": session.inspect(some_id),
            "inspect_missing": session.inspect(-1),
            "ledger": session.ledger(),
            "species": session.species(),
            "scenario_sandbox": session.scenario(),
            "scenario": playing.scenario(),
            "name": session.name(SpeciesNameRequest(species_id=0, name="Grazers")),
            "quote_cull": session.quote(CullRequest(species_id=0, count=1)),
            "quote_fence": session.quote(FenceRequest(1.0, 1.0, 9.0, 9.0)),
            **{f"condition_{mode}": session.condition(mode) for mode in info.conditions},
            **{f"field_{layer}": session.field(layer) for layer in info.field_layers},
        }

    def test_every_public_read_returns_only_plain_values(self, session):
        for name, value in self.calls(session).items():
            assert_plain(value, name)

    def test_every_public_read_survives_json(self, session):
        for name, value in self.calls(session).items():
            json_round_trip(value)

    def test_the_walk_itself_rejects_a_live_array(self):
        """Otherwise the two tests above could pass by checking nothing."""
        with pytest.raises(AssertionError, match="numpy"):
            assert_plain({"x": np.zeros(3)})

    def test_every_public_method_is_exercised_above(self, session):
        """A method added to `Session` without being walked is a method whose return nobody
        checked. Control and write calls are exercised by the session tests instead."""
        walked = {"status", "describe", "terrain", "frame", "barriers", "history", "inspect",
                  "ledger", "species", "scenario", "name", "quote", "condition", "field"}
        control_and_write = {"advance", "save", "load", "request"}
        public = {name for name in vars(api.Session) if not name.startswith("_")}

        assert public == walked | control_and_write
