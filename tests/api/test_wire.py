"""The wire codec (#36): every value a session returns survives the trip, and nothing else gets in."""

import json

import numpy as np
import pytest

import api
from api import Column, CullRequest, FenceRequest, wire
from tests.api.test_session import session_on


@pytest.fixture(scope="module")
def session():
    session = session_on(founders=20, seed=2)
    session.advance(25)
    return session


def every_read(session):
    info = session.describe()
    some_id = int(session.frame().ids.to_numpy()[0])
    return {
        "status": session.status(),
        "describe": info,
        "terrain": session.terrain(),
        "frame": session.frame(),
        "barriers": session.barriers(),
        "history": session.history(),
        "inspect": session.inspect(some_id),
        "inspect_missing": session.inspect(-1),
        "ledger": session.ledger(),
        "quote": session.quote(CullRequest(species_id=0, count=1)),
        **{f"condition_{m}": session.condition(m) for m in info.conditions},
        **{f"field_{f}": session.field(f) for f in info.field_layers},
    }


def test_every_read_comes_back_equal(session):
    for name, value in every_read(session).items():
        assert wire.loads(wire.dumps(value)) == value, name


def test_requests_come_back_equal():
    for request in (CullRequest(0, 3), FenceRequest(1.0, 2.0, 9.5, 7.25)):
        assert wire.loads(wire.dumps(request)) == request


def test_a_progress_event_comes_back_equal():
    progress = api.Progress(done=5, owed=9, status=api.Status(tick=105, collapsed=False, living=40, splits=1))

    assert wire.loads(wire.dumps(progress)) == progress


def test_a_decoded_column_holds_the_same_numbers():
    column = Column.of(np.arange(12, dtype=np.float32).reshape(3, 4))

    assert np.array_equal(wire.loads(wire.dumps(column)).to_numpy(), column.to_numpy())


def test_only_the_boundarys_own_types_decode():
    """A decoder that built whatever class a message named would be a door into the server."""
    with pytest.raises(ValueError, match="not a type the boundary carries"):
        wire.decode({"$type": "Popen", "args": ["calc"]})
    with pytest.raises(ValueError, match="not a type the boundary carries"):
        wire.decode({"no": "type"})


def test_a_value_the_boundary_does_not_carry_is_refused_on_the_way_out():
    with pytest.raises(TypeError):
        wire.encode(np.zeros(3))


def test_non_finite_numbers_are_refused_rather_than_sent_to_a_browser_that_cannot_parse_them():
    with pytest.raises(ValueError):
        wire.dumps(api.Quote(cost=float("inf"), refusal=None))


def test_the_text_is_plain_json():
    """What a browser's `JSON.parse` reads (#292): no NaN, no Python-specific forms."""
    payload = wire.dumps(api.Status(tick=1, collapsed=False, living=2, splits=0))

    assert json.loads(payload) == {"tick": 1, "collapsed": False, "living": 2, "splits": 0, "$type": "Status"}
