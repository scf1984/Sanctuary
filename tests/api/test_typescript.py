"""The client's message types are generated from `api.values`, never written by hand (#306)."""

from typing import Optional

import pytest

from api import typescript, values


def test_the_committed_types_are_what_the_values_generate():
    """CI's staleness check. If this fails, run `python -m api.typescript --write` and commit."""
    assert typescript.TARGET.read_text(encoding="utf-8") == typescript.render()


def test_every_carried_type_has_an_interface():
    rendered = typescript.render()
    for name in values.__all__:
        if name != "Request":
            assert f"export interface {name} {{" in rendered


@pytest.mark.parametrize(
    "hint, spelling",
    [
        (int, "number"),
        (Optional[str], "string | null"),
        (tuple[str, ...], "readonly string[]"),
        (tuple[int, int], "readonly [number, number]"),
        (tuple[Optional[float], ...], "readonly (number | null)[]"),
        (values.Status, "Status"),
        (values.Column, "Column"),
    ],
)
def test_annotations_spell_as_their_wire_form(hint, spelling):
    assert typescript.typescript_type(hint) == spelling


def test_a_type_the_wire_cannot_carry_is_refused_rather_than_spelled_any():
    with pytest.raises(TypeError, match="no TypeScript form"):
        typescript.typescript_type(dict[str, int])


def test_writing_puts_the_file_where_the_client_reads_it(tmp_path, monkeypatch):
    target = tmp_path / "values.ts"
    monkeypatch.setattr(typescript, "TARGET", target)
    typescript.main(["--write"])
    assert target.read_text(encoding="utf-8") == typescript.render()
