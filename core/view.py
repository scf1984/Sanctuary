"""Per-entity debug/UI view, forbidden in tick loops (CLAUDE.md §2.3, §3.3).

The diagnostic viewer's click-to-inspect (§3.3) needs one entity's fields as plain Python
values. That is a Python-level scalar read against the SoA store — exactly what CLAUDE.md §2.3
says forfeits the whole performance case for SoA if it happens inside a tick loop. `EntityView`
takes a `Selection` (never a raw row index) narrowed to one row, and refuses to construct while a
`TickContext` reports a tick in progress, so a hot-loop use fails loudly (§8.7) at the call site.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np

from core.selection import Selection


class TickContext:
    """Whether a tick is currently executing, shared across one world's services and views.

    One instance per world (CLAUDE.md §4: no singletons — passed explicitly), so two worlds'
    tick state can never collide. The tick loop wraps each tick's work in
    ``with tick_context.tick(): ...``; `EntityView` checks `in_tick` at construction time.
    """

    def __init__(self) -> None:
        self.in_tick = False

    @contextmanager
    def tick(self) -> Iterator[None]:
        if self.in_tick:
            raise RuntimeError("TickContext.tick() is not reentrant")
        self.in_tick = True
        try:
            yield
        finally:
            self.in_tick = False


class EntityView:
    """A read-only snapshot of one entity's columns as plain Python values.

    Construction pulls `columns` out of `store` at `selection`'s single row and copies them out
    as native Python scalars/lists — after construction the view holds no reference to `store`,
    so it cannot be used to smuggle a live array reference past the point a tick loop would
    detect it.
    """

    def __init__(
        self,
        store: object,
        selection: Selection,
        columns: tuple[str, ...],
        tick_context: TickContext,
    ) -> None:
        if tick_context.in_tick:
            raise RuntimeError(
                "EntityView must not be constructed inside a tick loop; it is a per-entity "
                "Python-level access that forfeits the SoA performance case (CLAUDE.md §2.3)"
            )
        if len(selection) != 1:
            raise ValueError(
                f"EntityView requires a Selection of exactly one row, got {len(selection)}"
            )
        row = selection.to_indices()[0]
        self._fields = {name: _as_python(getattr(store, name)[row]) for name in columns}

    def __getattr__(self, name: str) -> object:
        try:
            return self._fields[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self) -> str:
        return f"EntityView({self._fields!r})"


def _as_python(value: np.ndarray | np.generic) -> object:
    """A numpy scalar or per-entity sub-array, converted to a native Python value."""
    return value.tolist() if isinstance(value, np.ndarray) else value.item()
