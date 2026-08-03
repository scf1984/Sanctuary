"""Domain service base: column-block ownership, enforced rather than documented (CLAUDE.md §2.3).

`Ecology`, `Genetics`, `Behaviour` and friends each own the store columns their issue assigns
them and expose ecological verbs over `Selection` (CLAUDE.md §2.3, §4). This module gives them a
shared base that registers each service's declared columns against a per-world `ColumnRegistry`,
so a second service claiming an already-owned column, or a service writing outside what it
declared, fails at the point of the mistake instead of relying on every future author reading
this file's docstring.
"""

from __future__ import annotations

from core.selection import Selection


class ColumnOwnershipError(Exception):
    """A service claimed or wrote a column it does not own."""


class ColumnRegistry:
    """Tracks which service owns each store column, for one world.

    One instance per world (CLAUDE.md §4: no singletons — passed explicitly to every service
    constructed against a given store), so two worlds' ownership claims can never collide.
    """

    def __init__(self) -> None:
        self._owner_by_column: dict[str, str] = {}

    def claim(self, columns: tuple[str, ...], owner: str) -> None:
        """Register `owner` as the sole writer of `columns`.

        Raises ColumnOwnershipError if any column is already claimed by a different owner.
        """
        for column in columns:
            existing = self._owner_by_column.get(column)
            if existing is not None and existing != owner:
                raise ColumnOwnershipError(
                    f"column '{column}' is already owned by {existing}; {owner} cannot also own it"
                )
        for column in columns:
            self._owner_by_column[column] = owner

    def owner_of(self, column: str) -> str | None:
        """The name of the service owning `column`, or None if unclaimed."""
        return self._owner_by_column.get(column)


class DomainService:
    """Base for services that own a subset of a store's columns.

    Subclasses set the class attribute `owns` to the column names they govern. Construction
    registers those columns against `registry`, raising ColumnOwnershipError if another service
    already owns one of them. `write` then rejects any column outside `owns`, which is the
    enforcement CLAUDE.md §2.3 asks for: a misdirected write is a caught error, not a silent
    cross-service mutation.
    """

    owns: tuple[str, ...] = ()

    def __init__(self, store: object, registry: ColumnRegistry) -> None:
        if not self.owns:
            raise ValueError(f"{type(self).__name__} must declare a non-empty `owns`")
        registry.claim(self.owns, type(self).__name__)
        self.store = store
        self._registry = registry

    def write(self, column: str, selection: Selection, values: object) -> None:
        """Vectorized-write `values` into `column` at `selection`'s rows.

        Raises ColumnOwnershipError if `column` is not in this service's declared `owns`.
        """
        self._guard(column)
        getattr(self.store, column)[selection.to_mask()] = values

    def write_at(self, column: str, rows: object, values: object) -> None:
        """Vectorized-write `values` into `column` at explicit `rows`, in the caller's order.

        The ordered sibling of `write`, and it exists because a `Selection` is a mask: masks are
        always ascending, so they cannot express a pairing whose members cross in row space. #191
        established the reading half of this (`Genetics.genes_at`, `expressed_at`) after a pairing
        built from *position* was silently rewired; predation (#179) is the first writer that needs
        the same, since a prey and the animal eating it are paired by where they are standing.

        Same ownership guard as `write`, deliberately — the guard is the point of the base class,
        and an ordered write that skipped it would be the cross-service mutation §2.3 forbids.

        Rows must be unique. A repeated row is written once under fancy indexing rather than
        accumulated, so a caller that pairs one entity twice would silently lose a term; that is a
        property of the caller's pairing, checked where the pairing is made rather than here (§8.2).
        """
        self._guard(column)
        getattr(self.store, column)[rows] = values

    def _guard(self, column: str) -> None:
        if column not in self.owns:
            owner = self._registry.owner_of(column)
            raise ColumnOwnershipError(
                f"{type(self).__name__} cannot write column '{column}'; "
                f"it is owned by {owner or 'no one'}"
            )
