"""Global SoA entity store: typed arrays, free-list allocation, capacity growth.

One array set covers every creature in the world, regardless of species (CLAUDE.md §2.3): a
species differs by which columns it expresses, not by having its own storage, so a world with
one species and a world with fifty both drive the same number of vectorized passes per tick.

Entities are addressed by stable id outside this module, never by row index (CLAUDE.md §1, §2.3):
ids are never reused, so a reference to a released entity fails loudly instead of silently
resolving to whatever new entity now occupies its old row. Row indices are an internal detail of
how columns are packed and are not returned to callers.
"""

from __future__ import annotations

import numpy as np

_COLUMN_NAMES = (
    "x",
    "y",
    "z",
    "energy",
    "age",
    "health",
    "exertion",
    "species_id",
    "drive_scores",
    "genes",
    "alive",
)

# Callers seed these via allocate()'s initial_values kwargs; "alive", "age" and "exertion" are set
# by allocate() itself and are not caller-settable. The last two for the same reason: both count
# something the entity itself did, and an entity that has just been allocated has done nothing.
_SEEDABLE_COLUMN_NAMES = frozenset(_COLUMN_NAMES) - {"alive", "age", "exertion"}


class EntityStoreFull(Exception):
    """allocate() needs more free rows than the store currently has.

    Capacity growth is checked at tick boundaries only (CLAUDE.md §2.3): allocate() never grows
    the arrays itself, so a mid-tick shortfall fails loudly here instead of resizing arrays that
    a vectorized operation elsewhere in the same tick may still hold a view into. The caller must
    call grow() between ticks and retry.
    """


class UnknownEntityError(Exception):
    """An id does not name a currently-live entity.

    Ids are never reused, so this fires for a released id, a double release, or a fabricated one
    — never for a row that quietly became a different entity (CLAUDE.md §8.7: fail loudly).
    """


class EntityStore:
    """Global physical-state columns for every entity, addressed by stable id.

    Columns, each shape ``(capacity,)`` unless noted:
      x, y, z: float32, world units. z is stored from the start per CLAUDE.md §2.6 ("z-capable
          data model first"), even though nothing yet moves along it.
      energy: float32, energy units.
      age: int64, ticks lived — the tick is the only clock (CLAUDE.md §2.1).
      health: float32, unit-free fraction, 0 (dead) to 1 (full health).
      exertion: float32, work per unit of expressed body size — recent effort, accumulated by
          movement and shed each tick by `core.behaviour.exertion.Exertion`, which owns it. Not
          energy units: the size-independent half of the movement bill, so one saturation constant
          means the same tiredness to a mouse and to an elephant.
      species_id: int32, opaque id into the species registry owned elsewhere; -1 means unset.
      drive_scores: ``(capacity, n_drives)`` float32, unit-free utility scores. Which drives
          exist is owned by the behaviour system (CLAUDE.md §8.3); this module only provides a
          column block of the width it's told to.
      genes: ``(capacity, n_genes)`` float32, unit-free gene values over one shared vocabulary
          (CLAUDE.md §2.3). Every entity holds every gene slot regardless of species; which genes
          a species expresses is owned by the genetics system's species registry, not this
          module — an unexpressed slot is stored and inherited exactly like an expressed one.
      alive: bool. True for rows currently holding a live entity.

    Dead rows sit on a free list and are handed back out by allocate(), so capacity grows only on
    a new simultaneous-population high-water mark, never on churn. Growing (grow()) always
    replaces every column with a fresh, larger array and copies old contents across — it never
    resizes in place, so a reference taken to a column before growth keeps its old values rather
    than being mutated out from under whatever held it. Callers must only call grow() at a tick
    boundary; see grow() for why.
    """

    def __init__(self, initial_capacity: int, n_drives: int, n_genes: int) -> None:
        if initial_capacity < 1:
            raise ValueError("initial_capacity must be at least 1")
        if n_drives < 1:
            raise ValueError("n_drives must be at least 1")
        if n_genes < 1:
            raise ValueError("n_genes must be at least 1")

        self._n_drives = n_drives
        self._n_genes = n_genes
        self._allocate_columns(initial_capacity)
        # Popped from the end, so the first allocation hands out row 0.
        self._free_rows: list[int] = list(range(initial_capacity - 1, -1, -1))
        self._row_to_id = np.full(initial_capacity, -1, dtype=np.int64)
        self._id_to_row: dict[int, int] = {}
        self._next_id = 0

    def _allocate_columns(self, capacity: int) -> None:
        self.x = np.zeros(capacity, dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.z = np.zeros(capacity, dtype=np.float32)
        self.energy = np.zeros(capacity, dtype=np.float32)
        self.age = np.zeros(capacity, dtype=np.int64)
        self.health = np.zeros(capacity, dtype=np.float32)
        self.exertion = np.zeros(capacity, dtype=np.float32)
        self.species_id = np.full(capacity, -1, dtype=np.int32)
        self.drive_scores = np.zeros((capacity, self._n_drives), dtype=np.float32)
        self.genes = np.zeros((capacity, self._n_genes), dtype=np.float32)
        self.alive = np.zeros(capacity, dtype=np.bool_)

    @property
    def capacity(self) -> int:
        return self.x.shape[0]

    @property
    def available(self) -> int:
        """Number of free rows allocate() can currently hand out without raising."""
        return len(self._free_rows)

    def free_row_mask(self) -> np.ndarray:
        """(capacity,) bool: True where the row is on the free list, available to allocate().

        Read-only visibility into free-list membership, for the invariant harness (CLAUDE.md
        §6) to detect a row marked both `alive` and free — which can only happen if something
        flips `alive` directly instead of going through release(), desyncing this store's own
        bookkeeping. The free list itself stays private; only membership is exposed.
        """
        mask = np.zeros(self.capacity, dtype=np.bool_)
        mask[self._free_rows] = True
        return mask

    def row_ids(self) -> np.ndarray:
        """(capacity,) int64: the stable id occupying each row, -1 where the row is free.

        A copy, for the same reason `free_row_mask` builds a fresh array: this is the store's own
        bookkeeping and a caller must not be able to mutate it.

        Exists because `alive` answers "is this row occupied" but not "is it still the *same*
        entity". Ids are never reused, so comparing two of these tells a reader whose row indices
        are meaningless to it — the renderer, blending two tick-boundary snapshots (#119) — whether
        a row held one continuous entity across an interval or was freed and handed to a newborn
        inside it. Nothing else can distinguish those: `release` and `allocate` both leave `alive`
        True at the ends of that interval.
        """
        return self._row_to_id.copy()

    def allocate(self, n: int, **initial_values: np.ndarray) -> np.ndarray:
        """Allocate ``n`` new rows from the free list and return their stable ids.

        Every column defaults to its zero value except ``alive`` (True), ``age`` (0) and
        ``exertion`` (0) — the last two reset explicitly rather than relying on the row being
        clean, since a reused row still holds its predecessor's years and its predecessor's
        tiredness, and a newborn has neither. Pass
        column-name keyword arguments of length ``n`` to seed specific columns in the same
        vectorized write — e.g. ``allocate(3, x=..., energy=...)`` — since the ids this call
        returns are the only supported way to address these rows again.

        Raises EntityStoreFull if fewer than ``n`` rows are free; this never grows the store
        itself (see class docstring and grow()).
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        unknown = set(initial_values) - _SEEDABLE_COLUMN_NAMES
        if unknown:
            raise ValueError(f"unknown or non-seedable columns: {sorted(unknown)}")
        for name, values in initial_values.items():
            if np.asarray(values).shape[0] != n:
                raise ValueError(f"initial value for '{name}' must have length {n}")
        if n > len(self._free_rows):
            raise EntityStoreFull(
                f"requested {n} rows but only {len(self._free_rows)} are free; "
                "call grow() at the next tick boundary before retrying"
            )

        rows = np.array([self._free_rows.pop() for _ in range(n)], dtype=np.int64)
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        self._next_id += n

        self.alive[rows] = True
        self.age[rows] = 0
        self.exertion[rows] = 0.0
        self._row_to_id[rows] = ids
        for id_, row in zip(ids.tolist(), rows.tolist()):
            self._id_to_row[id_] = row

        for name, values in initial_values.items():
            getattr(self, name)[rows] = values

        return ids

    def release(self, ids: np.ndarray) -> None:
        """Free the rows for ``ids``, making them available for reuse by allocate().

        Raises UnknownEntityError if any id does not name a currently-live entity.
        """
        ids = np.asarray(ids, dtype=np.int64)
        rows = np.empty(ids.shape[0], dtype=np.int64)
        for i, id_ in enumerate(ids.tolist()):
            row = self._id_to_row.get(id_)
            if row is None:
                raise UnknownEntityError(f"id {id_} is not a live entity")
            rows[i] = row

        self.alive[rows] = False
        self._row_to_id[rows] = -1
        for id_ in ids.tolist():
            del self._id_to_row[id_]
        self._free_rows.extend(rows.tolist())

    def grow(self) -> None:
        """Double capacity, preserving every column's existing values.

        Must only be called at a tick boundary. This module has no notion of "mid-tick" to
        check against, so the hazard this guards is invisible from in here: a vectorized
        operation elsewhere may hold a NumPy view into ``store.energy`` (or any other column)
        for the duration of one tick's computation, and this call replaces that array with a
        new object. The old view stays valid and keeps its pre-growth values (see the "no live
        view survives a resize" test) — it simply stops being the array new work reads and
        writes through, which silently produces wrong results if that happens mid-computation.
        """
        old_capacity = self.capacity
        new_capacity = old_capacity * 2
        old_columns = {name: getattr(self, name) for name in _COLUMN_NAMES}
        old_row_to_id = self._row_to_id

        self._allocate_columns(new_capacity)
        for name, old_array in old_columns.items():
            getattr(self, name)[:old_capacity] = old_array

        self._row_to_id = np.full(new_capacity, -1, dtype=np.int64)
        self._row_to_id[:old_capacity] = old_row_to_id

        self._free_rows.extend(range(new_capacity - 1, old_capacity - 1, -1))
