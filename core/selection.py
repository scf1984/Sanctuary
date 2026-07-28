"""Selection: the only currency crossing a domain-service boundary (CLAUDE.md §2.3).

A Selection is an opaque set of entity rows. Services expose and accept Selections, never raw
row indices, so a caller can narrow, combine, and invert entity sets without ever learning how
any service lays out its arrays internally.

Internally every Selection is a boolean mask over the full entity capacity, not an index array:
composition (``&``, ``|``, ``~``) is then always a single vectorized numpy op, regardless of how
sparse or dense the selection is, rather than a set-intersection over index lists.
"""

from __future__ import annotations

import numpy as np


class Selection:
    """An immutable, composable set of entity rows, backed by a boolean mask.

    mask: (capacity,) bool. True where the row is included.
    """

    __slots__ = ("_mask",)

    def __init__(self, mask: np.ndarray) -> None:
        if mask.ndim != 1:
            raise ValueError("Selection mask must be 1-dimensional")
        if mask.dtype != np.bool_:
            raise ValueError("Selection mask must be boolean")
        mask = mask.copy()
        mask.flags.writeable = False
        self._mask = mask

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> Selection:
        """Build a Selection from a boolean mask of shape (capacity,)."""
        return cls(np.asarray(mask))

    @classmethod
    def from_indices(cls, indices: np.ndarray, capacity: int) -> Selection:
        """Build a Selection from row indices into a store of the given capacity."""
        mask = np.zeros(capacity, dtype=np.bool_)
        mask[np.asarray(indices, dtype=np.int64)] = True
        return cls(mask)

    @classmethod
    def none(cls, capacity: int) -> Selection:
        """The empty selection over a store of the given capacity."""
        return cls(np.zeros(capacity, dtype=np.bool_))

    @classmethod
    def all(cls, capacity: int) -> Selection:
        """Every row of a store of the given capacity."""
        return cls(np.ones(capacity, dtype=np.bool_))

    @property
    def capacity(self) -> int:
        """The size of the store this selection was drawn over."""
        return self._mask.shape[0]

    def to_mask(self) -> np.ndarray:
        """The selection as a read-only (capacity,) boolean mask, for indexing a column."""
        return self._mask

    def to_indices(self) -> np.ndarray:
        """The selection as a sorted (n,) int64 array of row indices."""
        return np.flatnonzero(self._mask)

    def __len__(self) -> int:
        return int(np.count_nonzero(self._mask))

    def __and__(self, other: object) -> Selection:
        if not isinstance(other, Selection):
            return NotImplemented
        self._check_comparable(other)
        return Selection(self._mask & other._mask)

    def __or__(self, other: object) -> Selection:
        if not isinstance(other, Selection):
            return NotImplemented
        self._check_comparable(other)
        return Selection(self._mask | other._mask)

    def __invert__(self) -> Selection:
        return Selection(~self._mask)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Selection):
            return NotImplemented
        return self.capacity == other.capacity and bool(np.array_equal(self._mask, other._mask))

    def __repr__(self) -> str:
        return f"Selection({len(self)}/{self.capacity} rows)"

    def _check_comparable(self, other: Selection) -> None:
        if other.capacity != self.capacity:
            raise ValueError(
                f"cannot combine Selections over different capacities "
                f"({self.capacity} vs {other.capacity})"
            )
