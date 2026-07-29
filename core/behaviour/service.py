"""Behaviour domain service: authored drives competing each tick by utility score (issue #22).

This replaces the prototype's state-transition graph, which was declared and then consulted by
nothing (CLAUDE.md §1). A fixed set of drives each scores every entity, the highest score wins,
and the winner is what downstream systems act on. Two properties matter more than the scoring
itself:

- **Behaviour stays explainable.** `breakdown()` returns every drive's score by name, so "it fled
  because fear outscored hunger" is a fact recoverable from the store rather than a story told
  about it. The intervention gameplay (CLAUDE.md §2.7) and the diagnostic viewer (§3.3) both rest
  on that.
- **Drives are registered, not dispatched.** Adding one is a `register()` call; the scoring loop,
  the competition, and the inspection surface are all unchanged. There is deliberately no
  `if drive is hunger` anywhere in this module.

A drive is a **vectorized function over the global arrays producing a score column**, never a
per-entity Python call (CLAUDE.md §2.3). The loop here iterates drives — five of them — not
entities.

**Weights and thresholds are per-world config today, genes tomorrow.** Every drive in
`core.behaviour.drives` takes a frozen config dataclass. #23 replaces that source with the gene
matrix, at which point boldness and sociality evolve rather than being tuned; that substitution is
the entirety of #23's scope and nothing in this module changes for it.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from core.entities.store import EntityStore
from core.selection import Selection
from core.services import ColumnRegistry, DomainService


class DriveRegistrationError(Exception):
    """A drive cannot be registered: its name is taken, or the score block is already full."""


class Drive(Protocol):
    """A named, vectorized appetite: how much each entity wants one thing, right now.

    Collaborators — the plant field, the climate, the genetics service — are bound when the drive
    is constructed, not passed per call. That follows the precedent §6 sets for invariants: one
    uniform signature for every drive, with whatever a particular drive happens to need closed
    over, rather than a context argument enumerating domains that do not all exist yet (§8.2).
    """

    name: str

    def score(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32, unit-free: urgency per entity, in ascending row order.

        Zero means "no pull at all", and a drive scoring zero cannot win (see
        `Behaviour.winning_drive`). Scores are compared across drives directly, so their
        magnitudes are the tuning surface that decides which appetite outranks which.
        """
        ...


class Behaviour(DomainService):
    """Owns the `drive_scores` column block: what every entity currently wants, and how much.

    Registration order is the column order within the block, and it is also the tie-break: given
    equal scores the earlier-registered drive wins. That is the deterministic resolution #22 asks
    for — the simulation is non-deterministic in its randomness (§2.2), but a drive contest fed
    identical scores must not be, or a score breakdown could not be reconciled with the action
    taken and the viewer's explanation would be a guess.
    """

    owns = ("drive_scores",)

    # Narrows DomainService.store (typed `object`, the base being store-shape-agnostic) to the
    # concrete EntityStore whose `drive_scores` block this service fills.
    store: EntityStore

    def __init__(self, store: EntityStore, registry: ColumnRegistry) -> None:
        super().__init__(store, registry)
        self._drives: list[Drive] = []

    @property
    def drive_names(self) -> tuple[str, ...]:
        """Registered drive names, in the column order they occupy in `drive_scores`."""
        return tuple(drive.name for drive in self._drives)

    @property
    def _width(self) -> int:
        return int(self.store.drive_scores.shape[1])

    def register(self, drive: Drive) -> None:
        """Add `drive` to the contest, in the next free column of the score block.

        Raises DriveRegistrationError on a duplicate name — names address drives from the viewer
        and from `driven_by`, so two answering to one would silently resolve to the first — or if
        the store's block has no column left. The block's width is fixed at store construction, so
        overflowing it is a world-assembly error and belongs at assembly time, not at the first
        tick (§8.7).
        """
        if drive.name in self.drive_names:
            raise DriveRegistrationError(f"a drive named '{drive.name}' is already registered")
        if len(self._drives) == self._width:
            raise DriveRegistrationError(
                f"the store's drive_scores block holds {self._width} drives and all are taken; "
                f"construct the EntityStore with a wider n_drives to register '{drive.name}'"
            )
        self._drives.append(drive)

    def score(self, selection: Selection) -> None:
        """Run every drive over `selection` and record the results in `drive_scores`.

        Called once per tick, before anything reads a winner. The scores are stored rather than
        recomputed on demand because the winner, the breakdown the viewer shows, and whatever
        acted on the decision must all describe the *same* tick — recomputing would let an
        explanation drift from the action it explains as the world changed underneath it.
        """
        n = len(selection)
        scores = np.zeros((n, self._width), dtype=np.float32)
        for column, drive in enumerate(self._drives):
            values = np.asarray(drive.score(selection), dtype=np.float32)
            if values.shape != (n,):
                # Checked rather than left to NumPy: a scalar or length-1 return broadcasts
                # cleanly across the column, handing one animal's motivation to the whole world.
                raise ValueError(
                    f"drive '{drive.name}' scored shape {values.shape} for {n} entities; "
                    f"expected ({n},)"
                )
            scores[:, column] = values
        self.write("drive_scores", selection, scores)

    def scores(self, selection: Selection) -> np.ndarray:
        """(len(selection), n_drive_columns) float32: the raw score block, in ascending row order."""
        return self.store.drive_scores[selection.to_mask()]

    def breakdown(self, selection: Selection) -> dict[str, np.ndarray]:
        """Each registered drive's score column by name — the "why" behind the winner (§3.3).

        Returns arrays over the whole selection rather than one entity's values, so the viewer's
        overlays and its click-to-inspect read the same call. Per-entity scalar access is
        `core.view.EntityView`'s job and is forbidden in a tick loop; this is not.
        """
        scores = self.scores(selection)
        return {drive.name: scores[:, column] for column, drive in enumerate(self._drives)}

    def winning_drive(self, selection: Selection) -> np.ndarray:
        """(len(selection),) int32: the index of each entity's winning drive, or -1 for none.

        -1 means every drive scored zero: a creature that is fed, cool, safe, immature and healthy
        wants nothing, and reporting the first-registered drive for it — which is what a bare
        argmax over an all-zero row returns — would fabricate a motivation out of no evidence
        (§8.7). Downstream systems read -1 as "this entity has no reason to act this tick".

        Only the registered prefix of the block competes. The trailing columns of a store built
        with room to spare hold zeros, and a bare argmax would happily select one.
        """
        active = self.scores(selection)[:, : len(self._drives)]
        if active.shape[1] == 0:
            return np.full(len(selection), -1, dtype=np.int32)
        # argmax takes the first maximum, so ties fall to the earlier-registered drive.
        winner = np.argmax(active, axis=1).astype(np.int32)
        return np.where(active.max(axis=1) > 0.0, winner, -1).astype(np.int32)

    def driven_by(self, name: str, selection: Selection) -> Selection:
        """The entities in `selection` whose winning drive is `name`.

        This is where a decision leaves the behaviour service: #19 asks who is feeding, #20 who is
        seeking a mate, #25 who is on the move. Handing back a Selection rather than an action
        object keeps the *doing* with the issue that owns the mechanic — this service decides what
        an animal wants, never how the world changes as a result.
        """
        try:
            column = self.drive_names.index(name)
        except ValueError:
            raise KeyError(
                f"no drive named '{name}' is registered; have {list(self.drive_names)}"
            ) from None
        winners = np.zeros(selection.capacity, dtype=np.bool_)
        winners[selection.to_indices()] = self.winning_drive(selection) == column
        return Selection.from_mask(winners)
