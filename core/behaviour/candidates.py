"""The options one `Behaviour.choose` call puts in front of its drives, and readings they share (#258).

Every drive scores the same candidate positions (#114), and several of them read the same thing
there. Hunger, lust and fear each sample the self-excluded scent at the candidates, which is one
(n, n_options, n_channels) block whatever vector the drive then dots it against. At 100,000 animals
that read costs about 110 ms, so computing it per drive spent a fifth of §2.1's tick budget
recomputing a number already in hand (docs/spikes/option-sampling-cost.md).

**What it is for.** Handing every drive one object rather than bare `x, y`, so a reading more than
one drive needs is computed on first request and handed to the rest. Every candidate already lies on
its animal's side of every fence (`Behaviour.candidate_positions`, #278), so whether it can be
reached is not a reading this object carries: #169 asked it here, and clipping onto the wire left it
nothing to answer.

The same holds inside one drive and before any candidate is read, which is why `urgency` is handed
this object as well as `appeal` (#174). What each animal broadcasts is needed three times per call —
lust's own signature, fear's sniff where it stands, and the air at the candidates — and fear's
urgency and appeal read the same seventeen genes. Each was rebuilt per reader: about 55 ms of a
1,400 ms tick at 100,000 animals (docs/spikes/tick-budget.md).

**What it forbids.**

- **Outliving its call.** It is built inside `choose` and dropped when `choose` returns, which is
  what makes a stale reading unrepresentable rather than invalidated: nothing writes the cue field
  between one drive's `appeal` and the next, and nothing holds the object long enough to see the
  next tick's field. A memo inside `Scent` would need a lifetime nobody can see, which is the
  failure #174 rejected for the phenotype.
- **Being a context bag.** It carries the candidates, and readings that more than one reader in
  the call takes — two drives, or one drive's urgency and its appeal — and nothing else (§8.2).
- **Becoming the phenotype block.** Genes are on it only as that kind of reading: a set of columns
  two readers in one call build identically is built once. It is not the once-per-tick block #174
  weighed and the owner deferred (2026-11-07): nothing on it outlives the call, so a gene written
  mid-tick cannot be read stale through it, and a column set only one reader wants gains nothing
  from being put here.
- **A drive changing what the others see.** A shared reading is returned read-only, so a drive
  that scaled it in place would raise instead of silently rescaling every drive after it.
"""

from __future__ import annotations

import numpy as np

from core.ecology.cues import Broadcast, Scent
from core.genetics.service import Genetics
from core.selection import Selection


class Candidates:
    """The candidate positions of one `choose` call, for one selection, and what lies at them.

    x, y: (len(selection), n_options) float64 world positions, the point each option would take the
        animal toward, already clipped into the world and onto the near side of any wire. The null
        option is the last column.
    """

    def __init__(self, selection: Selection, x: np.ndarray, y: np.ndarray) -> None:
        self.selection = selection
        self.x = x
        self.y = y
        self._scent: Scent | None = None
        self._broadcast: Broadcast | None = None
        self._perceived: np.ndarray | None = None
        self._genetics: Genetics | None = None
        self._expressed: dict[tuple[int, ...], np.ndarray] = {}

    def broadcast(self, scent: Scent) -> Broadcast:
        """`scent.broadcast` of this call's selection, read once and returned read-only.

        Takes the `Scent` for the reason `perceived` does. A second `Scent` is refused rather than
        handed the first one's broadcast, which would be the first one's genes.
        """
        self._bind(scent)
        if self._broadcast is None:
            broadcast = scent.broadcast(self.selection)
            for column in (broadcast.x, broadcast.y, broadcast.emission, broadcast.signature):
                column.flags.writeable = False
            self._broadcast = broadcast
        return self._broadcast

    def perceived(self, scent: Scent) -> np.ndarray:
        """(n, n_options, n_channels) read-only: `scent.perceived_at` at the candidates, read once.

        Takes the `Scent` rather than being built with it because the drives are what bind it
        (`Drive` takes its collaborators at construction), and `Behaviour` reads no scent itself.
        A second `Scent` asking for a reading is refused rather than handed the first one's air,
        since a world wiring two would otherwise have one of them smell the other's field.
        """
        broadcast = self.broadcast(scent)
        if self._perceived is None:
            self._perceived = scent.perceived_at(broadcast, self.x, self.y)
            self._perceived.flags.writeable = False
        return self._perceived

    def expressed(self, genetics: Genetics, columns: np.ndarray) -> np.ndarray:
        """(n, len(columns)) float32 read-only: `genetics.expressed_columns`, built once per call.

        Keyed by the columns exactly as asked for, order included, so two readers share a block
        only when they would have built the identical one. A second `Genetics` is refused for the
        reason a second `Scent` is.
        """
        if self._genetics is None:
            self._genetics = genetics
        elif genetics is not self._genetics:
            raise ValueError(
                "these candidates were already read through a different Genetics; one choice "
                "reads one gene matrix"
            )
        key = tuple(np.asarray(columns).tolist())
        block = self._expressed.get(key)
        if block is None:
            block = genetics.expressed_columns(self.selection, columns)
            block.flags.writeable = False
            self._expressed[key] = block
        return block

    def _bind(self, scent: Scent) -> None:
        if self._scent is None:
            self._scent = scent
        elif scent is not self._scent:
            raise ValueError(
                "these candidates were already smelled through a different Scent; one choice "
                "reads one cue field"
            )
