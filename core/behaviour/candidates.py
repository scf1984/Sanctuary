"""The options one `Behaviour.choose` call puts in front of its drives, and readings they share (#258).

Every drive scores the same candidate positions (#114), and several of them read the same thing
there. Hunger, lust and fear each sample the self-excluded scent at the candidates, which is one
(n, n_options, n_channels) block whatever vector the drive then dots it against. At 100,000 animals
that read costs about 110 ms, so computing it per drive spent a fifth of §2.1's tick budget
recomputing a number already in hand (docs/spikes/option-sampling-cost.md).

**What it is for.** Handing every drive one object rather than bare `x, y`, so a reading more than
one drive needs is computed on first request and handed to the rest.

**What it forbids.**

- **Outliving its call.** It is built inside `choose` and dropped when `choose` returns, which is
  what makes a stale reading unrepresentable rather than invalidated: nothing writes the cue field
  between one drive's `appeal` and the next, and nothing holds the object long enough to see the
  next tick's field. A memo inside `Scent` would need a lifetime nobody can see, which is the
  failure #174 rejected for the phenotype.
- **Being a context bag.** It carries the candidates and readings that more than one drive takes
  from them, and nothing else (§8.2). The phenotype is not on it: whether genes are read once per
  call is #174's open question, and this object does not answer it by accident.
- **A drive changing what the others see.** A shared reading is returned read-only, so a drive
  that scaled it in place would raise instead of silently rescaling every drive after it.
"""

from __future__ import annotations

import numpy as np

from core.ecology.cues import Scent
from core.selection import Selection


class Candidates:
    """The candidate positions of one `choose` call, for one selection, and what lies at them.

    x, y: (len(selection), n_options) float64 world positions, the point each option would take the
        animal toward, already clipped into the world. The null option is the last column.
    """

    def __init__(self, selection: Selection, x: np.ndarray, y: np.ndarray) -> None:
        self.selection = selection
        self.x = x
        self.y = y
        self._scent: Scent | None = None
        self._perceived: np.ndarray | None = None

    def perceived(self, scent: Scent) -> np.ndarray:
        """(n, n_options, n_channels) read-only: `scent.perceived_at` at the candidates, read once.

        Takes the `Scent` rather than being built with it because the drives are what bind it
        (`Drive` takes its collaborators at construction), and `Behaviour` reads no scent itself.
        A second `Scent` asking for a reading is refused rather than handed the first one's air,
        since a world wiring two would otherwise have one of them smell the other's field.
        """
        if self._perceived is None:
            self._scent = scent
            self._perceived = scent.perceived_at(self.selection, self.x, self.y)
            self._perceived.flags.writeable = False
        elif scent is not self._scent:
            raise ValueError(
                "these candidates were already smelled through a different Scent; one choice "
                "reads one cue field"
            )
        return self._perceived
