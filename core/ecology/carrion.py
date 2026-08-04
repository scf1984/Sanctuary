"""Carrion: dead flesh lying on the ground, as a field (CLAUDE.md §2.5, issues #185 and #179).

#21 settled that a carcass is a **field** rather than a persisting entity row, for the reason #18
settled the same for plants: a scavenger eats *a body on the ground*, not a numbered corpse, and
identity would cost a row and a genome each on a population that turns over constantly. It was
built during #21 and removed before merge, because **nothing in the world could put mass into it** —
starvation is the only death that existed, and a starved animal has by definition metabolised
itself. #185 recorded that and named predation as the source. This is that source arriving.

**The mass comes from the gap between a wound and a mouthful.** A strike takes far more out of a
prey than the attacker can swallow in one tick, because killing is limited by force and eating is
limited by a gut. The difference is not destroyed — it is a body on the ground:

    damage (what the prey loses)  =  mouthful (what fits in a gut)  +  carrion (what is left lying)

which is the whole of why this field has anything in it. Before that split, a predator drained its
prey one gut-sized bite at a time and there was never a moment where more had died than had been
eaten.

**Eating it is grazing, and deliberately the same verb.** `graze` here is `Plants.graze` with a
different noun: contended per cell, splitting by fraction of demand when several animals want the
same carcass. Three things fall out of that rather than being built:

- **Scavenging exists** with nobody implementing it. Anything allocated toward flesh eats what is
  lying there, whether or not it did the killing.
- **A predator eats its own kill by staying**, which is #100's `commitment` gene finally having a
  payoff in food rather than only in not dithering. A lineage too flighty to hold a bearing kills
  and walks away from the carcass.
- **A kill is contested.** Standing crop is per cell, so a second animal on the same body takes a
  share — which is what makes a kill worth defending without a defence mechanic.

**Decay is a fraction, not a fixed amount**, so every carcass has the same half-life whatever its
size — #107's argument for exertion recovery, and this repository keeps one answer to it. A fixed
subtraction would make a large carcass last proportionally longer, which is backwards: rot works on
surface, not on volume.

**A carcass advertises itself, by the same diffusion grass does.** `scent` is the mass field spread
over the terrain by `Plants.forage_diffusion` — the identical cost-aware operator, deliberately
reused rather than reproduced, so meat is discounted by distance *and* by the climbing in between
exactly as grazing is (#93). Without it a scavenger could only find a body it was already standing
on, and the measurement said so: at 2,000 ticks the ground held 12,000 energy units of meat that
nobody ever came for.

**Standing carrion is already inside `Plants.exported_nutrients`**, and that is why `deposit` moves
no ledger. The energy was on the ledger while it sat in the animal's pool and stays there while it
lies on the ground; the field only records *where* some of that outstanding total physically is.
`decompose` is what finally pays it back, through `Plants.return_nutrients` — the same door faeces
already uses, so the ledger is debited in exactly one place. A conservation check that added this
field's mass to `Plants.total_nutrients()` would double-count it; that total alone is the whole
statement (§6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ecology.plants import Plants
from core.world.terrain import Terrain


@dataclass(frozen=True)
class CarrionConfig:
    """Per-world decomposition rules — never constants in `core/` (§2.1).

    decay_rate: the fraction of standing carrion that rots into the soil each tick, in (0, 1].
        A rate rather than an amount, so a half-life is a property of the world and not of a
        carcass's size. At 1 a body is gone the tick after it falls, which leaves nothing for a
        scavenger and no reason to defend a kill; near 0 the ground fills with meat that never
        returns to the soil and the nutrient loop stalls in a third pool.

        It is the knob that decides whether predation is worth it at all, and #2.1's warning applies
        to it against `FeedingConfig.intake_rate`: how fast a body rots and how fast an animal can
        eat are two halves of "can a predator finish its kill", and tuning either alone is
        meaningless.
    """

    decay_rate: float

    def __post_init__(self) -> None:
        if not 0.0 < self.decay_rate <= 1.0:
            raise ValueError(
                f"decay_rate must be in (0, 1], got {self.decay_rate}; at or below zero a carcass "
                "never returns to the soil and the nutrient loop stalls, and above 1 decomposition "
                "would return more mass than is lying there"
            )


class Carrion:
    """Dead flesh per terrain cell: deposited by predation, eaten by scavengers, rotted into soil.

    mass: `(h, w) float64, energy units` — standing dead flesh per cell, on the same grid and in the
        same units as `Plants.biomass`, because it is the same stuff at a different stage.
    scent: `(h, w) float32, energy units` — how much meat is *reachable* from each cell, the mass
        field spread by the plant field's own cost-aware diffusion.
    """

    # Declared rather than left to inference, exactly as `Plants` does and for the same reason:
    # `np.zeros(shape)` with a statically unknown shape resolves to the stubs' 1-D overload, which
    # then reports every 2-D index as out of range.
    mass: np.ndarray
    scent: np.ndarray

    def __init__(self, terrain: Terrain, plants: Plants, config: CarrionConfig) -> None:
        self.terrain = terrain
        self.plants = plants
        self.config = config
        self.mass = np.zeros(terrain.heights.shape, dtype=np.float64)
        # Rebuilt once per tick by `rebuild_scent`, never cached behind a stamp — §8.7's argument,
        # and the same one `Plants.rebuild_forage` and the cue field both already answer this way.
        self.scent = np.zeros(terrain.heights.shape, dtype=np.float32)

    def deposit(self, x: np.ndarray, y: np.ndarray, mass: np.ndarray) -> None:
        """Lay `mass` energy units of dead flesh at each world position.

        Moves no nutrient ledger: what is deposited was already outstanding while it sat in the
        animal's energy pool, and it stays outstanding while it lies on the ground. See the module
        docstring — `decompose` is the one place this field pays anything back.
        """
        mass = np.asarray(mass, dtype=np.float64)
        if np.any(mass < 0.0):
            raise ValueError("carrion deposit must be non-negative; nothing un-dies")
        rows, cols = self.terrain.cell_indices(x, y)
        np.add.at(self.mass, (rows, cols), mass)

    def graze(self, x: np.ndarray, y: np.ndarray, demand: np.ndarray) -> np.ndarray:
        """Take up to `demand` flesh at each world position; return what was taken.

        The identical contention rule `Plants.graze` uses, and identical for a reason: when several
        animals want one carcass each takes the same *fraction* of what it asked for, so the body
        empties exactly and a hungrier animal still gets proportionally more. That is what makes a
        kill worth standing on rather than a quantity that arrives whoever is nearby.

        The nutrients stay on the export ledger across this call, because they move from the ground
        into an animal without ever having been in the soil. `Feeding` returns the undigested part
        through `Plants.return_nutrients`, exactly as it does for grass.
        """
        demand = np.asarray(demand, dtype=np.float64)
        if np.any(demand < 0.0):
            raise ValueError("carrion demand must be non-negative")

        rows, cols = self.terrain.cell_indices(x, y)
        flat_cell = rows * self.mass.shape[1] + cols
        n_cells = self.mass.size

        # Aggregate demand per cell first: resolved independently, each scavenger would see the
        # whole carcass and n of them would eat n times what is lying there.
        demand_per_cell = np.zeros(n_cells, dtype=np.float64)
        np.add.at(demand_per_cell, flat_cell, demand)
        contested = demand_per_cell[flat_cell]
        standing = self.mass.reshape(-1)[flat_cell]
        share = np.where(contested > 0.0, np.minimum(1.0, standing / contested), 0.0)
        harvested = demand * share

        removed = np.zeros(n_cells, dtype=np.float64)
        np.add.at(removed, flat_cell, harvested)
        self.mass -= removed.reshape(self.mass.shape)
        # Clipped at zero only to absorb the float rounding of summing each cell's harvest twice;
        # `share <= 1` already bounds the real quantity. Same argument as `Plants.graze`.
        np.maximum(self.mass, 0.0, out=self.mass)
        return harvested

    def rebuild_scent(self) -> None:
        """Recompute what a scavenger can smell from a distance — a registered system (§2.1).

        Borrows the plant field's diffusion rather than owning a second one. That is not thrift: two
        operators would be two ranges to keep in step, and §2.1's warning about constants that must
        be tuned as a table is exactly about a pair like "how far grass advertises" and "how far a
        carcass does". A world that wants them to differ changes one config, not two modules.
        """
        self.scent = self.plants.forage_diffusion.spread(self.mass.astype(np.float32))

    def decompose(self) -> None:
        """Rot a fixed fraction of every cell's carrion into that cell's soil, once per tick.

        This is the call that finally debits the export ledger, through the same
        `Plants.return_nutrients` that faeces already uses (§2.5's closed loop, and #91's rule that
        one door is what makes conservation checkable).
        """
        rotted = self.mass * self.config.decay_rate
        occupied = np.nonzero(rotted > 0.0)
        if not occupied[0].size:
            return

        self.mass -= rotted
        rows, cols = occupied
        cell_size = self.terrain.cell_size
        self.plants.return_nutrients(
            cols.astype(np.float64) * cell_size,
            rows.astype(np.float64) * cell_size,
            rotted[occupied],
        )
