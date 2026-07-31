"""Feeding: the first energy transfer between trophic levels (CLAUDE.md §2.5, issue #19).

Before this, `TICK_ORDER` charged metabolic upkeep with nothing before it that could pay: energy
only ever *left* the animals, and an assembled world was a slow drain to nothing. This is where
sunlight, having entered the world as plant biomass (#18), finally reaches something alive.

Three rules govern the transfer, and all three are conservation statements rather than mechanics:

**A creature cannot gain more energy than was invested in building what it ate.** The harvest is
multiplied by a conversion in [0, 1] and nothing else, so §6's "energy is never created" holds by
construction rather than by a check. `core.ecology.diet` supplies the gut's *relative* competence
at this substrate; `assimilation_max` is what even a perfect gut leaves behind.

**The remainder is not destroyed.** What is not assimilated is faeces, and it goes straight back
into the soil of the cell it was eaten in. That is why a poor digester fertilises the ground it
grazes, with nobody writing a fertilisation mechanic — and it is what keeps `total_nutrients()`
exactly conserved across a step that moves nutrients out of the field and into an animal.

**A meal spans ticks, and no column records it.** §2.1 wants roughly 10² feeding events per
lifetime against reality's 10³–10⁴, which is a statement about *handling time*: an animal takes a
bounded mouthful per tick, so a meal is spread over many of them by the intake rate alone. The
scoping note on #19 called for a column holding what is being eaten and how much is left; with
carrion settled as a field (#21) and grazing being per-cell, the field under the animal *is* that
state, and the column would have had no reader (§8.2). Doggedness about staying put is already
#100's `commitment` acting on the hunger drive, not a second notion of the same thing.

**Intake does not scale with diet efficiency**, deliberately. Scaling demand by the same number
that scales yield would make the realised conversion `efficiency²`, silently doubling #102's
frontier exponent — two coefficients describing one preference, which §2.1 warns drift apart. So a
gut processes what a gut of that size can process, and what it *gets* is what it can digest. An
animal eating something it cannot use passes the whole mouthful through, wasting the tick and
fertilising the ground; it is self-punishing without a gate deciding what a creature is allowed to
try to eat.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.ecology.diet import Diet
from core.ecology.plants import Plants
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.registry import GeneRegistry, Unit
from core.genetics.service import Genetics
from core.selection import Selection


@dataclass(frozen=True)
class FeedingConfig:
    """Per-world feeding rules — never constants in `core/` (§2.1).

    intake_rate: energy units of biomass an animal of unit size can process per tick. This is the
        handling-time knob, and therefore the one that sets §2.1's ~10² feeding events per
        lifetime: a mouthful small against what a lifetime needs is what makes eating an activity
        rather than an instant.
    assimilation_max: the fraction of eaten biomass a *perfect* gut converts, in (0, 1]. Bounded
        above by 1 because anything more is energy created out of grass (§6). It is separate from
        the diet allocation because it is a fact about chemistry rather than about the eater —
        `core.ecology.diet` says how good this gut is *relative* to a perfect one, and this says
        what perfect means. Keeping them apart is what leaves #102's `share ** p` untouched.
    size_gene: the gene whose expressed value scales the mouthful. A bigger body processes more per
        tick, which is also what gives `size` its first benefit — until now it only ever charged
        upkeep (#17) and locomotion (#25), so selection had no reason to keep any.
    """

    intake_rate: float
    assimilation_max: float
    size_gene: str

    def __post_init__(self) -> None:
        if self.intake_rate <= 0.0:
            raise ValueError(f"intake_rate must be positive, got {self.intake_rate}")
        if not 0.0 < self.assimilation_max <= 1.0:
            raise ValueError(
                f"assimilation_max must be in (0, 1], got {self.assimilation_max}; above 1 is "
                "energy created out of what was eaten (§6), and at or below 0 nothing in the "
                "world can ever eat"
            )


class Feeding:
    """Moves energy from the plant field into animals, once per tick.

    Owns no store column. Energy is `Ecology`'s and biomass is `Plants`', so this service decides
    only *how much* moves and hands each side its own half — which is the same reason
    `core.behaviour.movement` cannot subtract from the pool it spends from (§2.3).
    """

    def __init__(
        self,
        store: EntityStore,
        plants: Plants,
        genetics: Genetics,
        ecology: Ecology,
        diet: Diet,
        genes: GeneRegistry,
        config: FeedingConfig,
    ) -> None:
        self.store = store
        self.plants = plants
        self.genetics = genetics
        self.ecology = ecology
        self.diet = diet
        self.config = config
        # A body size is a bare multiplier on the mouthful, hence dimensionless. Resolved through
        # the registry so a world declaring `size` as a length is rejected here rather than
        # producing a mouthful in the wrong denomination that nothing could notice (#112).
        self._size_index = genes.index_of(config.size_gene, unit=Unit.DIMENSIONLESS)

    def feed(self, selection: Selection) -> None:
        """Take one tick's mouthful for every entity in `selection`.

        `selection` is the caller's choice of who eats; pass the living. Nothing here filters, for
        the same reason `Ecology.drain` does not: a tick loop feeding anything other than the
        living is a bug in the loop rather than a condition to absorb quietly (§8.7).
        """
        mask = selection.to_mask()
        x = self.store.x[mask]
        y = self.store.y[mask]

        phenotype = self.genetics.expressed(selection)
        demand = self.config.intake_rate * phenotype[:, self._size_index]
        # Contention lives in `graze`: grazers sharing a cell take the same fraction of what each
        # asked for, so the cell empties exactly instead of each animal seeing the whole crop.
        harvested = self.plants.graze(x, y, demand)

        conversion = self.config.assimilation_max * self.diet.plant_efficiency(phenotype)
        self.ecology.gain(selection, harvested * conversion)
        self.plants.return_nutrients(x, y, harvested * (1.0 - conversion))
