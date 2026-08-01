"""What a set of tunings implies, before anything is run (issue #216).

A world's settings are forty-odd coefficients across a dozen config objects, and what a world *is* —
how many animals it holds, how long a generation takes — has until now been discovered by running
one and looking. This module does the arithmetic instead.

**It predicts forward, from tunings to numbers.** The inverse — searching for tunings that produce a
world you wanted — is deliberately not here (#216), and it would need this anyway: you cannot search
without something to evaluate.

**Nothing here restates a formula that lives elsewhere.** The forecast *builds* the world it is
forecasting and reads the real fields — `Plants.potential_growth`, `Ecology.upkeep`,
`Diet.plant_efficiency` — because a second copy of the growth curve or the cost table is a copy that
drifts, which is §4's rule about a declared rule having to be consulted. Building a world costs a
terrain generation and no ticks, so this is cheap in the way that matters: it does not simulate.

**It is a forecast and says so.** Two of its terms are exact and one is measured:

| term | where it comes from |
|---|---|
| food the field grows per tick | exact — the real `potential_growth` field, summed |
| what one animal costs per tick | exact **for the founding population**; selection moves bodies, so it drifts |
| how much of that growth animals actually capture | **measured**, not derived — see `CAPTURE_FRACTION` |

The third is why this is not a formula. What fraction of a field's production ends up as animal
upkeep depends on how well animals forage, how fast they eat, and what their guts have evolved
toward — none of which is a coefficient. It is measured once, recorded with its evidence, and
reported as its own term so a reader can see exactly which part of the answer is an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.selection import Selection

# What share of the field's growth ends up paying animal upkeep, over and above the losses the
# config already accounts for. Measured rather than derived (§8.5): over `build_demo_world` at
# 700 ticks with `solar_constant` at 4, 8 and 16, total upkeep against field growth read 0.069,
# 0.077 and 0.081 — near-constant across a fourfold change in the sun, which is what makes the
# whole relation usable. Divided by the founding assimilation share (0.5 x 0.25) that is a capture
# of roughly 0.62: about three-fifths of what grows is grazed before it senesces.
#
# It is a module constant rather than per-world config deliberately, and the distinction matters:
# §2.1's rule is that coefficients of the *simulation* are per-world, and this is a coefficient of
# a *model of* the simulation. A world does not have one; a forecast does. Overridable per call so
# a world whose foraging differs sharply can say so.
CAPTURE_FRACTION = 0.62


@dataclass(frozen=True)
class Forecast:
    """What a config implies, with the estimated part kept visible rather than folded away.

    field_growth: energy units per tick, over the whole field. Light-limited potential, summed —
        exact, and the term the sun scales directly.
    upkeep_per_animal: energy units per tick for one founding animal, at its own position's
        temperature. Exact for the founders and an underestimate later if selection favours bigger
        or faster bodies, which it usually does.
    assimilation_share: the fraction of grazed biomass that becomes animal energy — the gut's
        ceiling times the diet frontier at the founding allocation. Exact for the founders.
    capture_fraction: the estimated term. See the module docstring.
    carrying_capacity: the product of the four above, and the number this module exists for:

        ``field_growth x capture_fraction x assimilation_share / upkeep_per_animal``

        **It is a ceiling a world climbs toward, not where a world is.** A young population sits
        far below it through no fault of the model: sixty founders after five hundred ticks reach
        about 970 against a forecast of 5,557, and that gap is arrival time. Comparing the two
        needs a world that has had the ticks to get there.

        It is also a *scale* rather than a prediction of any particular run. Two worlds built from
        one config settle at different numbers (§2.2), and a world that has evolved for a thousand
        generations has different bodies than the one this was computed from.
    generation_ticks: maturity plus gestation, averaged over the founding draw. What §2.1's "a
        generation is about one real day" is a statement about, and the number that turns any
        answer in generations into an answer in time.
    """

    field_growth: float
    upkeep_per_animal: float
    assimilation_share: float
    capture_fraction: float
    carrying_capacity: float
    generation_ticks: float

    def as_dict(self) -> dict:
        """A plain mapping, for a client or a log. Every field is already a float."""
        return {
            "field_growth": self.field_growth,
            "upkeep_per_animal": self.upkeep_per_animal,
            "assimilation_share": self.assimilation_share,
            "capture_fraction": self.capture_fraction,
            "carrying_capacity": self.carrying_capacity,
            "generation_ticks": self.generation_ticks,
        }


def forecast(world, capture_fraction: float = CAPTURE_FRACTION) -> Forecast:
    """Read what `world`'s tunings imply. Runs no ticks and changes nothing.

    Takes a built world rather than a config, for two reasons. Building is `build_world`'s job alone
    (§7.2) and a second construction path here would be the duplicate assembly that rule exists to
    prevent; and the founders are what the upkeep and diet terms are read *from*, so a config on its
    own would mean drawing a population to measure — which is building a world with extra steps.

    Raises ValueError for a world with no founders left to measure: every term below is a property
    of a population, and a forecast over nobody would be a number with no referent (§8.7).
    """
    if not 0.0 < capture_fraction <= 1.0:
        raise ValueError(
            f"capture_fraction must be in (0, 1], got {capture_fraction}; animals cannot graze "
            "more than the field grows"
        )
    living = Selection.from_mask(world.store.alive & (world.store.age >= 0))
    if not len(living):
        raise ValueError("cannot forecast a world with no living animals to measure")

    expressed = world.genetics.expressed(living)
    growth = float(world.plants.potential_growth.sum())
    upkeep = float(np.mean(world.ecology.upkeep(living)))
    # The gut's ceiling times the diet frontier: a founding population undecided about what it eats
    # converts plants at the frontier's penalty for being a generalist (#102), and that penalty is
    # part of the answer rather than a detail of it.
    share = world.config.feeding.assimilation_max * float(
        np.mean(world.feeding.diet.plant_efficiency(expressed))
    )
    return Forecast(
        field_growth=growth,
        upkeep_per_animal=upkeep,
        assimilation_share=share,
        capture_fraction=capture_fraction,
        carrying_capacity=growth * capture_fraction * share / upkeep,
        generation_ticks=_generation_ticks(world, expressed),
    )


def _generation_ticks(world, expressed: np.ndarray) -> float:
    """Mean ticks from conception to the parent's own first breeding: gestation plus maturity.

    Read off the founders' expressed phenotype rather than off the config's founding ranges,
    because both are magnitudes (§2.5) and the mean of `abs` over a range that crosses zero is not
    the mean of the range. Reading the population is reading what the world actually has.
    """
    genes = world.genes
    maturity = expressed[:, genes.index_of(world.config.conception.maturity_gene)]
    gestation = expressed[:, genes.index_of(world.config.conception.gestation_gene)]
    return float(np.mean(maturity) + np.mean(gestation))
