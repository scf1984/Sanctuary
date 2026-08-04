"""Thirst with something behind it: water lost to heat, drunk at a lake, and paid for in energy
(CLAUDE.md §2.5, issue #156).

`Thirst` has scored since #22 and had **nothing to act on**. There was no hydration column, no way to
drink, and no consequence for never drinking, so the drive read ambient heat and pointed nowhere.
That is the sharpest instance of #126 — in the first assembled world all forty founders wanted water
in a world with no way to drink and not one animal moved for the entire run — and both world configs
still hold the thirst weight at 0.2 purely to stop it outscoring hunger, a coefficient chosen to
dodge a structural gap rather than to describe an ecology. #114 stopped that gap freezing anybody,
and this closes it.

## The column is a deficit, not a level

`dehydration` is 0 for a fully watered animal and rises toward 1 as it dries out. Storing the
*deficit* rather than the reserve is what lets a reused row be cleared to zero like every other
column (`_CLEARED_ON_ALLOCATE`) and have that mean the right thing: a newborn is not thirsty. Stored
as a level, zero would mean "born completely dry" and every young would die in its first tick unless
something remembered to seed it — a rule nobody would notice was missing until births existed.

It is the same shape `exertion` has for the same reason (#107), and the two are deliberately
identical in kind: a deficit that the world adds to and an action removes.

## It is a fraction, so body size cancels

A large animal holds more water and loses more of it, and both scale together, so the *fraction* it
has lost is size-independent — which means one loss rate means the same thing to a mouse and to an
elephant. That is #107's argument for exertion being work per unit of body size, applied to the one
other deficit in the world. A capacity in "water units" would need a second constant relating body
size to reserve, and it would buy nothing that the fraction does not already say.

## Death by dehydration falls out; it is not a rule

There is no dehydration check anywhere and no thirst-specific mortality. A dry animal simply costs
more to run — `Ecology.upkeep` multiplies by `1 + dehydration_penalty × dehydration` — so it burns
its energy pool faster, empties it, and dies through `Ecology.starving` and `Death`, which already
exist and are unchanged.

That is the shape §2.5 settles for senescence (*"death then falls out of starvation and predation,
mechanisms that already exist, rather than from an age check"*) and the one #147 proposes for
asphyxiation, and it is why the repository has one answer to "how does a failing body kill its
owner". A `hydration <= 0 → dead` branch would be a second mortality path that no invariant covers
and that every future depletion mechanic would copy.

**It also gets the ecology right.** A dehydrated animal is not merely on a timer: it is expensive,
so it must eat *more* to stay alive, which sends it foraging while it should be drinking. Thirst and
hunger therefore compete for the same animal's next heading rather than taking turns, which is what
the drive contest is for.

## Loss is scaled by heat, which is where the old thirst score went

`Thirst` used to read ambient temperature as its whole urgency. That term is not deleted — it moves
here, where it belongs: heat does not make an animal *want* water, it makes an animal *lose* water,
and wanting follows from having lost. Once it is a rate rather than a score, a hot world dries
animals out faster and a cold one barely at all, and the drive reads the same deficit either way.

## Drinking is a rate, and that is what makes a waterhole dangerous

A drink takes several ticks rather than filling instantly, exactly as a meal does. With water static
(#165 — a lake cannot shrink yet) there is no depletion to contend over, so **time at the water's
edge is the only cost a waterhole can charge**, and it is the real one: an animal standing still by
a lake is an animal a predator can reach (#179). Instant drinking would make water free and remove
the one place the map reliably concentrates animals.

Drinking itself costs no energy, for the reason §2.5 gives about resting: the cost was paid walking
there, and charging for the act would make water a third way to starve rather than the escape from
thirst it exists to be.

## Finding water is a diffused field, exactly as finding food is

`reachable` is the standing-water depth spread over the terrain by the same cost-aware operator the
forage field uses (#93), so a thirsty animal reads a heading off its gradient and water is
discounted by distance *and* by the climbing in between. Sampling `Water.is_drinkable_at` at the
candidate headings instead was tried in design and rejected: candidates sit one `look_ahead` away,
lakes are sparse, and a binary reading means thirst can only steer when a candidate lands *on*
water — which is #126 again in a new place, and the exact failure this issue exists to end.

**It is built once and never rebuilt**, because `Water` is derived from terrain and never advanced
(#165). That is the one difference from the forage field, which grazing changes every tick. When
water becomes dynamic this becomes a per-tick system beside `rebuild_forage`, and the comment on the
field says so rather than leaving the next reader to discover it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.entities.store import EntityStore
from core.selection import Selection
from core.services import ColumnRegistry, DomainService
from core.world.barriers import Barriers
from core.world.climate import Climate
from core.world.diffusion import CostAwareDiffusion, DiffusionConfig
from core.world.terrain import Terrain
from core.world.water import Water


@dataclass(frozen=True)
class HydrationConfig:
    """Per-world water rules — never constants in `core/` (§2.1).

    loss_rate: the fraction of its reserve an animal loses per tick at `neutral_temperature`, in
        (0, 1]. This is the clock on the whole mechanic: it decides how many ticks of not drinking
        an animal survives, and therefore how often the herd must return to water.
    heat_scaling: how much faster water is lost per degree above `neutral_temperature`. Where the
        old thirst score's temperature term went — heat does not make an animal *want* water, it
        makes an animal *lose* water. Zero is a world where climate does not dry anything out,
        which is legal and dull.
    neutral_temperature: degrees C at which `loss_rate` applies unscaled. Below it, loss slows
        rather than reversing: cold air does not hydrate anybody, so the scaling is floored at
        zero rather than allowed to make a cold animal gain water.
    drink_rate: the fraction of its reserve an animal restores per tick standing at water, in
        (0, 1]. A rate rather than a refill, so a drink takes several ticks and standing at the
        water's edge is exposure — the only cost a waterhole can charge while lakes cannot shrink
        (#165). It must be tuned against `loss_rate` as one pair (§2.1): the ratio of the two is
        how much of its life an animal spends drinking, and either alone says nothing.
    reachability: how far water advertises itself, spread by the same cost-aware operator the
        forage field uses (#93). Declared separately from `PlantsConfig.forage_diffusion` because
        a lake is visible from further than a meadow in most worlds, and nothing forces the two to
        move together.
    """

    loss_rate: float
    heat_scaling: float
    neutral_temperature: float
    drink_rate: float
    reachability: DiffusionConfig

    def __post_init__(self) -> None:
        if not 0.0 < self.loss_rate <= 1.0:
            raise ValueError(
                f"loss_rate must be in (0, 1], got {self.loss_rate}; at or below zero nothing ever "
                "needs to drink and thirst is inert again, and above 1 an animal loses more than "
                "it holds in a single tick"
            )
        if self.heat_scaling < 0.0:
            raise ValueError(
                f"heat_scaling must be non-negative, got {self.heat_scaling}; negative would make "
                "a hot world hydrating"
            )
        if not 0.0 < self.drink_rate <= 1.0:
            raise ValueError(
                f"drink_rate must be in (0, 1], got {self.drink_rate}; at or below zero water "
                "cannot be drunk and this mechanic does not exist"
            )


class Hydration(DomainService):
    """Owns the `dehydration` column: how much of its water each animal has lost, in [0, 1].

    Zero is fully watered and one is completely dry. See the module docstring for why the deficit
    rather than the reserve is what is stored, and why nothing here kills anybody.
    """

    owns = ("dehydration",)

    # Narrows DomainService.store (typed `object`, the base being store-shape-agnostic) to the
    # concrete EntityStore whose `dehydration` column this service governs.
    store: EntityStore

    def __init__(
        self,
        store: EntityStore,
        registry: ColumnRegistry,
        terrain: Terrain,
        climate: Climate,
        water: Water,
        config: HydrationConfig,
        barriers: Barriers | None = None,
    ) -> None:
        super().__init__(store, registry)
        self.terrain = terrain
        self.climate = climate
        self.water = water
        self.config = config
        # `(h, w) float32, world units`: how much standing water is *reachable* from each cell.
        # Built once rather than per tick, because `Water` is derived from terrain and never
        # advanced. When water becomes dynamic (#165) this becomes a registered system beside
        # `Plants.rebuild_forage`, and its range is the knob that decides how far a herd will walk
        # for a drink.
        self.reachable = CostAwareDiffusion(terrain, config.reachability, barriers).spread(
            water.depth.astype(np.float32)
        )

    def deficit(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32, dimensionless in [0, 1]: how dry each animal is."""
        return self.store.dehydration[selection.to_mask()]

    def lose(self, selection: Selection) -> None:
        """Charge one tick of water loss, scaled by the heat at each animal's own position.

        `selection` is the caller's choice of who dries out; pass the living. Nothing here filters,
        for the same reason `Ecology.drain` does not — a tick loop drying anything other than the
        living is a bug in the loop rather than a condition to absorb quietly (§8.7).

        Saturates at 1 rather than growing without bound. A deficit is a fraction of a reserve and
        an animal cannot lose more water than it has; letting it run past 1 would make the upkeep
        multiplier unbounded, so a long-dry animal would be charged arbitrarily rather than merely
        fatally. The cap is what keeps `dehydration` in the [0, 1] the invariant harness asserts.
        """
        mask = selection.to_mask()
        temperature = self.climate.temperature_at(self.store.x[mask], self.store.y[mask])
        # Floored at zero: cold air does not hydrate anybody, it merely fails to dry them out.
        heat = np.maximum(temperature - self.config.neutral_temperature, 0.0)
        lost = self.config.loss_rate * (1.0 + self.config.heat_scaling * heat)
        self.write("dehydration", selection, np.minimum(self.deficit(selection) + lost, 1.0))

    def drink(self, selection: Selection) -> None:
        """Restore one tick's worth of water to every animal standing at drinkable water.

        Standing *at* water is `Water.is_drinkable_at`, which is where the salinity question will
        land if one ever arrives — this asks "can I drink here", never "how deep is it".

        Costs no energy, for the reason §2.5 gives about resting: what a drink cost was paid walking
        here, and charging for the act would make water a third way to starve rather than the escape
        from thirst it exists to be.
        """
        mask = selection.to_mask()
        at_water = self.water.is_drinkable_at(self.store.x[mask], self.store.y[mask])
        drunk = np.where(at_water, self.config.drink_rate, 0.0)
        self.write("dehydration", selection, np.maximum(self.deficit(selection) - drunk, 0.0))

    def reachable_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The reachable-water field sampled at arbitrary points, in the field's own units.

        Accepts any shape, because #114 samples a whole `(n_entities, n_options)` block of
        candidate headings at once — mirroring `Plants.forage_at`, which is the field this one is
        the twin of.
        """
        rows, cols = self.terrain.cell_indices(x, y)
        return self.reachable[rows, cols]
