"""The authored drives: hunger, thirst, fear, lust and fatigue (issue #22).

Each is a vectorized function over the global arrays producing one score column, registered against
`core.behaviour.service.Behaviour` rather than dispatched from it. They share no base class and no
common constructor — a drive takes exactly the collaborators it reads, bound once at construction,
which is what lets a new drive be added without touching anything here.

**What a drive may read is deliberately narrow.** Each of these scores against state that exists
today, so none of them is a placeholder waiting to become real (§8.2). Where the ecology a drive
will eventually read has not been built, the docstring names the issue that will supply it and the
term the drive gains then. Two of those are worth stating up front, because the names promise more
than the current formulation delivers:

- **Thirst has no hydration reservoir.** No drinking mechanic is filed, so there is no pool to
  deplete and refill. What exists is the climate field, and heat load is the honest reading of it:
  an animal in hot ground wants water. When a hydration column arrives this becomes the ambient
  half of a two-term score rather than the whole of it.
- **Fatigue is not exertion.** Movement lands in #25, so nothing yet spends effort. Health deficit
  is what exists, and recovery is a real reason to rest.

Fear is the exception to that pattern: its shape is settled in CLAUDE.md §2.5 rather than improvised
here, precisely so that #24 adding sight and #97 adding wind are additions rather than rewrites.

**Weights are config now and genes at #23.** Every config carries a `weight`, which is what makes
five differently-shaped urgencies comparable at all — without it, "hunger 0.6 vs fear 0.6" would be
a coincidence of formula rather than a decision. #23 replaces the scalar with a per-entity gene
column, at which point boldness and sociality are inherited and selected rather than tuned
(CLAUDE.md §2.5); the shape of every score below is unchanged by that substitution.

Coefficients are per-world configuration, never constants in this module (CLAUDE.md §2.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.ecology.cues import Scent
from core.ecology.plants import Plants
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.genetics.vocabulary import GeneVocabulary
from core.selection import Selection
from core.world.climate import Climate


def _check_weight(weight: float) -> None:
    """A negative weight would invert a drive: the more urgent its cause, the less it wants."""
    if weight < 0:
        raise ValueError(f"drive weight must be non-negative, got {weight}")


@dataclass(frozen=True)
class HungerConfig:
    """Per-world tuning for the hunger drive.

    weight: multiplier on the drive's 0-1 shape, comparable against every other drive's weight.
    satiation_energy: joules. The pool level at or above which an animal wants no food at all.
        Hunger rises linearly as the pool falls below it, reaching `weight` at an empty pool, so
        this is also what decides how early in a decline feeding starts outranking everything else.
    forage_reluctance: world units. How far an animal will walk for food, in the distance discount
        CLAUDE.md §2.5 settles: small values keep grazers local and strip ground bare before they
        move on, large ones spread grazing pressure out. Must be positive — zero would divide by
        the distance alone and make the nearest non-empty cell infinitely preferable.
    sight_gene: the gene whose expressed value is the forage perception radius, in world units.
        Named here rather than assumed, exactly as `MetabolismConfig.insulation_gene` is, because
        the vocabulary is per-world.
    """

    weight: float
    satiation_energy: float
    forage_reluctance: float
    sight_gene: str

    def __post_init__(self) -> None:
        _check_weight(self.weight)
        if self.satiation_energy <= 0:
            raise ValueError(f"satiation_energy must be positive, got {self.satiation_energy}")
        if self.forage_reluctance <= 0:
            raise ValueError(
                f"forage_reluctance must be positive, got {self.forage_reluctance}; see the "
                "config docstring — zero makes the nearest crumb beat any distant meadow"
            )


class Hunger:
    """Wanting food, and knowing which patch of it is worth the walk.

    Score is the energy deficit against satiation — a property of the animal, not of the ground it
    is standing on. Whether there is anything to eat nearby belongs to `forage_target`, and keeping
    the two apart is what lets a starving animal in barren ground still read as starving rather
    than as content.
    """

    name = "hunger"

    def __init__(
        self,
        store: EntityStore,
        ecology: Ecology,
        genetics: Genetics,
        plants: Plants,
        vocabulary: GeneVocabulary,
        config: HungerConfig,
    ) -> None:
        self.store = store
        self.ecology = ecology
        self.genetics = genetics
        self.plants = plants
        self.config = config
        # Raises KeyError naming the vocabulary version if the gene does not exist.
        self._sight_index = vocabulary.index_of(config.sight_gene)

    def score(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: how far the pool has fallen below satiation, weighted."""
        deficit = 1.0 - self.ecology.energy(selection) / self.config.satiation_energy
        return (self.config.weight * np.clip(deficit, 0.0, 1.0)).astype(np.float32)

    def forage_target(self, selection: Selection) -> tuple[np.ndarray, np.ndarray]:
        """Where each forager should go to eat: (x, y), each (len(selection),) float64.

        This is the rule CLAUDE.md §2.5 settles and assigns to this drive: over every patch
        `Plants.perceive` reports (#93), take the argmax of `biomass / (1 + distance /
        forage_reluctance)`. Distance-discounted rather than raw, because a grazer that crosses its
        whole sight range for a marginally richer cell neither feeds efficiently nor produces the
        local grazing pressure the field model exists to express.

        Perception is gated by the expressed sight gene, so an animal only finds what it could pay
        to see — the field itself knows nothing of genes, which is why the radius is computed here.

        An animal that can see no food anywhere is returned its own position: there is no patch
        worth walking to, and inventing one would send it marching toward an empty cell chosen by
        whatever argmax broke the all-zero tie.
        """
        mask = selection.to_mask()
        x = self.store.x[mask].astype(np.float64)
        y = self.store.y[mask].astype(np.float64)
        radius = self.genetics.expressed(selection)[:, self._sight_index].astype(np.float64)

        patch_x, patch_y, biomass = self.plants.perceive(x, y, radius)
        distance = np.hypot(patch_x - x[:, None], patch_y - y[:, None])
        utility = biomass / (1.0 + distance / self.config.forage_reluctance)

        foragers = np.arange(len(selection))
        best = np.argmax(utility, axis=1)
        worth_walking = utility[foragers, best] > 0.0
        return (
            np.where(worth_walking, patch_x[foragers, best], x),
            np.where(worth_walking, patch_y[foragers, best], y),
        )


@dataclass(frozen=True)
class ThirstConfig:
    """Per-world tuning for the thirst drive.

    weight: multiplier on the drive's 0-1 shape.
    onset_temperature: degrees C below which an animal wants no water at all.
    saturation_temperature: degrees C at and above which thirst is maximal. Must exceed
        `onset_temperature`; the two being equal would make thirst a step function of the climate
        field and every animal on one side of a contour maximally thirsty at once.
    """

    weight: float
    onset_temperature: float
    saturation_temperature: float

    def __post_init__(self) -> None:
        _check_weight(self.weight)
        if self.saturation_temperature <= self.onset_temperature:
            raise ValueError(
                "saturation_temperature must exceed onset_temperature, got "
                f"{self.saturation_temperature} <= {self.onset_temperature}"
            )


class Thirst:
    """Wanting water, read from ambient heat at the animal's own position.

    See the module docstring: there is no hydration pool to deplete, because nothing drinks yet.
    Heat load is what the world can currently answer, and it produces the behaviour that matters
    ecologically — hot ground pushes animals toward water and therefore toward each other, which
    is where the competition for it will happen once drinking exists.
    """

    name = "thirst"

    def __init__(self, store: EntityStore, climate: Climate, config: ThirstConfig) -> None:
        self.store = store
        self.climate = climate
        self.config = config

    def score(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: weighted heat load, zero below the onset temperature."""
        mask = selection.to_mask()
        temperature = self.climate.temperature_at(self.store.x[mask], self.store.y[mask])
        span = self.config.saturation_temperature - self.config.onset_temperature
        load = (temperature - self.config.onset_temperature) / span
        return (self.config.weight * np.clip(load, 0.0, 1.0)).astype(np.float32)


@dataclass(frozen=True)
class LustConfig:
    """Per-world tuning for the lust drive.

    weight: multiplier on the drive's 0-1 shape.
    maturity_age: ticks. Below this an animal does not seek a mate at any energy level — the tick
        counter is the only clock (CLAUDE.md §2.1), so this is in ticks and never in real time.
    breeding_energy: joules. The pool level below which reproduction is not attempted at all.
        Gestation charges upkeep like any other trait (§2.5), so an animal that cannot afford it
        must not want it, or selection would favour breeding itself to death.
    abundant_energy: joules at which lust saturates. Must exceed `breeding_energy`.
    """

    weight: float
    maturity_age: int
    breeding_energy: float
    abundant_energy: float

    def __post_init__(self) -> None:
        _check_weight(self.weight)
        if self.maturity_age < 0:
            raise ValueError(f"maturity_age must be non-negative, got {self.maturity_age}")
        if self.abundant_energy <= self.breeding_energy:
            raise ValueError(
                "abundant_energy must exceed breeding_energy, got "
                f"{self.abundant_energy} <= {self.breeding_energy}"
            )


class Lust:
    """Wanting to breed: a maturity gate times whatever energy is spare above breeding cost.

    A gate rather than a ramp on age, because sexual maturity is a threshold in a way hunger is
    not. #20 owns what happens next — mate selection, gestation, and the cost of both; this only
    decides who is looking.
    """

    name = "lust"

    def __init__(self, store: EntityStore, ecology: Ecology, config: LustConfig) -> None:
        self.store = store
        self.ecology = ecology
        self.config = config

    def score(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: weighted energy surplus, zero before maturity."""
        mature = self.store.age[selection.to_mask()] >= self.config.maturity_age
        headroom = self.config.abundant_energy - self.config.breeding_energy
        surplus = (self.ecology.energy(selection) - self.config.breeding_energy) / headroom
        return (self.config.weight * np.where(mature, np.clip(surplus, 0.0, 1.0), 0.0)).astype(
            np.float32
        )


@dataclass(frozen=True)
class FearConfig:
    """Per-world tuning for the fear drive.

    weight: multiplier on the drive's 0-1 shape.
    scent_acuity_gene: the gene whose expressed value is scent detection sensitivity. Named here
        rather than assumed, as `MetabolismConfig.insulation_gene` is, because the vocabulary is
        per-world.
    aversion_genes: the genes holding this creature's aversion vector over cue space, in channel
        order. Must be exactly as long as the cue field has channels — a mismatch would silently
        weight channel *k* by aversion *k+1*, which is not an error any test would catch by
        accident.
    detection_threshold: perceived danger below which nothing is noticed at all. This is what
        makes acuity buy *range* rather than volume — see the class docstring. Must be positive:
        at zero every creature detects every trace from anywhere, and the gene collapses into a
        panic multiplier.
    saturation: perceived danger at which detection is certain. Must exceed `detection_threshold`;
        equal values would make the channel a step function, so every animal on one side of a
        contour would be maximally terrified at once.
    """

    weight: float
    scent_acuity_gene: str
    aversion_genes: tuple[str, ...]
    detection_threshold: float
    saturation: float

    def __post_init__(self) -> None:
        _check_weight(self.weight)
        if not self.aversion_genes:
            raise ValueError("aversion_genes must name at least one gene")
        if self.detection_threshold <= 0:
            raise ValueError(
                f"detection_threshold must be positive, got {self.detection_threshold}; see the "
                "config docstring — zero turns the scent gene into a panic multiplier"
            )
        if self.saturation <= self.detection_threshold:
            raise ValueError(
                "saturation must exceed detection_threshold, got "
                f"{self.saturation} <= {self.detection_threshold}"
            )


class Fear:
    """Wanting to be somewhere else, because something dangerous is near.

    Fear is a **noisy-OR over perception channels** (CLAUDE.md §2.5). Each channel is one sense
    with its own physics, reporting a detection probability in [0, 1]; fear is
    ``weight × (1 − Π(1 − p))``. Today there is exactly one channel — scent — so the product
    collapses to that single probability, and `_channels` is written as the explicit combination it
    will remain rather than as a special case to be rewritten when #24 registers sight.

    **Nothing here fears a species.** Danger is the dot product of the creature's own *aversion*
    vector with the cue concentration it can smell (`core.ecology.cues`) — both ends genetic, both
    heritable, both under selection. There is no threat table to author and none to extend when the
    world speciates, because there is nothing per-species anywhere in this drive.

    That is what makes the interesting behaviour emergent rather than implemented. Prey evolve
    aversion pointed at whatever signature predators emit; predators evolve signatures that drift
    away from it; a harmless lineage whose signature drifts toward a feared one is avoided for free.
    **Cannibalism is a lineage whose aversion overlaps its own signature** — no diagonal, no special
    case, and it can evolve on and off (CLAUDE.md §2.5).

    **Acuity buys range, not volume.** Concentration falls off monotonically from its source and
    detection is a threshold on concentration, so a keener nose crossing that threshold at a fainter
    trace is detecting the same creature from *further away* — sensitivity and range are the same
    parameter for a plume. This is why acuity may multiply a sampled field value here where a sight
    gene may not: for sight, the same construction would leave a far-seeing animal merely more
    frightened of what it could already see, which is the wrong selection pressure.
    """

    name = "fear"

    def __init__(
        self,
        store: EntityStore,
        genetics: Genetics,
        scent: Scent,
        vocabulary: GeneVocabulary,
        config: FearConfig,
    ) -> None:
        if len(config.aversion_genes) != scent.field.n_channels:
            raise ValueError(
                f"aversion_genes names {len(config.aversion_genes)} genes but the cue field has "
                f"{scent.field.n_channels} channels; they index the same space and must match"
            )
        self.store = store
        self.genetics = genetics
        self.scent = scent
        self.config = config
        # Raise KeyError naming the vocabulary version if any gene does not exist.
        self._acuity_index = vocabulary.index_of(config.scent_acuity_gene)
        self._aversion_indices = np.array(
            [vocabulary.index_of(name) for name in config.aversion_genes], dtype=np.int64
        )

    def score(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: weighted probability that something dangerous was detected."""
        detected = np.stack(self._channels(selection))
        # Noisy-OR: independent senses corroborate without ever summing past certainty, which is
        # what lets #24 add a channel without re-tuning every other drive's weight.
        return (self.config.weight * (1.0 - np.prod(1.0 - detected, axis=0))).astype(np.float32)

    def _channels(self, selection: Selection) -> list[np.ndarray]:
        """Each perception channel's detection probability, (len(selection),) float32 in [0, 1].

        #24 appends the sight channel here. Nothing outside a channel knows how its probability
        was computed, so that addition changes no other line in this class.
        """
        return [self._scent(selection)]

    def _scent(self, selection: Selection) -> np.ndarray:
        """Detection probability from smell alone."""
        expressed = self.genetics.expressed(selection)
        # How much of each cue channel is in the air here, against how much this particular
        # creature minds that channel — one vectorized pass, and the only place danger is defined.
        smelled = self.scent.perceive(selection)
        danger = (expressed[:, self._aversion_indices] * smelled).sum(axis=1)
        perceived = expressed[:, self._acuity_index] * danger
        span = self.config.saturation - self.config.detection_threshold
        return np.clip((perceived - self.config.detection_threshold) / span, 0.0, 1.0).astype(
            np.float32
        )


@dataclass(frozen=True)
class FatigueConfig:
    """Per-world tuning for the fatigue drive.

    weight: multiplier on the drive's 0-1 shape. The only parameter, because health is already a
        0-to-1 fraction on the store and needs no scale of its own.
    """

    weight: float

    def __post_init__(self) -> None:
        _check_weight(self.weight)


class Fatigue:
    """Wanting to rest, scored on health deficit.

    See the module docstring: exertion arrives with movement (#25), and until something spends
    effort there is nothing to be tired from. Recovery is the part that exists — an injured animal
    has a real reason to stop — and it competes against hunger and fear exactly as exertion will.
    """

    name = "fatigue"

    def __init__(self, store: EntityStore, config: FatigueConfig) -> None:
        self.store = store
        self.config = config

    def score(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: weighted health deficit, zero at full health."""
        deficit = 1.0 - self.store.health[selection.to_mask()]
        return (self.config.weight * np.clip(deficit, 0.0, 1.0)).astype(np.float32)
