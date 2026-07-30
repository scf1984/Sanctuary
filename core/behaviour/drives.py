"""The authored drives: hunger, thirst, fear, lust and fatigue (issue #22).

Each is a vectorized function over the global arrays producing one score column, registered against
`core.behaviour.service.Behaviour` rather than dispatched from it. They share no base class and no
common constructor — a drive takes exactly the collaborators it reads, bound once at construction,
which is what lets a new drive be added without touching anything here.

**What a drive may read is deliberately narrow.** Each of these scores against state that exists
today, so none of them is a placeholder waiting to become real (§8.2). Where the ecology a drive
will eventually read has not been built, the docstring names the issue that will supply it and the
term the drive gains then. One is left, and it is worth stating up front because the name promises
more than the current formulation delivers:

- **Thirst has no hydration reservoir.** No drinking mechanic is filed, so there is no pool to
  deplete and refill. What exists is the climate field, and heat load is the honest reading of it:
  an animal in hot ground wants water. When a hydration column arrives this becomes the ambient
  half of a two-term score rather than the whole of it.

Fatigue was the other, scoring health deficit alone while its name promised exertion. #107 gave it
the column that was missing (`core.behaviour.exertion`), so it now reads both, and that is the
shape thirst takes when its own missing half arrives.

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

from core.behaviour.exertion import Exertion
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
    satiation_energy: energy units. The pool level at or above which an animal wants no food at all.
        Hunger rises linearly as the pool falls below it, reaching `weight` at an empty pool, so
        this is also what decides how early in a decline feeding starts outranking everything else.
    detection_threshold: the forage-field reading, scaled by expressed sight, below which an
        animal notices nothing. **This is what gives the sight gene teeth** (#93): the forage field
        is one field per world and cannot carry a per-animal radius, so acuity enters as a threshold
        on what is sampled rather than as a range — exactly the rule §2.5 already settles for scent,
        and for the same reason. Without it every animal would detect every meadow faintly, sight
        range would be charged by the metabolic budget while buying nothing but predator avoidance,
        and half the selection pressure on the trait would vanish. Must be positive; at zero the
        faintest trace anywhere counts as food found.
    sight_gene: the gene whose expressed value scales what an animal can detect. Named here rather
        than assumed, exactly as `MetabolismConfig.insulation_gene` is, because the vocabulary is
        per-world.

    How far an animal will walk for food is no longer set here: it is the range of the plant
    field's own diffusion (`PlantsConfig.forage_diffusion`), because the distance discount and the
    spreading are one mechanism (#93).
    """

    weight: float
    satiation_energy: float
    detection_threshold: float
    sight_gene: str

    def __post_init__(self) -> None:
        _check_weight(self.weight)
        if self.satiation_energy <= 0:
            raise ValueError(f"satiation_energy must be positive, got {self.satiation_energy}")
        if self.detection_threshold <= 0:
            raise ValueError(
                f"detection_threshold must be positive, got {self.detection_threshold}; see the "
                "config docstring — at zero the faintest trace anywhere counts as food found"
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

        A heading read off the plant field's gradient (#93), turned into a point far enough away
        that `Movement.step` spends the whole tick walking toward it. The field has already applied
        the distance discount and the cost of the ground between, so there is nothing left here to
        rank: the direction the reading rises fastest *is* the answer to "toward what".

        This replaces an argmax over candidate patches scored by `biomass / (1 + distance /
        forage_reluctance)`. That rule could only discount distance, so a meadow across a gorge
        scored as well as one on open ground; the gradient discounts the walk itself.

        **Detection is a threshold, not a radius.** One field serves the whole population, so the
        sight phenotype cannot narrow what is sampled — it scales what is sampled, and an animal
        whose scaled reading falls below `detection_threshold` has found nothing (§2.5 settles the
        identical rule for scent). Such an animal is returned its own position: there is nothing
        worth walking to, and inventing a destination would send it marching at a cell chosen by
        whichever way the numerical noise happened to lean.
        """
        mask = selection.to_mask()
        x = self.store.x[mask].astype(np.float64)
        y = self.store.y[mask].astype(np.float64)
        sight = self.genetics.expressed(selection)[:, self._sight_index].astype(np.float64)

        field = self.plants.forage_field()
        gradient_x, gradient_y, strength = self.plants.forage_gradient(field, x, y)

        slope = np.hypot(gradient_x, gradient_y)
        found = (sight * strength >= self.config.detection_threshold) & (slope > 0.0)
        # A unit heading, then one field-range's worth of travel along it. Any distance at least a
        # tick's reach would move the animal identically — `Movement.step` stops at its own reach —
        # and the range is the honest choice: past it the field carries nothing, so it is the
        # furthest point this reading can actually vouch for.
        pace_out = self.plants.config.forage_diffusion.range
        scale = np.where(found, pace_out / np.where(slope > 0.0, slope, 1.0), 0.0)
        # Clamped into the world, because a heading is a direction and a direction near the edge
        # points out of it. `Movement.step` consumes targets without bounding them — deliberately,
        # since a step never overshoots one — so an out-of-world destination raises out of the
        # middle of a tick from `Terrain.elevation_at`. The contract this replaced returned real
        # in-world cell centres and could not express the problem; a heading can, so the drive that
        # invents it is where it stops.
        terrain = self.plants.terrain
        return (
            np.clip(x + gradient_x * scale, 0.0, terrain.world_width),
            np.clip(y + gradient_y * scale, 0.0, terrain.world_height),
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
    breeding_energy: energy units. The pool level below which reproduction is not attempted at all.
        Gestation charges upkeep like any other trait (§2.5), so an animal that cannot afford it
        must not want it, or selection would favour breeding itself to death.
    abundant_energy: energy units at which lust saturates. Must exceed `breeding_energy`.
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
    aversion_genes: one gene block per **aversion direction**, each naming its genes in cue-channel
        order. Every block must be exactly as long as the cue field has channels — a mismatch would
        silently weight channel *k* by aversion *k+1*, which no test would catch by accident.

        Two directions rather than one, because a single direction can only point at one region of
        cue space. A creature that must fear both wolves and eagles — two unrelated signatures —
        would need one direction pointing between them, which then also fires at everything
        *in* between, harmless things included. A second direction lets the two be feared
        independently (CLAUDE.md §2.5).
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
    aversion_genes: tuple[tuple[str, ...], ...]
    detection_threshold: float
    saturation: float

    def __post_init__(self) -> None:
        _check_weight(self.weight)
        if not self.aversion_genes:
            raise ValueError("aversion_genes must name at least one aversion direction")
        if any(not block for block in self.aversion_genes):
            raise ValueError("every aversion direction must name at least one gene")
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

    Fear is a **noisy-OR over perception channels** (CLAUDE.md §2.5), where a channel is one
    *(aversion direction × sense)* pair reporting a detection probability in [0, 1]; fear is
    ``weight × (1 − Π(1 − p))``. Today that is two directions over one sense — smell — and #24
    adds sight by widening the same product, not by changing its shape.

    A creature carries more than one aversion direction because a single one can only point at one
    region of cue space. Pointing it between two unrelated threats would fire at everything in
    between, harmless creatures included; two directions fear the two independently.

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
        mismatched = [
            block for block in config.aversion_genes if len(block) != scent.field.n_channels
        ]
        if mismatched:
            raise ValueError(
                f"every aversion direction must name {scent.field.n_channels} genes, one per cue "
                f"channel; {len(mismatched)} of {len(config.aversion_genes)} do not"
            )
        self.store = store
        self.genetics = genetics
        self.scent = scent
        self.config = config
        # Raise KeyError naming the vocabulary version if any gene does not exist.
        self._acuity_index = vocabulary.index_of(config.scent_acuity_gene)
        # (n_directions, n_channels): one row of gene columns per aversion direction, so scoring
        # every direction is one matrix product rather than a loop over genes.
        self._aversion_indices = np.array(
            [[vocabulary.index_of(name) for name in block] for block in config.aversion_genes],
            dtype=np.int64,
        )

    def score(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: weighted probability that something dangerous was detected."""
        detected = np.stack(self._channels(selection))
        # Noisy-OR: independent detections corroborate without ever summing past certainty, which
        # is what lets #24 add a sense without re-tuning every other drive's weight.
        return (self.config.weight * (1.0 - np.prod(1.0 - detected, axis=0))).astype(np.float32)

    def _channels(self, selection: Selection) -> list[np.ndarray]:
        """Each channel's detection probability, (len(selection),) float32 in [0, 1].

        One channel per aversion direction, all reading the same nose. #24 appends the same
        directions read through sight instead. Nothing outside a channel knows how its probability
        was computed, so that addition changes no other line in this class.
        """
        expressed = self.genetics.expressed(selection)
        # Sampled once and shared: the field read is the expensive part, and every direction
        # weighs the same air differently.
        smelled = self.scent.perceive(selection)
        acuity = expressed[:, self._acuity_index]
        return [
            self._detect(acuity * (expressed[:, direction] * smelled).sum(axis=1))
            for direction in self._aversion_indices
        ]

    def _detect(self, danger: np.ndarray) -> np.ndarray:
        """Turn perceived danger into a detection probability: nothing below the threshold,
        certainty at saturation, linear between."""
        span = self.config.saturation - self.config.detection_threshold
        return np.clip((danger - self.config.detection_threshold) / span, 0.0, 1.0).astype(
            np.float32
        )


@dataclass(frozen=True)
class FatigueConfig:
    """Per-world tuning for the fatigue drive.

    weight: multiplier on the drive's 0-1 shape.
    exertion_saturation: accumulated work per unit of body size at which exhaustion alone is
        reason enough to stop — the point where `Exertion`'s open-ended quantity becomes the
        0-to-1 shape every drive competes in. Must be positive; at zero any movement at all would
        pin fatigue at maximum and the drive would stop discriminating between a stroll and a
        chase.

        It is not derived from `MovementConfig`, even though what fills the column comes from
        there, because the two answer different questions: that config says what a step *costs*,
        this one says how much work is *too much*, and an animal bred for endurance is a world
        where the second moved and the first did not.
    """

    weight: float
    exertion_saturation: float

    def __post_init__(self) -> None:
        _check_weight(self.weight)
        if self.exertion_saturation <= 0:
            raise ValueError(
                f"exertion_saturation must be positive, got {self.exertion_saturation}; "
                "at zero any movement whatsoever pins fatigue at maximum"
            )


class Fatigue:
    """Wanting to rest, from injury or from recent effort.

    Two independent reasons to do one thing, so they compose the way CLAUDE.md §2.5 settled for
    fear's perception channels — **noisy-OR**, not a sum or a maximum. An animal that is both hurt
    and spent is more inclined to stop than one that is only either, the score stays inside [0, 1]
    like every other drive, and a third reason to rest could be added later without inflating this
    one past saturation and forcing every other drive's weight to be retuned. Reusing that
    composition rather than inventing a second one is the point: the repository should not have two
    answers to "how do independent urgencies combine".

    Exertion is the term that was missing until #107. Health deficit alone made a creature that had
    sprinted across a ridge indistinguishable from one that stood still all tick, so resting was
    selected for only as recovery from injury — see `core.behaviour.exertion` for what the column
    records and why it is not the energy pool.
    """

    name = "fatigue"

    def __init__(self, store: EntityStore, exertion: Exertion, config: FatigueConfig) -> None:
        self.store = store
        self.exertion = exertion
        self.config = config

    def score(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: weighted urgency to rest, zero for a healthy idle animal."""
        injury = np.clip(1.0 - self.store.health[selection.to_mask()], 0.0, 1.0)
        spent = np.clip(
            self.exertion.exerted(selection) / self.config.exertion_saturation, 0.0, 1.0
        )
        # Noisy-OR over the two reasons: neither alone can saturate the score, and both together
        # exceed either — an injured animal that has also just been running wants to stop most.
        return (self.config.weight * (1.0 - (1.0 - injury) * (1.0 - spent))).astype(np.float32)
