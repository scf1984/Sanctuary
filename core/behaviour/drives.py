"""The authored drives: hunger, thirst, fear, lust, fatigue (issue #22) and thermal (#249).

Each is a vectorized function over the global arrays producing one score column, registered against
`core.behaviour.service.Behaviour` rather than dispatched from it. They share no base class and no
common constructor — a drive takes exactly the collaborators it reads, bound once at construction,
which is what lets a new drive be added without touching anything here.

**What a drive may read is deliberately narrow.** Each of these scores against state that exists
today, so none of them is a placeholder waiting to become real (§8.2). Where the ecology a drive
will eventually read has not been built, the docstring names the issue that will supply it and the
term the drive gains then.

Two drives used to promise more than they read. Thirst scored ambient heat as its whole urgency,
with no reservoir to deplete. #156 gave it one: urgency is now the hydration deficit, and heat
reaches it only physically, by speeding water loss in `Hydration`. **Thirst reads no temperature**,
and `TestThirst.test_thirst_reads_no_temperature` pins that (#254). It matters because §2.5
forbids two drives scoring one signal, and temperature belongs to #249's thermal drive. Fatigue
scored health deficit alone while its name promised exertion. #107 gave it the missing column
(`core.behaviour.exertion`), so it now reads both.

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

from core.behaviour.candidates import Candidates
from core.behaviour.exertion import Exertion
from core.ecology.carrion import Carrion
from core.ecology.cues import Scent
from core.ecology.diet import Diet
from core.ecology.hydration import Hydration
from core.ecology.metabolism import Metabolism
from core.ecology.plants import Plants
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.genetics.registry import GeneRegistry, Unit
from core.selection import Selection
from core.world.climate import Climate
from core.world.terrain import Terrain


class DriveWeight:
    """One drive's weight, read per entity from a gene (#23, CLAUDE.md §2.5).

    §2.5 makes the weights genes rather than constants, so **boldness, sociality and parental
    investment arise from selection instead of being designed**. A population under heavy predation
    evolves caution with nobody writing a caution rule; the same population, unpredated, evolves it
    away again, because a frightened animal forgoes food for nothing.

    Every drive resolves its weight identically, which is why this exists at all — it is the fifth
    repetition of the same three lines (§8.3), not an abstraction written ahead of need.

    **Read as a magnitude, which is what deletes the old check.** `_check_weight` used to reject a
    negative weight at construction, because a negative one inverts a drive: the more urgent its
    cause, the less it wants. A magnitude gene cannot express that, so the check is gone rather than
    moved — §8.7's preference for an unrepresentable failure over a validated one, the same move
    #111 made for gene costs.
    """

    def __init__(self, genetics: Genetics, genes: GeneRegistry, gene: str) -> None:
        self.genetics = genetics
        # A weight scales a drive's 0-1 shape, so it is a bare multiplier and carries no dimension.
        self._column = np.array([genes.index_of(gene, unit=Unit.DIMENSIONLESS)], dtype=np.int64)

    def of(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: each entity's own weighting for this drive."""
        return self.genetics.expressed_columns(selection, self._column)[:, 0]


def detected(reading: np.ndarray, threshold: float) -> np.ndarray:
    """(n, n_options) float64 in [0, 1]: what a drive perceives of a field it has sampled (#253).

    `reading` is the caller's sample of its field at each candidate, already scaled by whatever
    acuity gene sharpens that sense. Readings below `threshold` are not perceived at all. What
    survives is scaled to the animal's own best option.

    **What it is for.** Every drive that steers by a continuous field (hunger's forage and prey,
    thirst's water, lust's company, and #249's temperature next) perceives through this one rule,
    so their appeals stay on one scale and comparable once the utilities are summed. The rule
    itself is §2.5's: for a field that serves the whole population, sensitivity and range are one
    parameter, so acuity can only scale what is sampled and a threshold is what gives it teeth.

    **What it forbids.**
    - **Steering by a raw reading.** Without the gate every animal detects every source in the
      world faintly, and an acuity gene that is charged for buys nothing.
    - **A flat non-zero appeal when nothing is perceived.** The row comes back all zeros, so the
      drive drops out of the sum and the null option can win, instead of the animal marching
      whichever way numerical noise leaned.
    - **A negative appeal.** A reading worse than nothing (a smell opposite to your own, ground
      worse than where you stand) is discarded by the positive threshold, never ranked.

    Sampling and acuity stay with the caller on purpose. Each field is read differently (a
    diffused grid, a self-excluded cue blend dotted against a signature), and thirst has no acuity
    gene at all. A parameter only one caller needed would be §8.2's speculative generality.
    `Fear` applies the same gate but not this function (#265): its appeal is *aversion*, scaled so
    the worst option scores 0 rather than the best scoring 1, and it sums the gated readings of
    several aversion directions before scaling. It keeps this rule's other half, all zeros when
    nothing is detected.
    """
    found = np.where(reading >= threshold, reading, 0.0)
    best = found.max(axis=1, keepdims=True)
    return np.divide(found, best, out=np.zeros_like(found), where=best > 0.0)


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
        than assumed, exactly as `MetabolismConfig.heat_tolerance_gene` is, because the vocabulary is
        per-world.

    How far an animal will walk for food is no longer set here: it is the range of the plant
    field's own diffusion (`PlantsConfig.forage_diffusion`), because the distance discount and the
    spreading are one mechanism (#93).
    """

    weight_gene: str
    satiation_energy: float
    detection_threshold: float
    sight_gene: str

    def __post_init__(self) -> None:
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
    is standing on. Whether there is anything to eat nearby belongs to `appeal`, and keeping
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
        carrion: Carrion,
        scent: Scent,
        diet: Diet,
        genes: GeneRegistry,
        config: HungerConfig,
    ) -> None:
        self.store = store
        self.ecology = ecology
        self.genetics = genetics
        self.plants = plants
        self.carrion = carrion
        self.scent = scent
        self.diet = diet
        self.config = config
        # Raises KeyError naming the vocabulary version if the gene does not exist. Acuity
        # scales what is sampled from a field rather than a radius (#93), so it is dimensionless.
        # (2,) int: [sight, the diet's flesh allocation], the two genes `appeal` reads.
        self._columns = np.array(
            [genes.index_of(config.sight_gene, unit=Unit.DIMENSIONLESS), diet.animal_derived_index],
            dtype=np.int64,
        )
        self._weight = DriveWeight(genetics, genes, config.weight_gene)

    def urgency(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: how far the pool has fallen below satiation, weighted."""
        deficit = 1.0 - self.ecology.energy(selection) / self.config.satiation_energy
        return (self._weight.of(selection) * np.clip(deficit, 0.0, 1.0)).astype(np.float32)

    def appeal(self, selection: Selection, candidates: Candidates) -> np.ndarray:
        """(n, n_options) float32 in [0, 1]: how much *food* each option leads toward.

        **Food is whatever this animal's diet is allocated toward**, so one drive points a herbivore
        at grass and a carnivore at meat, and nothing had to decide which of the two an animal is
        (#179):

            appeal = (1 − animal_share) × grazing_appeal + animal_share × prey_appeal

        The blend is the diet allocation itself and not a second gene, because "what am I looking
        for" and "what can I digest" are the same question asked before and after the meal. An
        omnivore genuinely splits its attention, which is the behavioural half of #102's frontier:
        being mediocre at both foods now also means steering toward both at once.

        **Grazing** is the diffused forage field read at each candidate, scaled by the sight
        phenotype and gated by `detection_threshold`. **Detection is a threshold, not a radius**:
        one field serves the whole population, so sight cannot narrow what is sampled, only scale
        it (§2.5, #93).

        **Prey** is *meat in that direction*, of either kind: the cue field summed over its
        channels — how much live animal lies that way, with no opinion about which — plus the
        diffused carrion field, which is how a scavenger finds a body it is not already standing on.
        Summing them rather than ranking them is what makes hunting and scavenging one appetite: a
        lineage does not choose between the two strategies, it is drawn to whichever is nearer, and
        which one it ends up living on is an outcome of where it happens to be. That answers #179's Decision 1 with the option that
        needs no new gene, and the cost is that a predator cannot specialise on one prey's
        signature: smell is blunt here exactly as §2.5 requires it to be for fear, and for the same
        reason — the air holds a blend, and only a linear readout composes correctly on a blend.
        A per-lineage appetite *direction* in cue space is the richer answer and is additive to the
        vocabulary rather than a replacement, so it stays open (recorded on #179).

        Each half is normalised by the animal's own best option before blending, which does two
        things at once. It puts appeal on the same [0, 1] scale every other drive uses, so summed
        utilities are comparable; and it makes an animal that detects nothing return **all zeros**
        rather than a flat non-zero preference — so hunger drops out of the sum entirely and the
        null option can win. An animal with no food in sight rests instead of marching whichever way
        the numerical noise leaned, which is the failure `forage_target` had to special-case.
        """
        phenotype = self.genetics.expressed_columns(selection, self._columns)
        sight = phenotype[:, 0].astype(np.float64)
        x, y = candidates.x, candidates.y
        grazing = sight[:, None] * self.plants.forage_at(self.plants.forage, x, y)
        # Summed over cue channels: how much animal lies that way, with no opinion about which.
        # Signed channels can cancel, and a cancelled blend is *less* smell rather than a negative
        # amount of it, so the sum is floored rather than allowed to invert the appeal.
        prey = sight[:, None] * (
            np.maximum(candidates.perceived(self.scent).sum(axis=-1), 0.0)
            + self.plants.forage_at(self.carrion.scent, x, y)
        )

        # The allocation is the expressed column itself, which is all `Diet.animal_share` reads.
        share = phenotype[:, 1].astype(np.float64)[:, None]
        threshold = self.config.detection_threshold
        return (
            (1.0 - share) * detected(grazing, threshold) + share * detected(prey, threshold)
        ).astype(np.float32)


@dataclass(frozen=True)
class ThirstConfig:
    """Per-world tuning for the thirst drive.

    weight_gene: the gene whose expressed value multiplies this drive's 0-1 shape. A gene rather
        than a constant, so temperament evolves rather than being designed (§2.5, #23). Read as a
        magnitude: a negative weight would invert the drive.
    detection_threshold: the reachable-water reading below which an animal notices no water at all.
        The identical rule hunger applies to forage and fear applies to scent (§2.5, #93):
        sensitivity and range are the same parameter for a diffused field, so a threshold is what
        stops every animal in the world detecting every lake faintly. Must be positive; at zero the
        faintest trace anywhere counts as water found.

    The temperature terms that used to live here are gone. Heat does not make an animal *want*
    water, it makes an animal *lose* water, so `onset_temperature` and `saturation_temperature`
    moved to `HydrationConfig` as the scaling on the loss rate (#156). Wanting then follows from
    having lost, which is what lets one number — the deficit — be both the urgency and the thing
    drinking removes.
    """

    weight_gene: str
    detection_threshold: float

    def __post_init__(self) -> None:
        if self.detection_threshold <= 0:
            raise ValueError(
                f"detection_threshold must be positive, got {self.detection_threshold}; at zero "
                "the faintest trace anywhere counts as water found and the gate buys nothing"
            )


class Thirst:
    """Wanting water, and knowing which way it lies.

    Urgency is the hydration deficit `Hydration` owns — a property of the animal, not of the ground
    it stands on — and appeal is the reachable-water field read at each candidate heading. Keeping
    the two apart is what lets a parched animal in dry country still read as parched rather than as
    content, exactly as hunger separates an empty pool from a bare meadow.

    Both halves were missing until #156: the drive scored ambient temperature and returned a flat
    appeal, so it wanted water it could not find and could not have drunk if it had.
    """

    name = "thirst"

    def __init__(
        self,
        store: EntityStore,
        hydration: Hydration,
        genetics: Genetics,
        genes: GeneRegistry,
        config: ThirstConfig,
    ) -> None:
        self.store = store
        self.hydration = hydration
        self.genetics = genetics
        self.config = config
        self._weight = DriveWeight(genetics, genes, config.weight_gene)

    def urgency(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: how much of its water the animal has lost, weighted.

        Already in [0, 1] because the column is a fraction, so unlike hunger's pool this needs no
        satiation constant to normalise against — how dry you are *is* how much you want to drink.
        """
        return (self._weight.of(selection) * self.hydration.deficit(selection)).astype(np.float32)

    def appeal(self, selection: Selection, candidates: Candidates) -> np.ndarray:
        """(n, n_options) float32 in [0, 1]: how much water each option leads toward.

        The diffused reachable-water field sampled at each candidate and gated by
        `detection_threshold`, then normalised by the animal's own best option — the identical
        shape `Hunger.appeal` uses, and identical deliberately: two drives reading two fields the
        same way is what keeps their contributions comparable once the utilities are summed.

        Normalising by the best option is what makes an animal that detects nothing return **all
        zeros** rather than a flat non-zero preference, so thirst drops out of the sum entirely and
        cannot push an animal whichever way the numerical noise leaned.
        """
        reading = self.hydration.reachable_at(candidates.x, candidates.y).astype(np.float64)
        return detected(reading, self.config.detection_threshold).astype(np.float32)


@dataclass(frozen=True)
class ThermalConfig:
    """Per-world tuning for the thermal drive (#249).

    weight_gene: the gene whose expressed value multiplies this drive's 0-1 shape (§2.5, #23).
    acuity_gene: the gene whose expressed value scales the upkeep saving an animal perceives toward
        each candidate: its thermal sense. A gene rather than a constant so how finely an animal
        reads a gradient is under selection, and it must carry a positive cost in the vocabulary,
        because it only ever helps (§2.5). Named rather than assumed, as every sense gene is.
    detection_threshold: energy units per tick. The acuity-scaled saving below which no candidate
        reads as better at all, the gate `detected` applies (#253). It is what gives the acuity gene
        teeth: without it every animal perceives every gradient however faint, and a charged sense
        buys nothing. Must be positive.
    saturation_upkeep: energy units per tick. The thermoregulation bill at which the drive reaches
        its full weight. Denominated in upkeep rather than degrees, because what the drive is for is
        the energy the climate costs, and a degree costs a cold-tolerant animal less than a tender
        one. Tune it against the world's ordinary upkeep (§2.1): it says how large a share of a
        body's running cost the weather may take before escaping it outranks everything else.
    """

    weight_gene: str
    acuity_gene: str
    detection_threshold: float
    saturation_upkeep: float

    def __post_init__(self) -> None:
        if self.detection_threshold <= 0:
            raise ValueError(
                f"detection_threshold must be positive, got {self.detection_threshold}; at zero "
                "every gradient however faint is perceived and the acuity gene buys nothing"
            )
        if self.saturation_upkeep <= 0:
            raise ValueError(
                f"saturation_upkeep must be positive, got {self.saturation_upkeep}"
            )


class Thermal:
    """Wanting a bearable temperature, and knowing which way it lies (#249).

    **Urgency is the bill.** It is the thermoregulation term `Metabolism` charges this tick, read
    through the animal's own heat and cold tolerance (#176) and scaled against `saturation_upkeep`.
    So the drive wants exactly as much as the climate is costing, and a tolerant animal is driven
    less by the same weather. Both follow from the cost rather than from an authored comfort band.

    **Appeal is the saving.** For each candidate, how much less that bill would be standing there,
    read from the derived temperature field (#159) at the candidate, scaled by the thermal acuity
    gene and perceived through `detected` (#253). Only savings count: a candidate that would cost
    more is discarded by the threshold, never ranked. Neither direction is authored. A cold animal
    goes toward warmth and a hot one toward shade because those are the savings.

    **The only drive that scores temperature** (§2.5: two drives reading one number turn the
    contest into a coin flip). Thirst reads the hydration deficit, and heat reaches it only
    physically, through `Hydration`'s water loss (#156, #254).

    What this cannot do yet: the field is read where the animal can look, one `look_ahead` away, so
    an animal follows the local gradient and has no memory of where it was comfortable. That is
    enough for seasonal migration on a map whose climate straddles neutral, and not on one too hot
    everywhere for any ground to be comfortable (docs/spikes/thermal-drive.md §2 and §5).
    """

    name = "thermal"

    def __init__(
        self,
        store: EntityStore,
        climate: Climate,
        metabolism: Metabolism,
        genetics: Genetics,
        genes: GeneRegistry,
        config: ThermalConfig,
    ) -> None:
        self.store = store
        self.climate = climate
        self.metabolism = metabolism
        self.genetics = genetics
        self.config = config
        # (3,) int: [heat tolerance, cold tolerance, acuity]. Acuity scales a sampled saving
        # rather than a radius, as sight and scent acuity do (#93).
        self._columns = np.append(
            metabolism.thermal_columns, genes.index_of(config.acuity_gene, unit=Unit.DIMENSIONLESS)
        )
        self._weight = DriveWeight(genetics, genes, config.weight_gene)

    def urgency(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: the thermoregulation bill against saturation, weighted."""
        mask = selection.to_mask()
        temperature = self.climate.temperature_at(self.store.x[mask], self.store.y[mask])
        tolerances = self.genetics.expressed_columns(selection, self.metabolism.thermal_columns)
        bill = self.metabolism.thermoregulation(tolerances, temperature)
        shape = np.clip(bill / self.config.saturation_upkeep, 0.0, 1.0)
        return (self._weight.of(selection) * shape).astype(np.float32)

    def appeal(self, selection: Selection, candidates: Candidates) -> np.ndarray:
        """(n, n_options) float32 in [0, 1]: how much of the climate's bill each option would save."""
        mask = selection.to_mask()
        phenotype = self.genetics.expressed_columns(selection, self._columns)
        tolerances = phenotype[:, :2]
        here = self.climate.temperature_at(self.store.x[mask], self.store.y[mask])
        there = self.climate.temperature_at(candidates.x, candidates.y)
        bill_here = self.metabolism.thermoregulation(tolerances, here.astype(np.float64))
        bill_there = self.metabolism.thermoregulation(tolerances, there.astype(np.float64))
        acuity = phenotype[:, 2].astype(np.float64)[:, None]
        saving = acuity * (bill_here[:, None] - bill_there)
        return detected(saving, self.config.detection_threshold).astype(np.float32)


@dataclass(frozen=True)
class LustConfig:
    """Per-world tuning for the lust drive.

    weight_gene: the gene whose expressed value multiplies this drive's 0-1 shape. A gene
        rather than a constant, so temperament evolves rather than being designed (§2.5,
        #23). Read as a magnitude: a negative weight would invert the drive.
    maturity_gene: the gene whose expressed value is how many ticks an animal must live before it
        seeks a mate at all. A **gene** rather than a constant (§2.5, #20): age at first
        reproduction is among the most strongly selected life-history traits there is, and a world
        that fixes it decides by hand what the environment should be deciding. Read as a magnitude,
        so a lineage that drifts below zero matures at once rather than never.
        The tick counter is the only clock (§2.1), so what it expresses is a count of ticks and
        never a span of real time.
    scent_acuity_gene: the gene whose expressed value is scent detection sensitivity. The *same*
        gene fear reads: mate-finding and threat-detection are one nose, so a keener nose finds
        both from further away and there is no separate sense to evolve.
    detection_threshold: the scaled reading below which nobody has been found. Sensitivity and
        range are the same parameter for a plume (§2.5), so this is what gives acuity teeth —
        without it every animal detects every other faintly and the gene only scales enthusiasm.
    breeding_energy: energy units. The pool level below which reproduction is not attempted at all.
        Gestation charges upkeep like any other trait (§2.5), so an animal that cannot afford it
        must not want it, or selection would favour breeding itself to death.
    abundant_energy: energy units at which lust saturates. Must exceed `breeding_energy`.
    """

    weight_gene: str
    maturity_gene: str
    scent_acuity_gene: str
    detection_threshold: float
    breeding_energy: float
    abundant_energy: float

    def __post_init__(self) -> None:
        if self.detection_threshold <= 0:
            raise ValueError(
                f"detection_threshold must be positive, got {self.detection_threshold}; at or "
                "below zero every animal detects every other faintly and acuity buys nothing"
            )
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

    def __init__(
        self,
        store: EntityStore,
        ecology: Ecology,
        genetics: Genetics,
        scent: Scent,
        genes: GeneRegistry,
        config: LustConfig,
    ) -> None:
        self.store = store
        self.ecology = ecology
        self.genetics = genetics
        self.scent = scent
        self.config = config
        # A count of ticks, so dimensionless: §2.1 makes the tick the only clock, and
        # nothing here is a length or an energy.
        self._maturity_column = np.array(
            [genes.index_of(config.maturity_gene, unit=Unit.DIMENSIONLESS)], dtype=np.int64
        )
        self._acuity_column = np.array(
            [genes.index_of(config.scent_acuity_gene, unit=Unit.DIMENSIONLESS)], dtype=np.int64
        )
        self._weight = DriveWeight(genetics, genes, config.weight_gene)

    def appeal(self, selection: Selection, candidates: Candidates) -> np.ndarray:
        """(n, n_options) float32 in [0, 1]: how much company-like-me each option leads toward.

        **The vector is the animal's own signature** (CLAUDE.md §2.5), and that is the whole of
        mate-finding: no genes of its own, no threat table, and it tracks speciation for free —
        a daughter species inherits its parents' signature, so its members are drawn to each other
        without anything being told that a split happened.

        Self-exclusion is not optional here, it is the mechanic. An animal's own plume is the
        strongest match to its own signature anywhere in the world, so reading the raw field would
        make every animal maximally attracted to the cell it is already standing in — a rule that
        says never move. `Scent.perceived_at` subtracts it exactly, at every candidate (#188).

        Scaled by scent acuity and gated by `detection_threshold`, exactly as hunger reads the
        forage field: **detection is a threshold, not a radius**, because one field serves the whole
        population and acuity can only scale what is sampled, never narrow it.

        Normalised by each animal's own best option, so an animal that senses nobody returns all
        zeros and lust drops out of the utility sum entirely rather than voting flatly for
        everywhere — the same reason `Hunger.appeal` normalises.
        """
        acuity = self.genetics.expressed_columns(selection, self._acuity_column)[:, 0].astype(
            np.float64
        )
        perceived = candidates.perceived(self.scent)
        own = self.scent.signature_of(selection).astype(np.float64)
        # Dot each candidate's cue reading against the searcher's own signature. Negative means it
        # smells like the opposite of itself, which the positive threshold discards.
        reading = acuity[:, None] * np.einsum("nkc,nc->nk", perceived, own)
        return detected(reading, self.config.detection_threshold).astype(np.float32)

    def urgency(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: weighted energy surplus, zero before maturity.

        Maturity is per-entity because it is a gene (#20): two animals of the same age can differ
        in whether they are looking yet, which is what lets selection tune age at first
        reproduction against the world rather than a designer choosing it.
        """
        maturity = self.genetics.expressed_columns(selection, self._maturity_column)[:, 0]
        mature = self.store.age[selection.to_mask()] >= maturity
        headroom = self.config.abundant_energy - self.config.breeding_energy
        surplus = (self.ecology.energy(selection) - self.config.breeding_energy) / headroom
        return (self._weight.of(selection) * np.where(mature, np.clip(surplus, 0.0, 1.0), 0.0)).astype(
            np.float32
        )


@dataclass(frozen=True)
class FearConfig:
    """Per-world tuning for the fear drive.

    weight_gene: the gene whose expressed value multiplies this drive's 0-1 shape. A gene
        rather than a constant, so temperament evolves rather than being designed (§2.5,
        #23). Read as a magnitude: a negative weight would invert the drive.
    scent_acuity_gene: the gene whose expressed value is scent detection sensitivity. Named here
        rather than assumed, as `MetabolismConfig.heat_tolerance_gene` is, because the vocabulary is
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

    weight_gene: str
    scent_acuity_gene: str
    aversion_genes: tuple[tuple[str, ...], ...]
    detection_threshold: float
    saturation: float

    def __post_init__(self) -> None:
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
        genes: GeneRegistry,
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
        # Raise KeyError naming the vocabulary version if any gene does not exist. Acuity
        # scales a sampled concentration and an aversion is a direction in cue space; neither
        # carries a dimension.
        self._weight = DriveWeight(genetics, genes, config.weight_gene)
        aversion = np.array(
            [
                [genes.index_of(name, unit=Unit.DIMENSIONLESS) for name in block]
                for block in config.aversion_genes
            ],
            dtype=np.int64,
        )
        # (1 + n_directions * n_channels,) int: acuity, then every aversion gene. The only columns
        # `_channels` reads.
        self._columns = np.concatenate(
            [[genes.index_of(config.scent_acuity_gene, unit=Unit.DIMENSIONLESS)], aversion.ravel()]
        )
        # (n_directions, n_channels): where each direction's genes sit in that block, so scoring
        # every direction is one matrix product rather than a loop over genes.
        self._aversion_positions = 1 + np.arange(aversion.size).reshape(aversion.shape)

    def appeal(self, selection: Selection, candidates: Candidates) -> np.ndarray:
        """(n, n_options) float32 in [0, 1]: how much safer each option is than the worst (#265).

        Danger at a candidate is what urgency reads where the animal stands, read there instead:
        acuity times each aversion direction dotted against the self-excluded air
        (`Scent.perceived_at`, #188), with readings below `detection_threshold` and negative ones
        discarded by the same gate urgency uses, then summed over directions. Appeal is
        ``1 − danger / max danger over the options``, so the most dangerous option scores 0 and one
        where nothing is detected scores 1. That is **relative safety**, chosen on #265.

        **Not read through the saturating curve.** Urgency saturates, so a direction taken from it
        would go flat exactly when the threat is closest: every candidate beside a pack reads as
        certain danger, and flight would vanish at point-blank range. How *much* danger there is
        stays with urgency. Appeal only says which way is safer, which is why it is scaled to the
        animal's own worst option and adds no constant that would have to stay in step with the
        threshold and saturation (§2.1).

        **All zeros when nothing is detected anywhere**, so fear drops out of the sum and the other
        drives decide. That is the rule `detected` states for every other drive, and fear takes it
        here rather than voting equally for every option.

        The known risk, recorded on #265: because the scale is relative, a faint detection spreads
        appeal across the options as widely as a strong one, and only its small urgency keeps it
        quiet. How often that spread decides a choice is measured in
        docs/spikes/fear-direction.md (the #207 question).
        """
        expressed = self.genetics.expressed_columns(selection, self._columns).astype(np.float64)
        perceived = candidates.perceived(self.scent).astype(np.float64)
        acuity = expressed[:, 0]
        # (n, n_directions, n_channels): each animal's aversion directions, in cue-channel order.
        aversion = expressed[:, self._aversion_positions]
        # How strongly each candidate's air points along each aversion direction, scaled by how
        # keen the nose is: (n, n_options, n_directions), the per-candidate form of `_channels`.
        # A batched matmul rather than the equivalent einsum, which takes NumPy's unblocked path
        # here: 133 ms against 25 ms at 100,000 animals and nine options
        # (docs/spikes/fear-direction.md).
        reading = acuity[:, None, None] * (perceived @ aversion.transpose(0, 2, 1))
        danger = np.where(reading >= self.config.detection_threshold, reading, 0.0).sum(axis=2)
        worst = danger.max(axis=1, keepdims=True)
        safety = 1.0 - np.divide(danger, worst, out=np.ones_like(danger), where=worst > 0.0)
        return safety.astype(np.float32)

    def urgency(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: weighted probability that something dangerous was detected."""
        detected = np.stack(self._channels(selection))
        # Noisy-OR: independent detections corroborate without ever summing past certainty, which
        # is what lets #24 add a sense without re-tuning every other drive's weight.
        return (self._weight.of(selection) * (1.0 - np.prod(1.0 - detected, axis=0))).astype(np.float32)

    def _channels(self, selection: Selection) -> list[np.ndarray]:
        """Each channel's detection probability, (len(selection),) float32 in [0, 1].

        One channel per aversion direction, all reading the same nose. #24 appends the same
        directions read through sight instead. Nothing outside a channel knows how its probability
        was computed, so that addition changes no other line in this class.
        """
        expressed = self.genetics.expressed_columns(selection, self._columns)
        # Sampled once and shared: the field read is the expensive part, and every direction
        # weighs the same air differently.
        smelled = self.scent.perceive(selection)
        acuity = expressed[:, 0]
        return [
            self._detect(acuity * (expressed[:, direction] * smelled).sum(axis=1))
            for direction in self._aversion_positions
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

    weight_gene: the gene whose expressed value multiplies this drive's 0-1 shape. A gene
        rather than a constant, so temperament evolves rather than being designed (§2.5,
        #23). Read as a magnitude: a negative weight would invert the drive.
    exertion_saturation: accumulated work per unit of body size at which exhaustion alone is
        reason enough to stop — the point where `Exertion`'s open-ended quantity becomes the
        0-to-1 shape every drive competes in. Must be positive; at zero any movement at all would
        pin fatigue at maximum and the drive would stop discriminating between a stroll and a
        chase.

        It is not derived from `MovementConfig`, even though what fills the column comes from
        there, because the two answer different questions: that config says what a step *costs*,
        this one says how much work is *too much*, and an animal bred for endurance is a world
        where the second moved and the first did not.

    travel_effort: the share of an animal's restfulness that an ordinary step over level ground
        costs, in (0, 1]. **This is the number #207 was about.** Fatigue used to score 1 on the
        null option and 0 on every travelling one, so the gap between resting and walking was
        pinned at the maximum a drive can express — its spread across options equalled its whole
        urgency, measured at 0.921 against hunger's 0.210, and hunger consequently never once
        decided a direction despite ranking the food perfectly.

        Naming the gap makes it a per-world statement about how much walking takes out of a tired
        animal, rather than an implicit "everything". At 1 the old behaviour is recovered exactly,
        which is why the interval is open at the top and closed at the bottom: the degenerate case
        is reachable and stays available, and zero — walking as restful as lying down — would make
        rest unreachable and bring back the drive that cannot act (#126).
    climb_tolerance: world units of *ascent* over which a travelling option loses what is left of
        its appeal, e-folding. Must be positive. This is what gives fatigue a direction rather than
        a bit: a tired animal prefers the valley to the ridge, so terrain shapes where a herd rests
        without anybody placing a bedding ground.

        **Only gain counts, never net elevation change**, which is the identical rule §2.5 settles
        for what a step costs — descent is priced at its horizontal distance and no more. Fatigue
        and the cost function disagreeing about what is tiring would be the same class of defect as
        #112's unit mismatch: two readings of one physical fact, drifting apart with nothing to
        catch it.
    """

    weight_gene: str
    exertion_saturation: float
    travel_effort: float
    climb_tolerance: float

    def __post_init__(self) -> None:
        if self.exertion_saturation <= 0:
            raise ValueError(
                f"exertion_saturation must be positive, got {self.exertion_saturation}; "
                "at zero any movement whatsoever pins fatigue at maximum"
            )
        if not 0.0 < self.travel_effort <= 1.0:
            raise ValueError(
                f"travel_effort must be in (0, 1], got {self.travel_effort}; at zero walking is "
                "as restful as lying down and a tired animal never stops (#126, #207)"
            )
        if self.climb_tolerance <= 0:
            raise ValueError(
                f"climb_tolerance must be positive, got {self.climb_tolerance}; "
                "at zero any ascent at all makes an option maximally tiring"
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

    def __init__(
        self,
        store: EntityStore,
        exertion: Exertion,
        genetics: Genetics,
        terrain: Terrain,
        genes: GeneRegistry,
        config: FatigueConfig,
    ) -> None:
        self.store = store
        self.exertion = exertion
        self.terrain = terrain
        self.config = config
        self._weight = DriveWeight(genetics, genes, config.weight_gene)

    def appeal(self, selection: Selection, candidates: Candidates) -> np.ndarray:
        """(n, n_options) float32: how restful each option looks, 1 for staying put.

        ```
        appeal(travelling) = (1 − travel_effort) × exp(−ascent / climb_tolerance)
        appeal(staying)    = 1
        ```

        Staying put is maximally restful because it proposes no displacement, so it costs nothing
        and the animal recovers (#107). That is what makes rest reachable without a mode, a flag or
        a state column (#114). It is identified by **zero displacement rather than by its column
        index**: `Behaviour` does put the null option last, but an option that proposes going
        nowhere is restful because it goes nowhere. Since #209 `Behaviour` never offers a travelling
        candidate that clips onto the animal's own position, so the two definitions agree on every
        option it scores. Displacement is kept as the test because it is the reason, not the label.

        **Travelling is graded, not vetoed, and that is #207.** This used to return exactly 0 on
        every travelling option, so the drive's spread across options equalled its entire urgency
        — 0.921 measured against hunger's 0.210 — and hunger, which ranks the food correctly 0.998
        of the time, never once decided a direction. A drive's influence over a *ranking* is how
        far its appeal varies between the options, not how much it wants; expressing "rest" as an
        all-or-nothing bit spent the largest voice in the contest on one bit of information.

        **Ascent is what gives it a direction.** Distance cannot: every candidate sits one
        `look_ahead` away, so it is identical across the options and discriminates nothing. Relief
        is the one thing that differs, and it differs in the direction that matters — a tired
        animal prefers the valley to the ridge, so where a herd settles is a reading of the terrain
        rather than an authored bedding ground.

        **Only gain counts, never net elevation change**, matching what a step actually costs
        (§2.5, #113): descent is charged its horizontal distance and no more. Fatigue calling a
        downhill option restful while `Movement` charges it the same as level ground would be two
        readings of one physical fact drifting apart, which is the shape of defect #112 was.

        Elevation is sampled at the candidate rather than walked (#113's cell crossing) because
        this is a *perception* and not a bill: an animal looks at a slope and judges it, it does
        not survey the path first. The walk is what the pool is charged against, and it still is.
        """
        x, y = candidates.x, candidates.y
        mask = selection.to_mask()
        here_x = self.store.x[mask].astype(np.float64)[:, None]
        here_y = self.store.y[mask].astype(np.float64)[:, None]
        ascent = np.maximum(
            self.terrain.elevation_at(x, y) - self.terrain.elevation_at(here_x, here_y), 0.0
        )
        travelling = (
            1.0 - self.config.travel_effort
        ) * np.exp(-ascent / self.config.climb_tolerance)
        # Exactly zero, not near it: `Behaviour.candidate_positions` writes the animal's own
        # coordinates into the null column unchanged, so the comparison is on identical floats.
        return np.where(np.hypot(x - here_x, y - here_y) == 0.0, 1.0, travelling).astype(np.float32)

    def urgency(self, selection: Selection) -> np.ndarray:
        """(len(selection),) float32: weighted urgency to rest, zero for a healthy idle animal."""
        injury = np.clip(1.0 - self.store.health[selection.to_mask()], 0.0, 1.0)
        spent = np.clip(
            self.exertion.exerted(selection) / self.config.exertion_saturation, 0.0, 1.0
        )
        # Noisy-OR over the two reasons: neither alone can saturate the score, and both together
        # exceed either — an injured animal that has also just been running wants to stop most.
        return (self._weight.of(selection) * (1.0 - (1.0 - injury) * (1.0 - spent))).astype(np.float32)
