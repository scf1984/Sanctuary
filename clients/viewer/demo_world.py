"""The world the diagnostic viewer looks at: a real assembled world, not a scatter of markers.

**Separate from `app.py` because `app.py` imports pygame at module scope.** CI installs `.[dev]`
and never the viewer extra — deliberately, since a run that never installs pygame is a standing
check that the core is runnable headless (CLAUDE.md §3) — so nothing importing `app.py` is
collectable there. World-building sitting inside that module is precisely why its `EntityStore`
call could go a whole gene-matrix release without `n_genes` and without anyone noticing (#110):
the one entry point that runs the simulation as a program was the one module no test could load.
The rule `render.py` already states — pygame lives in `app.py` and nowhere else — earns its keep
only if everything worth testing stays on this side of it.

This module is now **config and nothing else**. `core.world.assembly.build_world` (#115) does the
wiring, so the viewer cannot drift into a second assembly disagreeing with the settled tick order —
which is what §7.2 exists to prevent, and what the previous version of this file was explicitly
waiting for.

**These numbers are one world, not defaults.** Nothing else should import them. A world tuned for
watching — small enough to fill a window, warm enough that plants grow, populated enough that
something is always happening — is not the world a competition (#41) or a shipped starting state
(#101) would want, and dressing it up as a shared default is how a viewing preference becomes an
ecological constant (§2.1: coefficients are per world, never constants in `core/`).
"""

from __future__ import annotations

from core.behaviour.drives import (
    FatigueConfig,
    FearConfig,
    HungerConfig,
    LustConfig,
    ThirstConfig,
)
from core.behaviour.exertion import ExertionConfig
from core.behaviour.service import BehaviourConfig
from core.behaviour.movement import MovementConfig
from core.ecology.conception import ConceptionConfig
from core.entities.growth import GrowthConfig
from core.ecology.cues import CueFieldConfig, ScentGenes
from core.ecology.diet import DietConfig
from core.ecology.feeding import FeedingConfig
from core.ecology.metabolism import MetabolismConfig
from core.ecology.plants import PlantsConfig
from core.ecology.carrion import CarrionConfig
from core.ecology.predation import PredationConfig
from core.genetics.expression import GeneticsConfig
from core.genetics.registry import ExpressionMode, GeneSpec, Unit
from core.world.diffusion import DiffusionConfig
from core.world.assembly import World, WorldConfig, build_world
from core.world.climate import ClimateConfig
from core.world.terrain import TerrainConfig
from core.world.interventions import Interventions
from metrics import MetricHistory, MetricsConfig

_GRID = 80
_CELL_SIZE = 1.0
_WORLD_EXTENT = (_GRID - 1) * _CELL_SIZE

# Cue space is eight-dimensional, per CLAUDE.md §2.5's settled floor. The signature genes and the
# two aversion directions are that reserved block, named here because the vocabulary is per world.
_SIGNATURE_GENES = tuple(f"signature_{channel}" for channel in range(8))
_AVERSION_GENES = (
    tuple(f"aversion0_{channel}" for channel in range(8)),
    tuple(f"aversion1_{channel}" for channel in range(8)),
)
def _cue_gene(name: str, meaning: str) -> GeneSpec:
    """A cue-space gene: signed, free, and dimensionless.

    Signed because cue space is a space — a signature is a position in it and an aversion a
    direction through it, and the sign is what doubles its discriminating power (§2.5, #104). Free
    because a cost on a signed gene subtracts from upkeep wherever the value is negative, which is
    a discount selection would chase rather than a charge (#136) — and because §2.5 wants exactly
    this block drifting neutrally, as the molecular clock that makes two isolated populations
    recognisably different.
    """
    return GeneSpec(
        name=name,
        cost=0.0,
        expression_mode=ExpressionMode.SIGNED,
        unit=Unit.DIMENSIONLESS,
        description=meaning,
    )


_GENES = (
    GeneSpec(
        name="size",
        cost=0.01,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description="Body scale. Multiplies every locomotion cost, and is never a mass (§2.6).",
    ),
    GeneSpec(
        name="speed",
        cost=0.01,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.LENGTH,
        description="Top speed, in world units per tick — a length, since the tick is unitless.",
    ),
    GeneSpec(
        name="agility",
        # Costed like speed, and it must be costed at all: turning faster is pure benefit, so a
        # free agility gene runs away in every world (§2.5's rule for insulation). `Movement`
        # refuses to build without a positive figure here.
        cost=0.01,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.LENGTH,
        description=(
            "How fast velocity may change, in world units per tick per tick — a length, since the "
            "tick is unitless. Divided by size, so a big body corners badly and speed trades "
            "against nimbleness (#204)."
        ),
    ),
    GeneSpec(
        name="haste",
        cost=0.0,
        expression_mode=ExpressionMode.EXPONENTIAL,
        unit=Unit.DIMENSIONLESS,
        description=(
            "How readily a reason to move becomes speed: the scale on which a utility advantage "
            "over resting is read as a pace. Exponential because it is a scale and must stay "
            "strictly positive — a negative one would make a better option slower. Free, because "
            "hurrying already pays `exertion_premium` on every world unit (#203)."
        ),
    ),
    GeneSpec(
        name="insulation",
        cost=0.01,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "Damps thermoregulation upkeep with diminishing returns. Must charge a positive cost: "
            "a gene that only reduces upkeep and charges nothing is unbounded free benefit."
        ),
    ),
    GeneSpec(
        name="sight",
        cost=0.01,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description="Visual acuity: scales what is sampled from the forage field, not a radius.",
    ),
    GeneSpec(
        name="scent_emission",
        cost=0.01,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "Broadcast strength on the scent modality. This world charges it, which §2.5 argues is "
            "a trap — low emission is already a survival benefit, so a cost makes silence both "
            "cheaper and safer and drives emission to zero. Kept as-is here because changing a "
            "cost moves outcomes (§2.8); filed separately rather than fixed in a refactor."
        ),
    ),
    GeneSpec(
        name="scent_acuity",
        cost=0.01,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description="Scent sensitivity, which for a diffused plume is also detection range.",
    ),
    *(_cue_gene(name, "Position in cue space: what this creature smells like.")
      for name in _SIGNATURE_GENES),
    *(_cue_gene(name, "First aversion direction: one region of cue space that frightens it.")
      for name in _AVERSION_GENES[0]),
    *(_cue_gene(name, "Second aversion direction, so two unrelated threats need not be averaged.")
      for name in _AVERSION_GENES[1]),
    GeneSpec(
        name="choice_temperature",
        cost=0.0,
        expression_mode=ExpressionMode.EXPONENTIAL,
        unit=Unit.DIMENSIONLESS,
        description=(
            "Boltzmann temperature for option sampling: low commits to the best heading, high "
            "explores. Read through `exp` so it can never reach zero however far it drifts. Free, "
            "because exploring badly is its own price."
        ),
    ),
    GeneSpec(
        name="commitment",
        cost=0.0,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "How doggedly a bearing is held across ticks: the bonus a candidate earns for "
            "continuing last tick's heading, and so half the width of the band a competing drive "
            "must clear to turn the animal. Read as a magnitude, because a negative bonus would "
            "reward reversing. Free: dithering and tunnel vision are each their own price (#100)."
        ),
    ),
    GeneSpec(
        name="diet_animal_derived",
        cost=0.0,
        expression_mode=ExpressionMode.UNIT_INTERVAL,
        unit=Unit.DIMENSIONLESS,
        description=(
            "How a gut is split between plant and animal food: 0 is a pure herbivore, 1 a pure "
            "carnivore, and what is given to one side is taken from the other. An allocation "
            "rather than a pair of capacities, because two capacities can both rise and this "
            "cannot (#102, #146). Free: being allocated for food this world does not hold is its "
            "own price, immediately."
        ),
    ),
    GeneSpec(
        name="maturity_age",
        cost=0.0,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "Ticks an animal must live before it seeks a mate at all. A gene rather than a "
            "constant, because age at first reproduction is among the most strongly selected "
            "life-history traits there is and a world that fixes it decides by hand what the "
            "environment should decide (§2.5, #20). Free: late maturity is already paid for in "
            "generations forgone."
        ),
    ),
    GeneSpec(
        name="gestation_length",
        cost=0.0,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "Ticks between conception and birth, carried as a negative starting age until the "
            "young is born (#20). A gene because life-history theory puts it under exactly the "
            "selection this world applies: a short gestation returns a parent to breeding sooner. "
            "Read as a magnitude, so a lineage drifting below zero is born at once rather than "
            "never. Free: a hurried young is its own price."
        ),
    ),
    GeneSpec(
        name="hunger_weight",
        cost=0.0,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "How heavily hunger counts in the drive contest. A gene rather than a constant, so "
            "temperament evolves rather than being designed (§2.5, #23) — an animal that weights "
            "hunger badly for its world is outcompeted by one that does not. Read as a magnitude, "
            "because a negative weight would invert the drive. Free: mis-weighting is its own "
            "price, immediately."
        ),
    ),
    GeneSpec(
        name="thirst_weight",
        cost=0.0,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "How heavily thirst counts in the drive contest. A gene rather than a constant, so "
            "temperament evolves rather than being designed (§2.5, #23) — an animal that weights "
            "thirst badly for its world is outcompeted by one that does not. Read as a magnitude, "
            "because a negative weight would invert the drive. Free: mis-weighting is its own "
            "price, immediately."
        ),
    ),
    GeneSpec(
        name="fear_weight",
        cost=0.0,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "How heavily fear counts in the drive contest. A gene rather than a constant, so "
            "temperament evolves rather than being designed (§2.5, #23) — an animal that weights "
            "fear badly for its world is outcompeted by one that does not. Read as a magnitude, "
            "because a negative weight would invert the drive. Free: mis-weighting is its own "
            "price, immediately."
        ),
    ),
    GeneSpec(
        name="lust_weight",
        cost=0.0,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "How heavily lust counts in the drive contest. A gene rather than a constant, so "
            "temperament evolves rather than being designed (§2.5, #23) — an animal that weights "
            "lust badly for its world is outcompeted by one that does not. Read as a magnitude, "
            "because a negative weight would invert the drive. Free: mis-weighting is its own "
            "price, immediately."
        ),
    ),
    GeneSpec(
        name="fatigue_weight",
        cost=0.0,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "How heavily fatigue counts in the drive contest. A gene rather than a constant, so "
            "temperament evolves rather than being designed (§2.5, #23) — an animal that weights "
            "fatigue badly for its world is outcompeted by one that does not. Read as a magnitude, "
            "because a negative weight would invert the drive. Free: mis-weighting is its own "
            "price, immediately."
        ),
    ),
    GeneSpec(
        name="mutability",
        cost=0.0,
        expression_mode=ExpressionMode.MAGNITUDE,
        unit=Unit.DIMENSIONLESS,
        description=(
            "Floors the spread of an offspring's inherited draw, so a lineage evolves its own "
            "evolvability. Free: an unfit brood is its own price (#104)."
        ),
    ),
)
_GENE_NAMES = tuple(gene.name for gene in _GENES)


def demo_world_config(n_entities: int, seed: int) -> WorldConfig:
    """One world's coefficients: warm, green, and small enough to watch."""
    return WorldConfig(
        terrain=TerrainConfig(
            width=_GRID,
            height=_GRID,
            # Relief a tenth of extent, the ratio `TerrainConfig` asks callers to choose (#112).
            # Elevation shares x and y's unit, so this is directly a statement about how steep the
            # world looks — and about what crossing a ridge costs, since `climb_cost` below is
            # denominated in that same unit as `transport_cost`.
            min_elevation=0.0,
            max_elevation=_WORLD_EXTENT / 10.0,
            cell_size=_CELL_SIZE,
            seed=seed,
        ),
        climate=ClimateConfig(equator_y=_WORLD_EXTENT / 2.0),
        plants=PlantsConfig(
            solar_constant=8.0,
            latitude_tilt=0.0,
            min_growth_temperature=0.0,
            optimal_growth_temperature=22.0,
            max_growth_temperature=45.0,
            nutrient_per_biomass=1.0,
            initial_soil_nutrients=400.0,
            senescence_rate=0.02,
            saturation_accumulation=20.0,
            max_rooting_depth=0.5,
            forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
        ),
        growth=GrowthConfig(
            # Grow when free rows fall below a tenth of the rows in use. Measured in
            # docs/spikes/conception-and-capacity.md: at the steepest growth a world managed about
            # 0.43% of occupancy allocated per tick, so a tenth is roughly twenty ticks of runway
            # — and since `grow` doubles, it is reached rarely.
            reserve_fraction=0.1,
        ),
        conception=ConceptionConfig(
            # World units. A contact distance, not a search radius: finding each other is the lust
            # drive's business (#188), and by the time two animals are this close they have walked.
            contact_range=2.0,
            # Energy units, moved out of the parents rather than charged and burned. Under a
            # founder's 180 so a healthy adult can breed more than once before feeding back up —
            # the breeding interval is emergent from that rather than being a constant.
            offspring_energy=60.0,
            maturity_gene="maturity_age",
            gestation_gene="gestation_length",
            # Genetic distance at which interbreeding reaches zero; #16 reads the same number.
            #
            # **Derived rather than chosen**: four times the 99th-percentile pairwise distance
            # *within* this world's population, which measures ~73 (docs/spikes/
            # interbreeding-threshold.md). The rule is what matters, not the value — the gate
            # exists to stop *diverged populations* interbreeding, so it must sit above the
            # spread a single healthy population already carries, or it throttles ordinary
            # reproduction instead.
            #
            # It shipped at 8.0 against a median distance of ~29, which rejected 94% of
            # candidate couples and suppressed births roughly 24-fold. **Re-derive this after
            # #193**: the distance metric is currently dominated by `maturity_age` and
            # `gestation_length`, and changing what distance measures changes this number
            # completely.
            speciation_threshold=300.0,
        ),
        diet=DietConfig(
            animal_derived_gene="diet_animal_derived",
            # Above 1 so a generalist is strictly worse than a specialist at that specialist's own
            # food (#146). At 2 an even split converts a quarter as well as a pure herbivore,
            # which in a world holding only plants is a strong pull toward herbivory — and that
            # pull is the point: nothing here declares these creatures herbivores.
            frontier_exponent=2.0,
        ),
        predation=PredationConfig(
            # Half the mating range: reaching a mate is a meeting and reaching prey is a catch.
            strike_range=1.0,
            # Energy units taken out of an equally-sized victim at full flesh allocation. Set
            # against a founder's 180 and a settled adult's ~50, so a committed carnivore kills a
            # healthy grazer in one or two strikes rather than gnawing it for a dozen ticks — the
            # force limit, deliberately an order of magnitude above `intake_rate`'s gut limit.
            strike_power=60.0,
        ),
        carrion=CarrionConfig(
            # A carcass loses a twentieth of itself per tick, so half of it is gone in ~14 ticks.
            # Long enough that a predator eating at `intake_rate` can finish a kill it stays with,
            # short enough that the ground does not fill with meat nobody claims — the two halves
            # of "is a kill worth defending", tuned as one against `intake_rate` (§2.1).
            decay_rate=0.05,
        ),
        feeding=FeedingConfig(
            # Chosen so a *naive* founder population is viable, which is a stricter and temporary
            # requirement than it looks: nothing dies (#21) and nothing breeds (#20), so selection
            # cannot move the diet distribution and the rate has to carry animals that were badly
            # allocated by chance. Measured in docs/spikes/grazing-equilibrium.md over 3 seeds and
            # 2,500 ticks: 10% of founders viable at 2.5, 47% at 6.0, 66% here. Revisit once #20
            # and #21 let selection do this work instead — it should look generous by then.
            intake_rate=9.0,
            # No gut extracts everything: the remainder is faeces, and it is what fertilises the
            # ground a herd grazes over.
            assimilation_max=0.5,
            size_gene="size",
        ),
        cue_field=CueFieldConfig(diffusion_range=3.0),
        metabolism=MetabolismConfig(
            basal_rate=0.05,
            thermoregulation_rate=0.01,
            neutral_temperature=20.0,
            insulation_gene="insulation",
        ),
        genetics=GeneticsConfig(
            mutability_gene="mutability",
            drift_margin=2.0,
        ),
        movement=MovementConfig(
            speed_gene="speed",
            size_gene="size",
            agility_gene="agility",
            haste_gene="haste",
            transport_cost=0.5,
            exertion_premium=2.0,
            climb_cost=1.0,
            # The pace an animal uses when moving buys it nothing over standing still. Everything
            # above this is bought with `haste` against a real utility advantage, and the gap to 1
            # is what the premium finally has to multiply (#203).
            walking_pace=0.4,
        ),
        exertion=ExertionConfig(recovery_rate=0.2),
        hunger=HungerConfig(
            weight_gene="hunger_weight", satiation_energy=200.0, detection_threshold=0.5, sight_gene="sight"
        ),
        # Thirst is held quiet on purpose — see its founding range below, which is where the
        # damping lives now that the weight is a gene (#23).
        thirst=ThirstConfig(weight_gene="thirst_weight", onset_temperature=25.0, saturation_temperature=40.0),
        fear=FearConfig(
            weight_gene="fear_weight",
            scent_acuity_gene="scent_acuity",
            aversion_genes=_AVERSION_GENES,
            detection_threshold=0.05,
            saturation=1.0,
        ),
        lust=LustConfig(
            weight_gene="lust_weight",
            maturity_gene="maturity_age",
            scent_acuity_gene="scent_acuity",
            detection_threshold=0.05,
            breeding_energy=120.0,
            abundant_energy=250.0,
        ),
        fatigue=FatigueConfig(
            weight_gene="fatigue_weight",
            exertion_saturation=20.0,
            # Measured, not chosen (§8.5, docs/spikes/who-steers.md). At the 1.0 this replaces,
            # fatigue's spread across options was 0.93 against hunger's 0.20 — it decided every
            # direction in the world while hunger, which ranks the food correctly 0.998 of the
            # time, decided none (#207). At 0.25 that spread is 0.34, comparable to the
            # commitment band's 0.29, and population and condition are unchanged: this is a
            # latent pathology removed rather than an equilibrium moved.
            travel_effort=0.25,
            # World units of ascent. Roughly twice the *largest* rise measured between an animal
            # and one of its candidates (p50 0.33, p99 1.46, max 1.87 over a settled world), so
            # the discount is gentle and graded across the whole observed range rather than
            # saturating on ordinary ground. At 1.0 — comparable to the biggest rises — fatigue
            # steered downhill hard enough to cost condition (energy 57.0 against 60.4) while
            # leaving its spread at 0.55, still the loudest voice in the contest.
            climb_tolerance=4.0,
        ),
        behaviour=BehaviourConfig(
            # Eight headings is enough that a forager can follow a gradient without the walk
            # visibly staircasing, and the per-entity jitter makes the effective resolution
            # continuous across the population.
            n_candidates=8,
            # One diffusion range: the distance over which the forage field carries information,
            # so it is the furthest a candidate reading can vouch for.
            look_ahead=4.0,
            commitment_gene="commitment",
            choice_temperature_gene="choice_temperature",
        ),
        scent_genes=ScentGenes(emission_gene="scent_emission", signature_genes=_SIGNATURE_GENES),
        genes=_GENES,
        # Naive founders (§2.5, #101): a uniform draw over the whole cue space rather than a chosen
        # point, so nothing here writes down what a lineage smells like or what frightens it.
        founder_gene_ranges={
            "size": (0.8, 1.2),
            "speed": (1.0, 3.0),
            # World units per tick per tick, against top speeds of 1–3 and sizes around 1, so a
            # founder needs roughly two to five ticks to reach its own top speed from rest and as
            # long again to reverse. Wide enough that founders differ in nimbleness from the first
            # generation, which is what selection needs to build a predator and its prey out of.
            "agility": (0.3, 0.9),
            "insulation": (0.0, 0.5),
            "sight": (2.0, 6.0),
            "scent_emission": (0.5, 1.5),
            "scent_acuity": (0.5, 1.5),
            **{name: (0.0, 1.0) for name in _SIGNATURE_GENES},
            **{name: (-1.0, 1.0) for name in _AVERSION_GENES[0] + _AVERSION_GENES[1]},
            # Around the 0.02 the drift spike measured as a working mutation-drift balance, drawn
            # rather than fixed so founders vary in evolvability like anything else
            # (docs/spikes/speciation-drift.md).
            "mutability": (0.01, 0.03),
            # Around the 0.15 this was as a constant, which is a band of 0.3 against drive
            # utilities that reach 1 — enough to hold a bearing through indifferent ground and not
            # enough to hold one against an appetite. Drawn rather than fixed, so founders differ
            # in doggedness and selection has something to act on from the first generation.
            "commitment": (0.05, 0.25),
            # Drive weights, drawn around 1 so founders differ in temperament from the first
            # generation and selection has something to act on (§2.5, #23).
            "hunger_weight": (0.6, 1.4),
            # Thirst founds an order of magnitude quieter than the rest, and this is a
            # workaround rather than a tuning choice: a drive that *wins* with no mechanic
            # behind it leaves the animal standing still (#126), and nothing drinks (#156).
            # It was a config weight of 0.2 before the weights became genes (#23); drawing it
            # from the same range as the others silently discarded that, and thirst then took
            # 100% of every well-fed animal's decision.
            "thirst_weight": (0.1, 0.3),
            "fear_weight": (0.6, 1.4),
            "lust_weight": (0.6, 1.4),
            "fatigue_weight": (0.6, 1.4),
            # Around zero, so `exp` puts founding temperatures around 1 — the scale at which a
            # utility difference of one is a decisive preference rather than a faint one.
            "choice_temperature": (-0.3, 0.3),
            # Read through `exp`, so this founds haste between 1 and about 4. Chosen against a
            # measurement rather than by feel (§8.5): over 400 founders at tick 60 the drive
            # advantage of the best option over resting has a median of ~0 and a 99th percentile
            # of ~0.2, so haste 1 leaves even a strongly-motivated animal ambling at 0.44 while
            # haste 4 puts it at 0.73 — the band across which the gene visibly does something.
            # Founders therefore differ from "barely hurries" to "hurries readily", which is what
            # selection needs to act on. See docs/spikes/pace-and-momentum.md.
            "haste": (0.0, 1.4),
            # Spread across zero, which the logistic squash reads as allocations either side of an
            # even split. Founders are therefore *undecided* about what they eat rather than
            # declared herbivores — this world holds only plants, so selection settles it within a
            # few generations, and that is the point (§2.5, #101: founders are naive).
            "diet_animal_derived": (-1.0, 1.0),
            # Ticks. Wide, and low relative to a lifetime, so founders differ in how soon they
            # breed and selection has something to act on from the first generation.
            "maturity_age": (40.0, 120.0),
            # Ticks. Short against a lifetime, and drawn so founders differ from generation one.
            "gestation_length": (20.0, 60.0),
        },
        n_founders=n_entities,
        founder_energy=180.0,
    )


# Every 20 ticks, keeping 2,000 samples — the last 40,000 ticks, which at §2.1's rates is about
# four sim-months, or two generations. Long enough that a trait mean visibly moves across the
# window, and fine enough that a population crash does not fall between two samples.
DEMO_METRICS = MetricsConfig(every_n_ticks=20, history_limit=2_000)

# What the player starts with. A round number and openly arbitrary: what *generates* intervention
# currency is an open question (§5) settled only in the negative so far, so a demo world is handed a
# balance and nothing here creates more. Culling costs one per animal (#26), which makes this a few
# hundred animals' worth of stewardship before the budget is the constraint rather than the ecology.
DEMO_BUDGET = 500.0


def build_demo_world(seed: int, n_entities: int) -> World:
    """The assembled world the viewer runs, reproducible from `seed`.

    Generation is a pure function of `seed` (§2.2): the simulation itself is non-deterministic, but
    a world the viewer cannot rebuild is one whose crash cannot be replayed.
    """
    world = build_world(demo_world_config(n_entities, seed), seed=seed)
    # Attached rather than assembled, because a recorder reads a finished world and `core/` must
    # stay importable without `metrics/` (§4). This is the composition root doing composition.
    world.loop.metrics = MetricHistory(
        world.store, world.genetics, world.plants, world.genes.vocabulary, DEMO_METRICS
    )
    world.loop.interventions = Interventions(balance=DEMO_BUDGET)
    return world
