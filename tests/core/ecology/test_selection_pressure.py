"""Issue #17's "done when": raising a trait without a compensating benefit must reduce survival,
and the energy invariant must hold across a long run.

These are statistical and directional, never exact (CLAUDE.md §6, §8.1): cohorts are drawn from
overlapping distributions and the assertions are about which cohort outlasts which, over many
seeds. Nothing here asserts a survival time.

Survival is measured as ticks-until-the-pool-empties rather than as death, because the death path
belongs to #21 and does not exist yet. That is the same quantity: an animal at zero energy has run
out of the only budget it has, and #21 will consume exactly the `starving()` selection this
measures.
"""

import numpy as np
import pytest

from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.ecology.service import Ecology
from core.entities.store import EntityStore
from core.genetics.expression import GeneticsConfig
from core.genetics.service import Genetics
from core.genetics.species import SpeciesRegistry
from core.invariants import default_registry
from core.selection import Selection
from core.services import ColumnRegistry
from core.world.climate import Climate, ClimateConfig
from core.world.terrain import Terrain
from core.world.tick import TickLoop

from tests.support.genes import gene_registry
from tests.support.plants import plant_field


GENE_NAMES = ("size", "speed", "sight", "insulation", "mutability")

# Every gene declares how its stored value is read (#104). These are all quantities, so all fold
# across zero; `mutability` is in the vocabulary because inheritance's spread floor is a gene, and
# every world needs one even when — as here — nothing in these tests breeds.
GENETICS_CONFIG = GeneticsConfig(
    mutability_gene="mutability",
    drift_margin=2.0,
)
GENE_REGISTRY = gene_registry(GENE_NAMES, {"size": 2.0, "speed": 3.0, "insulation": 1.0})

METABOLISM_CONFIG = MetabolismConfig(
    dehydration_penalty=0.0,
    basal_rate=1.0,
    thermoregulation_rate=0.5,
    neutral_temperature=20.0,
    insulation_gene="insulation",
)

STARTING_ENERGY = 100.0
COHORT_SIZE = 200


def make_world(capacity=1024, equator_temperature=20.0, latitude_gradient=0.0):
    store = EntityStore(initial_capacity=capacity, n_drives=1, n_genes=len(GENE_NAMES))
    registry = ColumnRegistry()
    vocabulary = GENE_REGISTRY
    species = SpeciesRegistry(vocabulary)
    genetics = Genetics(store, registry, species, vocabulary, GENETICS_CONFIG)
    terrain = Terrain(np.zeros((41, 41), dtype=np.float32), cell_size=1.0)
    climate = Climate(
        terrain,
        ClimateConfig(
            equator_y=0.0,
            equator_temperature=equator_temperature,
            latitude_gradient=latitude_gradient,
        ),
    )
    ecology = Ecology(
        store,
        registry,
        genetics,
        climate,
        Metabolism(vocabulary, METABOLISM_CONFIG),
        # Every row this fixture can allocate is endowed with STARTING_ENERGY it never grazed
        # for, and excretion returns nutrients against the export ledger (#21).
        plant_field(terrain, climate, founding_stock=capacity * STARTING_ENERGY),
    )
    return store, species, genetics, ecology


def add_cohort(store, genetics, species_id, gene_values, y=0.0):
    """Allocate one cohort with the given (n, n_genes) genotypes; returns its Selection."""
    n = gene_values.shape[0]
    ids = store.allocate(
        n,
        x=np.zeros(n, dtype=np.float32),
        y=np.full(n, y, dtype=np.float32),
        energy=np.full(n, STARTING_ENERGY, dtype=np.float32),
        species_id=np.full(n, species_id, dtype=np.int32),
    )
    rows = np.array([store._id_to_row[i] for i in ids.tolist()], dtype=np.int64)
    selection = Selection.from_indices(rows, capacity=store.capacity)
    genetics.set_genes(selection, gene_values)
    return selection


def genotypes(rng, n, **gene_means):
    """(n, n_genes) float32 drawn around per-gene means, so cohorts overlap rather than being
    two uniform blocks -- the comparison has to survive individual variation.
    """
    values = np.zeros((n, len(GENE_NAMES)), dtype=np.float32)
    for name, mean in gene_means.items():
        column = rng.normal(mean, 0.2, size=n)
        values[:, GENE_NAMES.index(name)] = np.clip(column, 0.0, None)
    return values


def mean_survival_ticks(store, ecology, cohorts, max_ticks=200):
    """Mean ticks each cohort's members took to empty their pool, advancing all cohorts together.

    An entity that never empties is recorded at `max_ticks + 1` (right-censored). With the
    energies and costs used here every entity starves well inside the window, so the censoring
    value never enters a mean -- it is there so a mistuned test fails loudly instead of silently
    averaging a truncated distribution.
    """
    all_rows = np.concatenate([c.to_indices() for c in cohorts])
    starved_at = np.full(store.capacity, max_ticks + 1, dtype=np.int64)
    everyone = Selection.from_indices(all_rows, capacity=store.capacity)

    for tick in range(1, max_ticks + 1):
        ecology.drain(everyone)
        newly_starved = (store.energy <= 0) & (starved_at > max_ticks) & store.alive
        starved_at[newly_starved] = tick

    return [float(starved_at[cohort.to_indices()].mean()) for cohort in cohorts]


class TestACostlyTraitWithoutABenefitIsSelectedAgainst:
    @pytest.mark.parametrize("seed", range(8))
    def test_the_faster_cohort_empties_its_pool_first(self, seed):
        """Speed buys nothing in this world -- fleeing and foraging are #22 and #25 -- so the
        only thing a higher speed gene does here is charge more upkeep. This is the mechanism
        that stops everything evolving toward maximum everything (CLAUDE.md §2.5).
        """
        rng = np.random.default_rng(seed)
        store, species, genetics, ecology = make_world()
        species_id = species.register(GENE_NAMES)
        slow = add_cohort(store, genetics, species_id, genotypes(rng, COHORT_SIZE, speed=1.0))
        fast = add_cohort(store, genetics, species_id, genotypes(rng, COHORT_SIZE, speed=3.0))

        slow_survival, fast_survival = mean_survival_ticks(store, ecology, [slow, fast])

        assert fast_survival < slow_survival

    def test_the_same_holds_for_an_unexpressed_costly_gene_only_because_it_is_not_charged(self):
        """The cost follows expression, not the genotype (issue #17): a species that does not
        express speed carries the gene, pays nothing for it, and outlasts one that does.
        """
        rng = np.random.default_rng(19)
        store, species, genetics, ecology = make_world()
        expresses_speed = species.register(GENE_NAMES)
        ignores_speed = species.register(("size", "sight", "insulation"))
        genes = genotypes(rng, COHORT_SIZE, speed=3.0)
        charged = add_cohort(store, genetics, expresses_speed, genes)
        dormant = add_cohort(store, genetics, ignores_speed, genes.copy())

        charged_survival, dormant_survival = mean_survival_ticks(
            store, ecology, [charged, dormant]
        )

        assert charged_survival < dormant_survival


class TestClimateSelectsDifferentBuilds:
    """CLAUDE.md §2.5: thermoregulation cost differs by temperature, so a cold zone selects for
    insulation and a temperate one selects against it -- with nobody designing that outcome. The
    crossover is the claim; that neither build wins everywhere is the whole point.
    """

    @pytest.mark.parametrize("seed", range(4))
    def test_insulation_pays_for_itself_in_the_cold_and_costs_in_the_temperate_zone(self, seed):
        rng = np.random.default_rng(seed)
        # 20 degC (neutral) at y=0, falling 1 degC per world unit: y=0 is temperate, y=20 polar.
        store, species, genetics, ecology = make_world(
            equator_temperature=20.0, latitude_gradient=1.0
        )
        species_id = species.register(GENE_NAMES)

        temperate_bare = add_cohort(
            store, genetics, species_id, genotypes(rng, COHORT_SIZE, insulation=0.0), y=0.0
        )
        temperate_insulated = add_cohort(
            store, genetics, species_id, genotypes(rng, COHORT_SIZE, insulation=4.0), y=0.0
        )
        polar_bare = add_cohort(
            store, genetics, species_id, genotypes(rng, COHORT_SIZE, insulation=0.0), y=20.0
        )
        polar_insulated = add_cohort(
            store, genetics, species_id, genotypes(rng, COHORT_SIZE, insulation=4.0), y=20.0
        )

        survival = mean_survival_ticks(
            store,
            ecology,
            [temperate_bare, temperate_insulated, polar_bare, polar_insulated],
        )
        temperate_bare_ticks, temperate_insulated_ticks = survival[0], survival[1]
        polar_bare_ticks, polar_insulated_ticks = survival[2], survival[3]

        assert temperate_insulated_ticks < temperate_bare_ticks
        assert polar_insulated_ticks > polar_bare_ticks


class TestTheEnergyInvariantHoldsAcrossALongRun:
    def test_five_hundred_ticks_of_upkeep_never_trip_an_invariant(self):
        """The drain runs as a registered system under the real tick loop with debug checks on,
        so #7's harness -- including "no alive entity has negative energy" -- is evaluated after
        every one of the 500 ticks, not just at the end.
        """
        rng = np.random.default_rng(5)
        store, species, genetics, ecology = make_world(
            equator_temperature=20.0, latitude_gradient=1.0
        )
        species_id = species.register(GENE_NAMES)
        cohort = add_cohort(
            store,
            genetics,
            species_id,
            genotypes(rng, COHORT_SIZE, size=1.0, speed=1.0, insulation=1.0),
            y=15.0,
        )

        def metabolic_upkeep():
            ecology.drain(Selection.from_mask(store.alive))

        loop = TickLoop(
            store,
            systems=[metabolic_upkeep],
            invariants=default_registry(
                0.0,
                ecology.climate.terrain.world_width,
                0.0,
                ecology.climate.terrain.world_height,
            ),
            debug_checks=True,
        )

        loop.advance(500)

        assert loop.tick_count == 500
        # Every entity long since drained to exactly zero -- emptied, never overdrawn.
        assert (ecology.energy(cohort) == 0.0).all()
        assert ecology.starving(cohort) == cohort
