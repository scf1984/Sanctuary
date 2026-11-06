import numpy as np
import pytest

from core.ecology.metabolism import Metabolism, MetabolismConfig
from core.genetics.registry import ExpressionMode, GeneRegistry, GeneSpec, Unit

from tests.support.genes import gene_registry

# `aversion0_0` is here to make the fixture representative rather than convenient: a real
# vocabulary mixes quantities with cue-space directions, and the rule this module enforces is
# about the boundary between them (#136).
GENE_NAMES = ("size", "speed", "sight", "heat_tolerance", "cold_tolerance", "aversion0_0")

GENE_REGISTRY = gene_registry(
    GENE_NAMES,
    {
        "size": 2.0,
        "speed": 3.0,
        "sight": 0.0,
        "heat_tolerance": 1.0,
        "cold_tolerance": 1.0,
        "aversion0_0": 0.0,
    },
)


def make_metabolism(registry=GENE_REGISTRY, **overrides):
    defaults = dict(
        basal_rate=1.0,
        thermoregulation_rate=0.5,
        # Zero here on purpose: `Metabolism` prices a phenotype, and the dehydration multiplier is
        # applied by `Ecology.upkeep` against the column (#156). A non-zero value would put a term
        # in these assertions that this module does not read.
        dehydration_penalty=0.0,
        neutral_temperature=20.0,
        heat_tolerance_gene="heat_tolerance",
        cold_tolerance_gene="cold_tolerance",
    )
    defaults.update(overrides)
    return Metabolism(registry, MetabolismConfig(**defaults))


def expressed(**genes):
    """One (1, n_genes) expressed-phenotype row, named by gene rather than column index."""
    row = np.zeros((1, len(GENE_NAMES)), dtype=np.float32)
    for name, value in genes.items():
        row[0, GENE_NAMES.index(name)] = value
    return row


def upkeep_of(metabolism, temperature=20.0, **genes):
    return float(
        metabolism.upkeep(expressed(**genes), np.array([temperature], dtype=np.float32))[0]
    )


class TestConfigValidation:
    """The config is the whole "no free lunch" contract (CLAUDE.md §2.5), so every way of
    accidentally granting one has to fail at construction rather than at some later population
    explosion nobody traces back to here.
    """

    def test_a_negative_basal_rate_is_rejected(self):
        with pytest.raises(ValueError, match="basal_rate"):
            make_metabolism(basal_rate=-1.0)

    def test_a_negative_thermoregulation_rate_is_rejected(self):
        with pytest.raises(ValueError, match="thermoregulation_rate"):
            make_metabolism(thermoregulation_rate=-0.5)

    @pytest.mark.parametrize("free", ["heat_tolerance", "cold_tolerance"])
    def test_a_tolerance_gene_that_costs_nothing_is_rejected(self, free):
        # A tolerance only ever reduces upkeep, so a free one is unbounded free benefit -- the
        # exact runaway §2.5 forbids. Each side is checked on its own: one free side is enough.
        costs = {"size": 2.0, "speed": 3.0, "heat_tolerance": 1.0, "cold_tolerance": 1.0}
        costs[free] = 0.0
        with pytest.raises(ValueError, match=free):
            make_metabolism(registry=gene_registry(GENE_NAMES, costs))

    @pytest.mark.parametrize("side", ["heat_tolerance_gene", "cold_tolerance_gene"])
    def test_a_tolerance_gene_outside_the_vocabulary_is_rejected(self, side):
        with pytest.raises(KeyError):
            make_metabolism(**{side: "blubber"})

    def test_one_gene_serving_both_sides_is_rejected(self):
        # That is the symmetric damper #176 replaced: hot and cold would select the same build
        # again, and nothing else would notice.
        with pytest.raises(ValueError, match="different genes"):
            make_metabolism(cold_tolerance_gene="heat_tolerance")


class TestOnlyMagnitudeGenesMayCost:
    """A cost is only bounded below by zero if the phenotype it multiplies is (#136).

    Storage is signed (#104), and what makes an expressed value non-negative is the gene's
    expression mode, not inheritance. So a `SIGNED` gene — a cue-space direction, founded across
    zero by design — multiplied by a positive cost contributes a **negative** term to upkeep.

    That is not a crash waiting to happen so much as a subsidy: upkeep is a sum, so a negative
    term merely discounts the total, and the animal is charged less for holding its aversion one
    way round than the other. Selection acts on the discount, which is the free lunch §2.5's hard
    budget exists to forbid, running in reverse. Only when the discount exceeds everything else
    does the total go negative, and `Ecology.spend` rejects that — mid-tick, naming a module that
    did nothing wrong. Both failures are the same misconfiguration, and it is knowable at
    construction.
    """

    def test_a_signed_gene_costing_nothing_is_accepted(self):
        # §2.5 gives every cue gene a cost of zero, so a config following the document builds.
        assert make_metabolism() is not None

    def test_the_rule_is_about_the_mode_and_not_the_gene_name(self):
        # A quantity read as a magnitude may cost whatever it likes, whatever it is called: it is
        # the mode that guarantees a non-negative phenotype, so the same name flips from rejected
        # to accepted on the mode alone. Rejection itself is now the registry's to enforce, since
        # that is where a cost and a mode first meet (#111) -- see tests/core/genetics/test_registry.
        costed_magnitude = GeneRegistry(
            tuple(
                GeneSpec(name, cost, ExpressionMode.MAGNITUDE, Unit.DIMENSIONLESS, f"gene {name}")
                for name, cost in (
                    ("size", 2.0), ("speed", 3.0), ("sight", 0.0),
                    ("heat_tolerance", 1.0), ("cold_tolerance", 1.0), ("aversion0_0", 4.0),
                )
            )
        )
        assert upkeep_of(make_metabolism(registry=costed_magnitude), aversion0_0=1.0) == pytest.approx(5.0)


class TestTraitUpkeep:
    def test_upkeep_is_basal_when_nothing_is_expressed_at_neutral_temperature(self):
        metabolism = make_metabolism()
        assert upkeep_of(metabolism) == pytest.approx(1.0)

    def test_each_expressed_gene_charges_its_declared_cost_per_unit(self):
        metabolism = make_metabolism()
        # basal 1.0 + size 2.0*2 + speed 3.0*1
        assert upkeep_of(metabolism, size=2.0, speed=1.0) == pytest.approx(8.0)

    def test_raising_a_trait_raises_upkeep(self):
        metabolism = make_metabolism()
        cheap = upkeep_of(metabolism, speed=1.0)
        costly = upkeep_of(metabolism, speed=4.0)
        assert costly > cheap

    def test_a_zero_cost_gene_charges_nothing(self):
        metabolism = make_metabolism()
        assert upkeep_of(metabolism, sight=10.0) == upkeep_of(metabolism)

    def test_upkeep_is_one_value_per_row(self):
        metabolism = make_metabolism()
        rows = np.zeros((5, len(GENE_NAMES)), dtype=np.float32)
        temperature = np.full(5, 20.0, dtype=np.float32)

        upkeep = metabolism.upkeep(rows, temperature)

        assert upkeep.shape == (5,)
        assert upkeep.dtype == np.float32


class TestThermoregulation:
    def test_costs_nothing_at_the_neutral_temperature(self):
        metabolism = make_metabolism()
        assert upkeep_of(metabolism, temperature=20.0) == pytest.approx(1.0)

    def test_an_untolerant_body_pays_the_same_either_side(self):
        metabolism = make_metabolism()
        neutral = upkeep_of(metabolism, temperature=20.0)
        cold = upkeep_of(metabolism, temperature=0.0)
        hot = upkeep_of(metabolism, temperature=40.0)

        assert cold > neutral
        assert hot > neutral
        # With no tolerance on either side, distance is all that is charged.
        assert cold == pytest.approx(hot)

    def test_cost_is_quadratic_in_the_deviation(self):
        # Gentle near neutral so variation in the tolerances survives, steep far out so a stranded
        # lineage is pulled back hard (#176's second gate). Twice as far costs four times as much.
        metabolism = make_metabolism()
        near = upkeep_of(metabolism, temperature=25.0) - 1.0
        far = upkeep_of(metabolism, temperature=30.0) - 1.0
        assert near == pytest.approx(0.5 * 25.0)
        assert far == pytest.approx(4.0 * near)

    def test_tolerance_damps_linearly_outside_the_square(self):
        # The trap #176 names: damping the deviation *inside* the square would give tolerance
        # quadratic returns, and a gene that only reduces a cost must have diminishing ones (§2.5).
        # So (1 + tolerance) divides the squared bill: tolerance 1 halves it, 3 quarters it.
        metabolism = make_metabolism()
        bare = upkeep_of(metabolism, temperature=30.0) - 1.0
        one = upkeep_of(metabolism, temperature=30.0, heat_tolerance=1.0) - 1.0 - 1.0
        three = upkeep_of(metabolism, temperature=30.0, heat_tolerance=3.0) - 1.0 - 3.0
        assert one == pytest.approx(bare / 2.0)
        assert three == pytest.approx(bare / 4.0)

    @pytest.mark.parametrize(
        "gene, helps_at, useless_at",
        [("heat_tolerance", 40.0, 0.0), ("cold_tolerance", 0.0, 40.0)],
    )
    def test_each_tolerance_damps_only_its_own_side(self, gene, helps_at, useless_at):
        metabolism = make_metabolism()
        # 3 units cost 3.0 per tick. On its own side they must save more than that; on the other
        # side they save nothing and are dead weight.
        assert upkeep_of(metabolism, helps_at, **{gene: 3.0}) < upkeep_of(metabolism, helps_at)
        assert upkeep_of(metabolism, useless_at, **{gene: 3.0}) == pytest.approx(
            upkeep_of(metabolism, useless_at) + 3.0
        )

    @pytest.mark.parametrize("gene", ["heat_tolerance", "cold_tolerance"])
    def test_tolerance_is_pure_cost_at_the_neutral_temperature(self, gene):
        metabolism = make_metabolism()
        bare = upkeep_of(metabolism, temperature=20.0)
        tolerant = upkeep_of(metabolism, temperature=20.0, **{gene: 3.0})
        assert tolerant == pytest.approx(bare + 3.0)


class TestHotAndColdSelectDifferentBuilds:
    """#176's evidence, inverted. Under the symmetric `insulation` damper, upkeep swept over the
    gene at 5 and at 35 -- two cells equidistant from a neutral of 20 -- gave **identical curves**,
    so an arctic and a desert lineage evolved toward the same animal and climate gave #16 nothing
    to diverge on. With a tolerance per side, the cheapest build in each cell is the mirror of the
    other: it carries the tolerance its own climate charges for, and none of the other one.
    """

    SWEEP = np.linspace(0.0, 30.0, 61, dtype=np.float32)

    def cheapest_build(self, temperature):
        metabolism = make_metabolism(thermoregulation_rate=0.1)
        heat, cold = np.meshgrid(self.SWEEP, self.SWEEP, indexing="ij")
        rows = np.zeros((heat.size, len(GENE_NAMES)), dtype=np.float32)
        rows[:, GENE_NAMES.index("heat_tolerance")] = heat.ravel()
        rows[:, GENE_NAMES.index("cold_tolerance")] = cold.ravel()
        upkeep = metabolism.upkeep(rows, np.full(heat.size, temperature, dtype=np.float32))
        best = int(np.argmin(upkeep))
        return float(heat.ravel()[best]), float(cold.ravel()[best])

    def test_a_hot_cell_and_a_cold_cell_favour_different_builds(self):
        hot_heat, hot_cold = self.cheapest_build(35.0)
        cold_heat, cold_cold = self.cheapest_build(5.0)

        assert (hot_heat, hot_cold) != (cold_heat, cold_cold)
        # Each carries only the tolerance its own side charges for.
        assert hot_heat > 0.0 and hot_cold == 0.0
        assert cold_cold > 0.0 and cold_heat == 0.0
        # Equidistant and equally costed, so one build is exactly the other mirrored.
        assert hot_heat == pytest.approx(cold_cold)


class TestNoEnergyIsCreated:
    def test_upkeep_is_never_negative_across_a_wide_sweep_of_phenotypes(self):
        metabolism = make_metabolism()
        rng = np.random.default_rng(7)
        rows = rng.uniform(0.0, 20.0, size=(500, len(GENE_NAMES))).astype(np.float32)
        temperature = rng.uniform(-40.0, 60.0, size=500).astype(np.float32)

        assert (metabolism.upkeep(rows, temperature) >= 0.0).all()

    def test_upkeep_is_at_least_the_basal_rate(self):
        metabolism = make_metabolism()
        rng = np.random.default_rng(11)
        rows = rng.uniform(0.0, 20.0, size=(200, len(GENE_NAMES))).astype(np.float32)
        temperature = rng.uniform(-40.0, 60.0, size=200).astype(np.float32)

        assert (metabolism.upkeep(rows, temperature) >= 1.0).all()

    def test_upkeep_holds_when_signed_phenotypes_are_negative(self):
        """The sweeps above draw phenotypes from `[0, 20]`, which is why neither of them could
        ever have caught #136: a real expressed phenotype is *not* non-negative any more. Cue
        genes are read `SIGNED` and arrive as stored, so this draws across zero — and what keeps
        the total non-negative is that the constructor refused to cost those columns.
        """
        metabolism = make_metabolism()
        rng = np.random.default_rng(13)
        rows = rng.uniform(0.0, 20.0, size=(500, len(GENE_NAMES))).astype(np.float32)
        rows[:, GENE_NAMES.index("aversion0_0")] = rng.uniform(-20.0, 20.0, size=500)
        temperature = rng.uniform(-40.0, 60.0, size=500).astype(np.float32)

        assert (metabolism.upkeep(rows, temperature) >= 0.0).all()
