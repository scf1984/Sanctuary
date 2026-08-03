"""Dead flesh on the ground: deposited, contested, eaten, rotted (#185, #179).

The conservation group is the one that matters. Carrion is the third place a nutrient can be, and
the whole risk of adding it is double-entry — paying the ledger back when a body falls *and* again
when it rots would create nutrients out of a corpse (§6).
"""

import numpy as np
import pytest

from core.ecology.carrion import Carrion, CarrionConfig
from core.ecology.plants import Plants, PlantsConfig
from core.world.climate import Climate, ClimateConfig
from core.world.diffusion import DiffusionConfig
from core.world.terrain import Terrain
from core.world.water import Water


PLANTS_CONFIG = PlantsConfig(
    solar_constant=10.0,
    latitude_tilt=0.0,
    min_growth_temperature=0.0,
    optimal_growth_temperature=25.0,
    max_growth_temperature=45.0,
    nutrient_per_biomass=0.1,
    initial_soil_nutrients=100.0,
    senescence_rate=0.05,
    saturation_accumulation=50.0,
    max_rooting_depth=0.5,
    forage_diffusion=DiffusionConfig(range=4.0, climb_penalty=0.5),
)


def field(decay_rate=0.1, grid=11):
    """A flat world with a plant field and a carrion field over it."""
    terrain = Terrain(np.zeros((grid, grid), dtype=np.float32), cell_size=1.0)
    climate = Climate(
        terrain,
        ClimateConfig(equator_y=0.0, equator_temperature=25.0, latitude_gradient=0.0),
    )
    plants = Plants(terrain, climate, Water.generate(terrain), PLANTS_CONFIG)
    # Carrion cannot exist without the energy in it having been on the export ledger first — it was
    # in a living animal a moment ago. These tests deposit bodies directly rather than growing and
    # killing animals to make them, so the founding stock those animals would have carried is
    # recorded here. Without it `decompose` rightly refuses to return what never left (§6).
    plants.record_founding_stock(10_000.0)
    return plants, Carrion(terrain, plants, CarrionConfig(decay_rate=decay_rate))


def at(*positions):
    xs = np.array([p[0] for p in positions], dtype=np.float64)
    ys = np.array([p[1] for p in positions], dtype=np.float64)
    return xs, ys


class TestABodyLandsWhereItFell:
    def test_deposited_mass_is_on_the_ground(self):
        _, carrion = field()

        carrion.deposit(*at((3.0, 4.0)), np.array([25.0]))

        assert carrion.mass.sum() == pytest.approx(25.0)
        assert carrion.mass[4, 3] == pytest.approx(25.0)

    def test_two_bodies_in_one_cell_accumulate(self):
        """`np.add.at` rather than fancy-index assignment: the second body must not replace the
        first, which is the classic silent-loss shape of a repeated index."""
        _, carrion = field()

        carrion.deposit(*at((3.0, 4.0), (3.2, 4.1)), np.array([10.0, 15.0]))

        assert carrion.mass.sum() == pytest.approx(25.0)

    def test_nothing_un_dies(self):
        _, carrion = field()

        with pytest.raises(ValueError, match="non-negative"):
            carrion.deposit(*at((3.0, 4.0)), np.array([-1.0]))


class TestNutrientsAreNeitherLostNorDoubled:
    """The reason this field cannot simply pay the ledger when a body falls."""

    def test_a_falling_body_moves_no_ledger(self):
        """It was outstanding while it sat in the animal's pool and it is outstanding while it lies
        on the ground. Crediting here as well as on decay would return the same nutrients twice."""
        plants, carrion = field()
        before = plants.total_nutrients()

        carrion.deposit(*at((3.0, 4.0)), np.array([30.0]))

        assert plants.total_nutrients() == pytest.approx(before)

    def test_rotting_pays_the_ledger_back_exactly_once(self):
        plants, carrion = field(decay_rate=1.0)
        carrion.deposit(*at((3.0, 4.0)), np.array([30.0]))
        outstanding = plants.exported_nutrients

        carrion.decompose()

        assert carrion.mass.sum() == pytest.approx(0.0)
        assert plants.exported_nutrients == pytest.approx(
            outstanding - 30.0 * PLANTS_CONFIG.nutrient_per_biomass
        )

    def test_the_total_never_moves_across_a_whole_cycle(self):
        """Deposit, part-scavenge, rot — the sequence a real kill goes through (§6)."""
        plants, carrion = field()
        before = plants.total_nutrients()

        carrion.deposit(*at((3.0, 4.0)), np.array([40.0]))
        carrion.graze(*at((3.0, 4.0)), np.array([10.0]))
        # What a scavenger took is in its pool now, which this fixture has no animals to hold — so
        # it is returned here exactly as `Feeding` returns the undigested half.
        plants.return_nutrients(*at((3.0, 4.0)), np.array([10.0]))
        for _ in range(50):
            carrion.decompose()

        assert plants.total_nutrients() == pytest.approx(before, rel=1e-9)


class TestEatingItIsGrazing:
    def test_a_scavenger_takes_what_it_asks_for_when_there_is_enough(self):
        _, carrion = field()
        carrion.deposit(*at((3.0, 4.0)), np.array([50.0]))

        taken = carrion.graze(*at((3.0, 4.0)), np.array([12.0]))

        assert taken[0] == pytest.approx(12.0)
        assert carrion.mass.sum() == pytest.approx(38.0)

    def test_a_carcass_is_contested_by_fraction_of_demand(self):
        """The identical rule `Plants.graze` uses, and what makes a kill worth standing on: two
        scavengers on one body split it in proportion to appetite rather than each seeing all."""
        _, carrion = field()
        carrion.deposit(*at((3.0, 4.0)), np.array([9.0]))

        taken = carrion.graze(*at((3.0, 4.0), (3.1, 4.1)), np.array([6.0, 12.0]))

        assert taken.sum() == pytest.approx(9.0)
        assert taken[1] == pytest.approx(2.0 * taken[0])
        assert carrion.mass.sum() == pytest.approx(0.0)

    def test_bare_ground_yields_nothing_rather_than_raising(self):
        """A herbivore's flesh mouthful is tiny but non-zero, so this is the ordinary path for most
        of the population every tick — not an edge case."""
        _, carrion = field()

        assert carrion.graze(*at((3.0, 4.0)), np.array([5.0]))[0] == pytest.approx(0.0)


class TestRotIsAFractionRatherThanAnAmount:
    def test_every_carcass_has_the_same_half_life_whatever_its_size(self):
        """A fixed subtraction would make a large body last proportionally longer, which is
        backwards — rot works on surface, not on volume (#107's argument, reused)."""
        _, carrion = field(decay_rate=0.2)
        carrion.deposit(*at((1.0, 1.0), (5.0, 5.0)), np.array([100.0, 10.0]))

        carrion.decompose()

        assert carrion.mass[1, 1] / 100.0 == pytest.approx(carrion.mass[5, 5] / 10.0)

    def test_a_carcass_approaches_nothing_without_ever_going_negative(self):
        _, carrion = field(decay_rate=0.5)
        carrion.deposit(*at((3.0, 4.0)), np.array([10.0]))

        for _ in range(200):
            carrion.decompose()

        assert 0.0 <= carrion.mass.sum() < 1e-6

    def test_empty_ground_decomposes_without_touching_the_ledger(self):
        plants, carrion = field()
        before = plants.exported_nutrients

        carrion.decompose()

        assert plants.exported_nutrients == pytest.approx(before)

    @pytest.mark.parametrize("rate", [0.0, -0.1, 1.5])
    def test_an_impossible_decay_rate_is_refused(self, rate):
        with pytest.raises(ValueError, match="decay_rate"):
            CarrionConfig(decay_rate=rate)


class TestACarcassAdvertisesItself:
    def test_meat_is_smelled_from_a_cell_away(self):
        """Without this a scavenger could only find a body it was already standing on, and the
        measurement said so: 12,000 energy units of meat that nobody ever came for."""
        _, carrion = field()
        carrion.deposit(*at((4.0, 4.0)), np.array([100.0]))

        carrion.rebuild_scent()

        assert carrion.scent[4, 5] > 0.0
        assert carrion.scent[4, 4] > carrion.scent[4, 6]

    def test_bare_ground_smells_of_nothing(self):
        _, carrion = field()

        carrion.rebuild_scent()

        assert carrion.scent.sum() == pytest.approx(0.0)
