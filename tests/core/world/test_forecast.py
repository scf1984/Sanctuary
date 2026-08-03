"""What a set of tunings implies, before anything is run (#216).

Most of this is exact arithmetic over a built world and is asserted exactly. The last class is the
one that matters and it is statistical (§6): a forward model whose error against a real world was
never measured is an estimate wearing the costume of a formula.
"""

import dataclasses

import pytest

from clients.viewer.demo_world import build_demo_world, demo_world_config
from core.selection import Selection
from core.world.assembly import build_world
from core.world.forecast import CAPTURE_FRACTION, forecast


def world(seed=1, founders=60, **plants):
    config = demo_world_config(founders, seed)
    if plants:
        config = dataclasses.replace(
            config, plants=dataclasses.replace(config.plants, **plants)
        )
    return build_world(config, seed=seed)


class TestItReadsTheWorldRatherThanRestatingIt:
    def test_field_growth_is_the_real_potential_growth_field(self):
        """Not a re-derived growth curve. A second copy of that arithmetic is one that drifts from
        the field it is meant to describe (§4)."""
        built = world()

        assert forecast(built).field_growth == pytest.approx(
            float(built.plants.potential_growth.sum())
        )

    def test_upkeep_is_what_the_ecology_would_actually_charge(self):
        built = world()
        living = Selection.from_mask(built.store.alive & (built.store.age >= 0))

        assert forecast(built).upkeep_per_animal == pytest.approx(
            float(built.ecology.upkeep(living).mean()), rel=1e-6
        )

    def test_generation_length_is_read_off_the_founders_not_the_config_ranges(self):
        """Both genes are magnitudes (§2.5), and the mean of `abs` over a range crossing zero is
        not the mean of the range. Reading the population reads what the world actually has."""
        built = world()
        living = Selection.from_mask(built.store.alive & (built.store.age >= 0))
        expressed = built.genetics.expressed(living)
        maturity = expressed[:, built.genes.index_of("maturity_age")].mean()
        gestation = expressed[:, built.genes.index_of("gestation_length")].mean()

        assert forecast(built).generation_ticks == pytest.approx(
            maturity + gestation, rel=1e-5
        )


class TestTheRelationItEncodes:
    def test_doubling_the_sun_doubles_the_capacity(self):
        """The whole claim, asserted without running anything: capacity is proportional to what the
        field grows, and the sun is what scales that."""
        dim = forecast(world(solar_constant=4.0)).carrying_capacity
        bright = forecast(world(solar_constant=8.0)).carrying_capacity

        assert bright == pytest.approx(2.0 * dim, rel=0.02)

    def test_a_costlier_animal_means_fewer_of_them(self):
        """The other half of the relation. Doubling what one animal costs to keep alive halves how
        many the same field supports."""
        cheap = world()
        config = cheap.config
        costly = build_world(
            dataclasses.replace(
                config,
                metabolism=dataclasses.replace(
                    config.metabolism, basal_rate=config.metabolism.basal_rate * 4
                ),
            ),
            seed=1,
        )

        assert forecast(costly).carrying_capacity < forecast(cheap).carrying_capacity

    def test_the_estimated_term_is_reported_rather_than_folded_away(self):
        """A reader has to be able to see which part of the answer is measured rather than derived,
        which is the difference between a forecast and a formula."""
        reported = forecast(world())

        assert reported.capture_fraction == CAPTURE_FRACTION
        assert reported.carrying_capacity == pytest.approx(
            reported.field_growth
            * reported.capture_fraction
            * reported.assimilation_share
            / reported.upkeep_per_animal
        )

    def test_a_stated_capture_fraction_overrides_the_measured_one(self):
        """A world whose foraging differs sharply from the one the constant was measured on can
        say so, rather than being silently forecast against somebody else's world."""
        assert forecast(world(), capture_fraction=0.31).carrying_capacity == pytest.approx(
            forecast(world()).carrying_capacity / 2, rel=1e-6
        )


class TestItRefusesWhatItCannotAnswer:
    def test_a_world_with_no_animals_is_refused(self):
        """Every term is a property of a population; a forecast over nobody is a number with no
        referent (§8.7)."""
        built = world()
        built.store.release(built.store.row_ids()[built.store.alive])

        with pytest.raises(ValueError, match="no living animals"):
            forecast(built)

    @pytest.mark.parametrize("capture", [0.0, -0.1, 1.5])
    def test_an_impossible_capture_fraction_is_refused(self, capture):
        with pytest.raises(ValueError, match="capture_fraction"):
            forecast(world(), capture_fraction=capture)


class TestHowCloseItGets:
    """What the forecast is still true about, and what predation took away from it.

    The model was measured on a world of pure grazers and landed within 5% (`docs/spikes/
    forecast_accuracy.py`). #179 broke that, and not by a little: with predation running, a settled
    world reaches **about a tenth** of the forecast — 521 living against 5,361 at 700 ticks.

    The reason is structural rather than a mis-tuned coefficient. The forecast divides the field's
    production by what one animal costs to keep alive, which assumes every captured unit of energy
    ends up paying upkeep. Predation opens a second exit: a strike moves energy out of an animal
    into carrion, and whatever is not scavenged before it rots goes back to the soil having fed
    nobody. The whole of that drain is missing from the model.

    Fixing it belongs to #216, which owns the forecast (§7.2), so what is asserted here is the half
    that survives: **it is a ceiling, and a world stays under it.** That is still worth pinning —
    a world exceeding its own production forecast would mean energy arriving from somewhere the
    ledger does not know about.
    """

    def test_a_settled_world_stays_under_its_forecast(self):
        """Two hundred founders and seven hundred ticks, which is what the spike measured against.

        The lower bound is deliberately far below the observed tenth: this is a guard against the
        model inverting, not a pin on a number that #216 is expected to move once the predation
        drain is in it (§8.1).
        """
        built = build_demo_world(seed=1, n_entities=200)
        predicted = forecast(built).carrying_capacity

        built.loop.advance(700)
        actual = len(Selection.from_mask(built.store.alive & (built.store.age >= 0)))

        assert 0.0 < actual / predicted < 1.0, (
            f"forecast {predicted:.0f} against {actual} living: a world above its own production "
            "forecast is energy from outside the ledger"
        )
