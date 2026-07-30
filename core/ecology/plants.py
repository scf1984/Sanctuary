"""Plants: the world's only energy income, drawn against a finite soil-nutrient pool (#18).

Sunlight is the sole way energy enters the simulation (CLAUDE.md §2.5). Everything downstream —
what a herbivore can eat, how many herbivores a valley supports, what a predator can then support —
is a share of what this module produces. Carrying capacity is therefore never configured: it is
`area × primary productivity ÷ per-animal upkeep` (§2.3), and the first of those three terms lives
here.

**Plants are a field, not entities.** The issue that owns this abstraction required the choice be
recorded either way, so: growth is a per-cell quantity over the terrain grid, advanced as a handful
of whole-array operations, and there is no plant row in the entity store. Individual plants would
buy nothing the ecology needs — a herbivore grazes a patch, not a numbered shrub — while costing an
entity row and a genome each, on a population that would outnumber the animals by orders of
magnitude. The one thing the field model must still express is *local* competition, and it does:
grazing depletes the cell it happens in, and `graze` shares a contested cell between the grazers
standing in it. If some future mechanic genuinely needs plant identity (a fruiting tree an animal
returns to), that is a new issue, not a retrofit of this one.

**Two conserved quantities, tracked separately.** Energy is not conserved — it enters as sunlight
and leaves as metabolism (`core.ecology.service.Ecology`). Nutrients *are*: `total_nutrients()` is
constant for the lifetime of the field. Growth moves nutrients from soil into standing biomass,
senescence returns them, and grazing carries them out of the field entirely — held on the
`exported_nutrients` ledger, which is what #21's decomposition will eventually pay back. That
ledger is not bookkeeping for its own sake: without it, "nutrients are conserved" is unassertable
the moment anything eats, and §2.5's closure claim becomes untestable exactly when it starts to
matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.world.climate import Climate
from core.world.diffusion import CostAwareDiffusion, DiffusionConfig
from core.world.terrain import Terrain, bilinear_sample
from core.world.water import Water


@dataclass(frozen=True)
class PlantsConfig:
    """Per-world primary-productivity table (CLAUDE.md §2.1: tuned as a table, not as scattered
    literals, and never as constants inside `core/`).

    solar_constant: energy units of new biomass per cell per tick under a perpendicular sun, at the
        optimal temperature and saturated soil. Already net of photosynthetic efficiency — there
        is no separate efficiency term, because nothing in the simulation can observe incident
        light and converted biomass separately, and two coefficients multiplied together are two
        things to drift apart.
    latitude_tilt: radians of solar incidence added per world unit of distance from the climate's
        equator line. Larger values pack more climate into less world.
    min_growth_temperature, optimal_growth_temperature, max_growth_temperature: degrees C. Growth
        is zero at and beyond the two extremes and peaks at the optimum, rising and falling
        linearly between them.
    nutrient_per_biomass: nutrient units bound up in one energy unit of standing biomass. This is
        the exchange rate between the two conserved quantities, so it is also what converts a soil
        pool into a ceiling on standing crop.
    initial_soil_nutrients: nutrient units per cell at world creation. Since nutrients are
        conserved, this is the world's entire nutrient budget for all time — the single number
        that decides how much life the world can hold at once.
    senescence_rate: fraction of standing biomass that dies per tick, returning its nutrients to
        the cell's soil. Must be positive: without it nothing returns to the soil, growth runs
        until the soil is stripped bare, and the loop is open rather than closed.
    saturation_accumulation: flow accumulation (in cells draining through) at which soil moisture
        is saturated. Cells with less drainage above them are proportionally drier.
    max_rooting_depth: world units of standing water a cell can hold before its plants drown, the
        unit `Water.depth` reports (#112). Without
        this cutoff, lakes — which collect every cell's drainage — would be the most productive
        ground in the world, and every herbivore would live in one.
    forage_diffusion: how far standing crop makes itself known, and how much relief impedes that
        (#93, `core.world.diffusion`). Its `range` is the distance discount a forager applies —
        small values keep grazers local and strip ground bare before they move on, large ones
        spread grazing pressure out — and its `climb_penalty` is what makes a meadow behind a ridge
        less attractive than one on open ground the same distance away.
    """

    solar_constant: float
    latitude_tilt: float
    min_growth_temperature: float
    optimal_growth_temperature: float
    max_growth_temperature: float
    nutrient_per_biomass: float
    initial_soil_nutrients: float
    senescence_rate: float
    saturation_accumulation: float
    max_rooting_depth: float
    forage_diffusion: DiffusionConfig

    def __post_init__(self) -> None:
        if self.solar_constant < 0:
            raise ValueError(f"solar_constant must be non-negative, got {self.solar_constant}")
        if self.latitude_tilt < 0:
            raise ValueError(f"latitude_tilt must be non-negative, got {self.latitude_tilt}")
        if self.nutrient_per_biomass <= 0:
            raise ValueError(
                f"nutrient_per_biomass must be positive, got {self.nutrient_per_biomass}"
            )
        if self.initial_soil_nutrients < 0:
            raise ValueError(
                f"initial_soil_nutrients must be non-negative, got {self.initial_soil_nutrients}"
            )
        if not 0.0 < self.senescence_rate <= 1.0:
            raise ValueError(
                f"senescence_rate must be in (0, 1], got {self.senescence_rate}; see the config "
                "docstring — a zero rate leaves the nutrient loop open"
            )
        if self.saturation_accumulation <= 0:
            raise ValueError(
                f"saturation_accumulation must be positive, got {self.saturation_accumulation}"
            )
        if self.max_rooting_depth < 0:
            raise ValueError(
                f"max_rooting_depth must be non-negative, got {self.max_rooting_depth}"
            )
        if not (
            self.min_growth_temperature
            < self.optimal_growth_temperature
            < self.max_growth_temperature
        ):
            raise ValueError(
                "growth temperatures must satisfy min < optimal < max, got "
                f"{self.min_growth_temperature} < {self.optimal_growth_temperature} < "
                f"{self.max_growth_temperature}"
            )


class Plants:
    """The world's plant field: standing biomass and the soil nutrients it is built from.

    biomass:          (height, width) float64, energy units. Standing crop per cell — what a
                      grazer eats.
    soil_nutrients:   (height, width) float64, nutrient units. The unbound pool growth draws on.
    potential_growth: (height, width) float64, energy units per cell per tick. Light-,
                      temperature- and moisture-limited gain, before the soil is consulted.
                      Static: it is a pure function of terrain, climate and water, all of which are
                      themselves static between terraforming interventions, so it is computed once
                      here rather than per tick.
    moisture:         (height, width) float64, dimensionless in [0, 1]. Soil wetness, already
                      folded into `potential_growth`; kept as its own attribute because the
                      diagnostic viewer's overlays (§3.3) need to show *why* a region is barren,
                      and "dry" and "too cold" are indistinguishable in the product.
    exported_nutrients: scalar float, nutrient units. Nutrients grazed out of the field and not
                      yet returned — see the module docstring.

    All four grid fields align cell-for-cell with `terrain.heights`.

    Unusually for this codebase the mutable fields are float64 rather than float32. §2.3's case
    for float32 is entity arrays, where the row count is the thing being scaled; there are far
    fewer cells than entities and they are not touched per-entity. Meanwhile `total_nutrients()`
    is a §6 conservation invariant checked every tick over a world that runs for weeks, and
    float32's ~7 significant digits lose a per-tick change of 1e-3 against a pool of 1e4 outright
    — the invariant would drift and then trip on arithmetic rather than on a bug.

    This is deliberately not a `DomainService`: `core.services.ColumnRegistry` arbitrates
    ownership of *entity store columns*, and plants own no entity rows at all. The ownership rule
    it exists to enforce still holds here by construction — nothing outside this class writes
    these arrays.
    """

    # Declared rather than left to inference: `np.zeros(shape)` with a statically unknown shape
    # resolves to the stubs' 1-D overload, which then reports every 2-D index as out of range.
    biomass: np.ndarray
    soil_nutrients: np.ndarray
    potential_growth: np.ndarray
    moisture: np.ndarray
    exported_nutrients: float

    def __init__(
        self, terrain: Terrain, climate: Climate, water: Water, config: PlantsConfig
    ) -> None:
        if water.depth.shape != terrain.heights.shape:
            raise ValueError("water and terrain must share a grid shape")

        self.terrain = terrain
        self.climate = climate
        self.water = water
        self.config = config

        self.moisture = _moisture(water, config)
        self.potential_growth = (
            config.solar_constant
            * _insolation(terrain, climate, config)
            * _temperature_response(climate, config)
            * self.moisture
        )
        self.forage_diffusion = CostAwareDiffusion(terrain, config.forage_diffusion)
        self.biomass = np.zeros(terrain.heights.shape, dtype=np.float64)
        self.soil_nutrients = np.full(
            terrain.heights.shape, config.initial_soil_nutrients, dtype=np.float64
        )
        self.exported_nutrients = 0.0

    def grow(self) -> None:
        """Advance the field one tick: senescence, then nutrient-limited growth.

        Senescence runs first so the nutrients it frees are available to the same tick's growth.
        Ordering it the other way would make a cell's productivity depend on a pool that is one
        tick stale, which shows up as an oscillation in tightly nutrient-limited ground rather
        than as a settled equilibrium.

        Growth is `min(light-limited potential, what the soil can pay for)`. Nothing caps standing
        biomass directly: the cap is the cell's nutrient budget, and the equilibrium is wherever
        senescence losses meet the gain, so carrying capacity falls out of the terrain rather than
        being authored (§2.3).
        """
        senesced = self.biomass * self.config.senescence_rate
        self.biomass -= senesced
        self.soil_nutrients += senesced * self.config.nutrient_per_biomass

        affordable = self.soil_nutrients / self.config.nutrient_per_biomass
        growth = np.minimum(self.potential_growth, affordable)
        self.biomass += growth
        self.soil_nutrients -= growth * self.config.nutrient_per_biomass

    def biomass_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """(n,) float64, energy units: standing biomass in the cell containing each world position.

        Sampled to the containing cell rather than bilinearly interpolated, unlike every other
        field query in `core/world`. Those fields are read-only; this one is grazed. An
        interpolated read would promise a grazer biomass drawn from four cells while `graze`
        depletes one, so the amount an animal can see and the amount it can eat would disagree.
        """
        rows, cols = self._cell_indices(x, y)
        return self.biomass[rows, cols]

    def graze(self, x: np.ndarray, y: np.ndarray, demand: np.ndarray) -> np.ndarray:
        """Harvest up to `demand` biomass at each world position; return what was taken.

        x, y:   (n,) world units — where each grazer is standing.
        demand: (n,) energy units — what each grazer would eat if the cell could supply it.
        returns (n,) float64 energy units, one entry per grazer, in the order given.

        Grazers sharing a cell contend for it: when total demand exceeds the standing crop, each
        takes the same *fraction* of what it asked for, so the cell empties exactly and a hungrier
        animal still gets proportionally more. This — not the biomass field itself — is what makes
        space matter to a herbivore, since spreading out is the only way to avoid the split.

        The harvested nutrients leave the field onto `exported_nutrients`; the caller (#19
        feeding) is receiving the *energy*, and #21 is what will return the nutrients.
        """
        demand = np.asarray(demand, dtype=np.float64)
        if np.any(demand < 0):
            raise ValueError("grazing demand must be non-negative")

        rows, cols = self._cell_indices(x, y)
        n_cells = self.biomass.size
        flat_cell = rows * self.biomass.shape[1] + cols

        # Aggregate demand per cell first: without this, each grazer resolved independently would
        # see the full standing crop and n grazers would harvest n times what the cell holds.
        demand_per_cell = np.zeros(n_cells, dtype=np.float64)
        np.add.at(demand_per_cell, flat_cell, demand)
        contested = demand_per_cell[flat_cell]
        standing = self.biomass.reshape(-1)[flat_cell]
        share = np.where(contested > 0.0, np.minimum(1.0, standing / contested), 0.0)
        harvested = demand * share

        removed = np.zeros(n_cells, dtype=np.float64)
        np.add.at(removed, flat_cell, harvested)
        self.biomass -= removed.reshape(self.biomass.shape)
        # Clipped at zero only to absorb the float rounding of summing each cell's harvest twice
        # (once as `harvested`, once as `removed`); `share <= 1` already bounds the real quantity.
        np.maximum(self.biomass, 0.0, out=self.biomass)
        # Ledgered from `removed`, not from `harvested`, so what leaves the biomass array and what
        # is recorded as owed are the same number to the last bit and conservation holds exactly.
        self.exported_nutrients += float(removed.sum()) * self.config.nutrient_per_biomass
        return harvested

    def forage_field(self) -> np.ndarray:
        """``(height, width)`` float32: how much grazing is *reachable* from every cell (#93).

        Standing crop diffused over the terrain, so a cell's reading accumulates every meadow
        within range, each discounted by how far away it is and by the climbing between. Rebuilt on
        demand rather than cached, because grazing changes the source every tick and a stale field
        would send foragers to ground that was stripped bare while they walked.

        This is the whole of what a forager can know about food it is not standing on. It replaces
        a per-forager list of candidate patches (the original #93 contract), and the reason is that
        the list could only ever be ranked by *distance*: a meadow across a gorge scored exactly as
        well as one on open ground. Diffusion makes the discount and the terrain the same
        mechanism, so `forage_reluctance` is now this field's `range` rather than a second
        coefficient in a drive (§2.1: tune as a table, not as constants that drift apart).
        """
        return self.forage_diffusion.spread(self.biomass.astype(np.float32))

    def forage_at(self, field, x, y):
        """The forage field's value at each ``(x, y)``, in the field's own units.

        Accepts any shape, because #114 samples a whole block of candidate options at once —
        ``(n_entities, n_options)`` — and `_cell_indices` is elementwise, so one call serves the
        whole population's whole option set rather than one call per option (§2.3).

        Sampled to the containing cell rather than interpolated, matching `biomass_at`: a forager
        that walks there will graze *that* cell, so what it can see and what it can eat agree.
        """
        rows, cols = self._cell_indices(x, y)
        return field[rows, cols]

    def forage_gradient(
        self, field: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Which way food lies, and how strongly, for foragers at ``(x, y)``.

        field: a `forage_field()` result. Passed in rather than rebuilt per call, because one tick
            has one field and every forager reads the same one — building it per drive query would
            pay for the diffusion once per caller.
        returns (gx, gy, strength): the two gradient components, and the field's own value where
            each forager stands, all ``(n,)`` float64.

        **It ranks nothing and gates nothing.** Whether the reading is strong enough to notice is a
        question about the animal — its sight phenotype against a detection threshold — and belongs
        to the drive that asks (§2.5, the same split scent already uses). The field knows nothing of
        genes, which is exactly why this returns a strength instead of a yes.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError("x and y must be the same length")
        gradient_x, gradient_y = self.forage_diffusion.gradient_at(field, x, y)
        strength = bilinear_sample(
            field, x, y, self.terrain.cell_size, self.terrain.world_width,
            self.terrain.world_height,
        )
        return gradient_x, gradient_y, strength

    def total_nutrients(self) -> float:
        """Every nutrient unit the world contains: in soil, bound in biomass, or grazed away.

        Invariant (CLAUDE.md §6, §2.5): constant for the lifetime of the field. Growth, senescence
        and grazing all move nutrients between these three terms and none of them creates or
        destroys any.
        """
        bound = float(self.biomass.sum()) * self.config.nutrient_per_biomass
        return float(self.soil_nutrients.sum()) + bound + self.exported_nutrients

    def _cell_indices(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Grid (row, col) of the cell containing each world position.

        Raises ValueError outside the world, matching `core.world.terrain.bilinear_sample` — a
        position off the map is a bug in whatever moved the entity, and defaulting it to an edge
        cell would let animals graze a border strip forever (§8.7).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        width = self.terrain.world_width
        height = self.terrain.world_height
        if not (np.all((x >= 0) & (x <= width)) and np.all((y >= 0) & (y <= height))):
            raise ValueError("position outside terrain bounds")

        cell_size = self.terrain.cell_size
        # floor(v + 0.5) rather than np.round: round() is banker's rounding, so a position exactly
        # on a cell boundary would land in different cells depending on the boundary's parity.
        cols = np.floor(x / cell_size + 0.5).astype(np.int64)
        rows = np.floor(y / cell_size + 0.5).astype(np.int64)
        return rows, cols


def _insolation(terrain: Terrain, climate: Climate, config: PlantsConfig) -> np.ndarray:
    """(height, width) float64 in [0, 1]: the fraction of a perpendicular sun each cell receives.

    The sun sits overhead at the climate's equator line and leans toward it elsewhere, so its
    incidence angle grows with latitude; the received fraction is the cosine between the incoming
    rays and the cell's surface normal. That single dot product produces both effects the issue
    asks for at once — flat ground dims with latitude, and a slope tilted toward the equator
    out-produces its opposite face at the same latitude, which is why one side of a ridge greens
    up before the other.

    Cells whose normal faces away from the sun are clipped to zero rather than allowed to go
    negative. This is self-shadowing only; a mountain does not cast a shadow onto the valley
    behind it, which would need the terrain line-of-sight work that #24 owns.
    """
    rows = terrain.heights.shape[0]
    row_y = np.arange(rows, dtype=np.float64) * terrain.cell_size
    # Clamped at the horizon: past a quarter turn the sun has set, and letting the angle run on
    # would bring its cosine back up and make the far pole bright again.
    incidence = np.clip(
        config.latitude_tilt * (row_y - climate.config.equator_y), -np.pi / 2, np.pi / 2
    )

    # Sun direction (x, y, z), pointing from the ground toward the sun. It has no x component:
    # this models a daily average, in which the east-west swing cancels and only the north-south
    # lean survives.
    sun_y = -np.sin(incidence)[:, None]
    sun_z = np.cos(incidence)[:, None]

    # Surface normal from slope and aspect. `aspect` is the *downhill* direction, and a surface
    # normal tilts downhill too (a slope dropping south faces south), so the horizontal part of
    # the normal points along aspect rather than against it.
    slope = terrain.slope.astype(np.float64)
    aspect = terrain.aspect.astype(np.float64)
    normal_y = np.sin(slope) * np.sin(aspect)
    normal_z = np.cos(slope)

    return np.clip(normal_y * sun_y + normal_z * sun_z, 0.0, None)


def _temperature_response(climate: Climate, config: PlantsConfig) -> np.ndarray:
    """(height, width) float64 in [0, 1]: growth rate as a fraction of the optimum.

    A triangular response — linear up from `min_growth_temperature`, linear down to
    `max_growth_temperature`. Deliberately the simplest unimodal curve there is: the real shape is
    a species-specific measured thing, and inventing a sigmoid would imply a precision no
    measurement here supports (§8.5). What matters ecologically is only that growth has a
    tolerated band with a peak inside it, which is what turns the temperature field into
    productivity bands and therefore into climate zones worth migrating between.
    """
    temperature = climate.temperature.astype(np.float64)
    rising = (temperature - config.min_growth_temperature) / (
        config.optimal_growth_temperature - config.min_growth_temperature
    )
    falling = (config.max_growth_temperature - temperature) / (
        config.max_growth_temperature - config.optimal_growth_temperature
    )
    return np.clip(np.minimum(rising, falling), 0.0, 1.0)


def _moisture(water: Water, config: PlantsConfig) -> np.ndarray:
    """(height, width) float64 in [0, 1]: plant-available soil water.

    Derived from flow accumulation — how many cells drain through this one — because that is what
    the heightmap already says about where water goes (§2.6). Ridges shed their water and stay
    dry; valley floors collect it. The response is logarithmic: the difference between a cell
    draining one hectare and ten is large, between a hundred and a thousand it is not, because
    soil saturates.

    Cells under more than `max_rooting_depth` of standing water produce nothing at all — see the
    config docstring for why that cutoff has to exist.
    """
    accumulation = water.flow_accumulation.astype(np.float64)
    wetness = np.log1p(accumulation) / np.log1p(config.saturation_accumulation)
    drowned = water.depth.astype(np.float64) > config.max_rooting_depth
    return np.where(drowned, 0.0, np.clip(wetness, 0.0, 1.0))
