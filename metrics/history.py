"""A world's recorded history: one `Sample` per cadence, in plain serialisable values (#30).

The quantities are chosen so that a reader can answer "why did this population crash" and "is this
population still evolving" without holding a reference to anything inside `core/`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

from core.ecology.plants import Plants
from core.entities.store import EntityStore
from core.genetics.service import Genetics
from core.genetics.vocabulary import GeneVocabulary
from core.selection import Selection


@dataclass(frozen=True)
class MetricsConfig:
    """Per-world recording policy.

    every_n_ticks: how often a sample is taken. Per-world rather than a constant for the reason
        §2.4 gives the wake schedule: the cadence decides how much detail is kept, never how fast
        the world moves, and a soak run (#48) wants a coarser one than a live view does. Must be
        positive — zero would mean a history that never fills.
    history_limit: how many samples to keep, oldest discarded first. A world runs indefinitely
        (§2.1: ~6 real days to a herbivore lifetime, and it keeps running while nobody watches), so
        an unbounded list is a slow leak that only shows up in the deployment that matters. Must be
        positive.

        Discarding rather than downsampling, deliberately: a downsampled history changes what an
        old sample *means* — an average over a widening window — and a client plotting it would be
        drawing two different quantities on one axis without being told. Keeping a fixed window of
        the real thing is honest, and the long view belongs in the metrics store §3.2 puts in
        Postgres rather than in a process's memory.
    """

    every_n_ticks: int
    history_limit: int

    def __post_init__(self) -> None:
        if self.every_n_ticks < 1:
            raise ValueError(
                f"every_n_ticks must be at least 1, got {self.every_n_ticks}; "
                "at zero the history never fills"
            )
        if self.history_limit < 1:
            raise ValueError(
                f"history_limit must be at least 1, got {self.history_limit}; "
                "at zero every sample is discarded as it is taken"
            )


@dataclass(frozen=True)
class Sample:
    """One reading of a world, at one tick. Every field is a plain value (see the package docstring).

    tick: the tick this was taken after. The only clock (§2.1).
    living: entities that are alive **and born** — a gestating row is allocated and carries a
        negative age (#20), so counting `alive` alone would report an unborn young as a member of
        the population.
    gestating: allocated and not yet born. Reported separately rather than folded in, because the
        two move independently and the gap between them is a leading indicator: gestation rising
        while living is flat is a generation arriving.
    conceptions, deaths: how many since the previous sample, **derived rather than reported**. Ids
        are never reused (§2.3), so the number issued between two samples is exactly the number of
        rows allocated, and every allocation is a conception; the shortfall against the change in
        occupancy is exactly the number released. That is why neither `Conception` nor `Death`
        has to know this module exists — a counter in each would be a second place for the truth
        to live, and it would drift the first time something else allocated a row.

        Zero for the first sample of a world, where there is no previous reading to difference
        against. Deaths carry no cause breakdown: starvation is the only one there is until #179
        adds predation, and a breakdown of one is a column that reads as a promise.
    median_energy: energy units. The median rather than the mean because a population is often
        bimodal — a cohort of well-fed adults and a cohort of newly independent young — and a mean
        sits between the two where nothing is.
    standing_biomass, soil_nutrients, exported_nutrients: energy and nutrient units over the whole
        plant field. The three together are §2.5's closed loop made visible: their total is what
        `nutrients_are_conserved` asserts never moves (§6), so a client plotting them sees the loop
        rather than being told about it.
    gene_names: the vocabulary, in column order, so a reader can name the two lists below without
        importing anything from `core/`.
    expressed_mean, expressed_spread: per gene, over the **expressed** phenotype of the living.
        Expressed and not stored, because storage is signed and expression is modal (§2.5) — a mean
        over stored cue genes averages a value spread across zero and means nothing, while a mean
        over an expressed magnitude is the population's actual body.

        **These two are what show evolution happening.** A count cannot: "240 herbivores" reads the
        same before and after selection moves the population's speed by a third. A mean that climbs
        across generations is selection, and a spread that collapses is a population running out of
        the variation it would need to adapt again.
    **There is deliberately no single diversity scalar**, and the reason is measured rather than
    aesthetic. The obvious one — the mean per-gene variance of the expressed phenotype — reads 17.5
    on the demo world at tick 120, of which almost all is `maturity_age` alone: it is founded over
    40–120 ticks while `size` is founded over 0.8–1.2, so a mean over raw variances is one gene's
    variance wearing the name of a population's diversity. That is #193's defect exactly, on the
    genetic *distance* metric, and it wants one scale-free composition rather than two.

    `expressed_spread` is the honest primitive and is what a plot needs anyway — a per-gene series
    cannot be dominated by another gene. §5's "within-species genetic diversity" is therefore
    settled only as far as *the numbers to compute it from*; the composition into one number waits
    on #193, and shipping a misleading scalar in the meantime would be worse than shipping none
    (§8.7). Its two siblings — species richness and Shannon entropy over species abundance — are
    deferred with #16.
    """

    tick: int
    living: int
    gestating: int
    conceptions: int
    deaths: int
    median_energy: float
    standing_biomass: float
    soil_nutrients: float
    exported_nutrients: float
    gene_names: tuple[str, ...]
    expressed_mean: tuple[float, ...]
    expressed_spread: tuple[float, ...]

    def as_dict(self) -> dict:
        """A plain mapping, ready for JSON. What crosses the process boundary §3.1 is heading for."""
        return asdict(self)


class MetricHistory:
    """Records `Sample`s on a cadence and answers questions about the series.

    store, genetics, plants: read, never written. This module owns no column and no field — it is a
        reader, which is what lets it run during offline catch-up without changing anything (§2.4).
    vocabulary: gene names in column order, so a sample can name its own numbers.

    Sampling is **not** a system in `TICK_ORDER`. That tuple is the declared rule for what a tick
    *does* and is frozen into the MAJOR version (§2.1, §2.8); recording a reading changes no
    outcome, so putting it there would make an observation part of the rule set and freeze the
    cadence for the life of a world. `TickLoop` calls it after the invariants instead, on the same
    footing as capacity growth (#127).
    """

    def __init__(
        self,
        store: EntityStore,
        genetics: Genetics,
        plants: Plants,
        vocabulary: GeneVocabulary,
        config: MetricsConfig,
    ) -> None:
        self.store = store
        self.genetics = genetics
        self.plants = plants
        self.vocabulary = vocabulary
        self.config = config
        self.samples: list[Sample] = []
        # What the previous sample saw, so conceptions and deaths are a difference rather than a
        # count somebody else has to keep. `None` until the first sample, which is why that one
        # reports zero for both.
        self._previous: Optional[tuple[int, int]] = None

    def record_if_due(self, tick: int) -> Optional[Sample]:
        """Take a sample if `tick` falls on the cadence, and return it; otherwise `None`.

        The caller passes the tick rather than this holding its own counter, so that a world
        advanced in one batch of a hundred and one advanced in a hundred batches of one record the
        same ticks — §2.4 forbids batching from changing outcomes, and a history that depended on
        how a client chose to call `advance` would be exactly that.
        """
        if tick % self.config.every_n_ticks:
            return None
        sample = self.sample(tick)
        self.samples.append(sample)
        # Oldest first: a client watching a live world wants the recent past, and the long view
        # belongs in the metrics store rather than in a process's memory (see `MetricsConfig`).
        del self.samples[: max(0, len(self.samples) - self.config.history_limit)]
        return sample

    def sample(self, tick: int) -> Sample:
        """Read the world now, without recording it. Pure — nothing in `core/` is written."""
        alive = self.store.alive
        born = alive & (self.store.age >= 0)
        living = Selection.from_mask(born)

        issued = self.store.ids_issued
        occupied = int(alive.sum())
        conceptions, deaths = self._flows(issued, occupied)
        self._previous = (issued, occupied)

        return Sample(
            tick=tick,
            living=len(living),
            gestating=occupied - len(living),
            conceptions=conceptions,
            deaths=deaths,
            median_energy=float(np.median(self.store.energy[born])) if len(living) else 0.0,
            standing_biomass=float(self.plants.biomass.sum()),
            soil_nutrients=float(self.plants.soil_nutrients.sum()),
            exported_nutrients=float(self.plants.exported_nutrients),
            gene_names=self.vocabulary.names,
            **self._phenotype(living),
        )

    def _flows(self, issued: int, occupied: int) -> tuple[int, int]:
        """(conceptions, deaths) since the previous sample, from the two counters alone.

        Ids are never reused, so `issued` rises by exactly one per row allocated and never falls,
        while `occupied` rises with allocation and falls with release. The difference between the
        two changes is therefore exactly the number of rows released — no service reports anything,
        and nothing can disagree.
        """
        if self._previous is None:
            return 0, 0
        previous_issued, previous_occupied = self._previous
        conceptions = issued - previous_issued
        return conceptions, conceptions - (occupied - previous_occupied)

    def _phenotype(self, living: Selection) -> dict:
        """Per-gene mean and spread of the expressed phenotype.

        One `expressed` call for both, because it rebuilds the whole `(n, n_genes)` block and asking
        twice would be a block nobody needed — the same argument #114 makes for sampling candidate
        positions once and sharing them across drives.
        """
        width = len(self.vocabulary.names)
        if not len(living):
            # An empty world has no phenotype rather than a zero one, but a `Sample` is a fixed
            # shape a client plots against, so the columns stay and read zero. This is the one
            # place that distinction is lost, and it is lost in favour of a series that does not
            # change width when the last animal dies.
            return {
                "expressed_mean": (0.0,) * width,
                "expressed_spread": (0.0,) * width,
            }
        expressed = self.genetics.expressed(living).astype(np.float64)
        return {
            "expressed_mean": tuple(expressed.mean(axis=0).tolist()),
            "expressed_spread": tuple(expressed.std(axis=0).tolist()),
        }

    def series(self, name: str) -> list:
        """One field of every recorded sample, oldest first — what a client plots against `ticks`.

        By name rather than by attribute so that a client naming a field that does not exist is
        told so (§8.7), instead of being handed an empty list that reads as "nothing happened".
        """
        if not hasattr(Sample, "__dataclass_fields__") or name not in Sample.__dataclass_fields__:
            raise KeyError(
                f"'{name}' is not a metric; recorded fields are "
                f"{sorted(Sample.__dataclass_fields__)}"
            )
        return [getattr(sample, name) for sample in self.samples]

    def gene_series(self, gene: str, field_name: str = "expressed_mean") -> list[float]:
        """One gene's `expressed_mean` (or `expressed_spread`) across the history.

        The series a plot of selection is drawn from: mean expressed `speed` rising over generations
        is the population adapting, and it is invisible in any population count.
        """
        if gene not in self.vocabulary.names:
            raise KeyError(f"'{gene}' is not in gene vocabulary v{self.vocabulary.version}")
        column = self.vocabulary.index_of(gene)
        return [getattr(sample, field_name)[column] for sample in self.samples]
