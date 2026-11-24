"""Scenario goals: checks against what a world recorded, never predictions (#292 decision 3, #317).

A scenario is a world plus a goal, and the free world keeps running underneath it (§2.7). A goal is
read, not run: it is a pure function of the lineage, the metric window and the extinctions the
recorder keeps, so it holds what it held whether or not anyone was watching (§2.4), survives a
save and reload with nothing of its own to save, and cannot disagree with the history it reads
(§3.2). That is why nothing here writes anything, and why a goal over something nobody records
needs that thing recorded first.

**It never forecasts** (§2.3, #216). "Open" means only that the outcome has not happened. Progress
is a measurement of now, which drift can move either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from core.genetics.speciation import Lineage
from metrics.history import MetricHistory

Outcome = Literal["open", "met", "lost"]


@dataclass(frozen=True)
class GoalReading:
    """Where a goal stands, in plain values (§4).

    outcome: "open", "met" or "lost". Met and lost are final: a goal met stays met whatever the
        world does next, and one lost stays lost, since extinction is permanent (§2.7).
    progress: dimensionless. Before the split, how far the founding species' two most separated
        groups have come toward one, at the latest sample in which the speciation check measured
        it (`Sample.species_divergence`): 1 is where the check splits them. None while no sample
        in the window holds a measurement. 1 from the split on.

        The latest *measured* sample rather than the latest sample, because the check skips a
        species whose two proposed groups are not both `min_founders` strong, and on the demo
        world it skipped up to a quarter of checks (`docs/spikes/scenario-progress.md`). Read
        raw, a progress bar would drop to nothing for a generation at a time for a reason no
        player could see.
    parent, daughter: the species the goal names, once the split it is waiting on has happened.
    split_tick: when that split happened.
    due_tick: the sample at which both must still be alive, a generation after the split.
    resolved_tick: when the goal was met or lost; None while it is open.
    reason: why it was lost, for a person; None otherwise.
    """

    outcome: Outcome
    progress: Optional[float]
    parent: Optional[int]
    daughter: Optional[int]
    split_tick: Optional[int]
    due_tick: Optional[int]
    resolved_tick: Optional[int]
    reason: Optional[str]


@dataclass(frozen=True)
class FirstSplit:
    """The first scenario's goal (#317, decisions 1 and 2): the world's first split is recorded,
    and the parent and daughter are both alive a generation later.

    generation_ticks: ticks the two must both survive the split for. A generation, so that the
        daughter has bred rather than merely been counted: a split founds a species from at least
        `min_founders` animals, which is a placeholder (§2.5), and a species that dies out before
        its first young are born was a stray group with an id.

    Lost when the founding species dies out before any split, which in a world of one species is
    its collapse, or when either species of the first split dies out before the due sample. The
    first split is the one named: a daughter that dies is a loss the player made, and a later split
    elsewhere does not undo it.

    The due sample is the first one at or after `split tick + generation_ticks`, because that is
    the first moment the recorder looks. Whether a species was alive between two samples is not
    recorded, so the goal asks what the recorder saw rather than what it can only guess.
    """

    generation_ticks: int
    founding_species: int = 0

    def __post_init__(self) -> None:
        if self.generation_ticks < 1:
            raise ValueError(
                f"generation_ticks must be at least 1, got {self.generation_ticks}; a daughter "
                "counted on the tick it split off has shown nothing about whether it can breed"
            )

    def assess(self, history: MetricHistory, lineage: Lineage) -> GoalReading:
        """Where the goal stands now, read off `history` and `lineage`, which it never writes."""
        latest = history.samples[-1] if history.samples else None
        splits = lineage.splits()
        if not splits:
            extinct = history.extinct_at(self.founding_species)
            if extinct is not None:
                return _resolved(
                    "lost",
                    None,
                    resolved_tick=extinct,
                    reason="the founding species died out before any group split off",
                )
            return _resolved("open", self._measured(history))

        first = splits[0]
        every = history.config.every_n_ticks
        # The first sample at or after a generation has passed: samples fall on the cadence.
        due = -(-(first.tick + self.generation_ticks) // every) * every
        named = {"parent": first.parent_species_id, "daughter": first.species_id}
        deaths = [
            (tick, role)
            for role, species in named.items()
            if (tick := history.extinct_at(species)) is not None and tick <= due
        ]
        if deaths:
            tick, role = min(deaths)
            outcome: Outcome = "lost"
            resolved: Optional[int] = tick
            reason: Optional[str] = (
                f"the {role} species died out within a generation of the split"
            )
        elif latest is not None and latest.tick >= due:
            outcome, resolved, reason = "met", due, None
        else:
            outcome, resolved, reason = "open", None, None
        return GoalReading(
            outcome=outcome,
            progress=1.0,
            parent=first.parent_species_id,
            daughter=first.species_id,
            split_tick=first.tick,
            due_tick=due,
            resolved_tick=resolved,
            reason=reason,
        )


    def _measured(self, history: MetricHistory) -> Optional[float]:
        """The founding species' divergence at the latest sample that measured it, or None.

        A sample holds 0 for a species the check did not measure (`Sample.species_divergence`).
        A measured reading is a distance between two centroids of real animals over the distance
        a split needs, and is 0 only if the two groups' signature means coincide exactly, which
        two groups drawn apart by a clustering never do.
        """
        for sample in reversed(history.samples):
            readings = sample.species_divergence
            if self.founding_species < len(readings) and readings[self.founding_species] > 0.0:
                return readings[self.founding_species]
        return None


def _resolved(
    outcome: Outcome,
    progress: Optional[float],
    resolved_tick: Optional[int] = None,
    reason: Optional[str] = None,
) -> GoalReading:
    """A reading from before any split, which names no species."""
    return GoalReading(
        outcome=outcome,
        progress=progress,
        parent=None,
        daughter=None,
        split_tick=None,
        due_tick=None,
        resolved_tick=resolved_tick,
        reason=reason,
    )
