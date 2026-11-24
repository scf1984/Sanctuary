"""Named scenarios: a world plus a goal, opened by name as a world is (#292 decision 3, #317).

A scenario is layered on the sandbox, never instead of it (§2.7). The world it names is built and
run exactly as it is without one, so choosing a scenario changes what the player is told and asked,
and nothing the world does. That is why a scenario is not part of a world's config and not in its
fingerprint: two worlds that differ only in their scenario run under one rule set.

Its words live here, beside the goal they describe, so that the briefing and the check cannot be
edited apart. They are content, like a world definition, and a client is handed them rather than
writing its own (§4).
"""

from __future__ import annotations

from dataclasses import dataclass

from metrics.goals import FirstSplit
from worlds.demo import GENERATION_TICKS, SPLIT_FOUNDERS


@dataclass(frozen=True)
class Scenario:
    """One scenario.

    world: the name of the world it runs on, one of `worlds.WORLDS`.
    title: what it is called, for a person.
    goal_text: the goal in one sentence, for a person.
    briefing: what a first-time player is told before they start, one paragraph per entry.
    goal: the check, against what the world recorded (`metrics.goals`).
    """

    world: str
    title: str
    goal_text: str
    briefing: tuple[str, ...]
    goal: FirstSplit


FIRST_SPLIT = Scenario(
    world="demo",
    title="A species of your own",
    goal_text=(
        "Fence a group off until it becomes a species of its own, and keep both it and the "
        "species it came from alive for a generation after the split."
    ),
    briefing=(
        "Every animal here is one species. Animals pass on their traits with small random "
        "changes, so a group that breeds only among itself slowly drifts away from everyone else. "
        "Drift far enough apart and the two groups can no longer interbreed: they have become "
        "two species.",
        "Fence some animals in to start that drift. Whoever is inside when the fence goes up "
        f"stays inside. A new species needs at least {SPLIT_FOUNDERS} animals, so fence in more "
        "than that, with grass and water. A small pen drifts apart sooner but can die out; a "
        "large one is safer and slower.",
        "It takes time: hours at the normal pace. Expect it to happen while you are away, since "
        "the world keeps running when you close the page. Fast-forward helps while you watch.",
        "The goal panel shows how far apart the two most different groups of animals are, "
        "checked about once a generation. Even a herd nobody fenced reads between a quarter and "
        "a half, and a fenced group stays close to that for much of the wait. It climbs once "
        "your group has drifted far enough to stand out from the rest, and the split follows.",
    ),
    goal=FirstSplit(generation_ticks=GENERATION_TICKS),
)

# The scenarios a world can be founded with, by the name a person types. One entry, so a mapping
# and not a framework (§8.2).
SCENARIOS = {"first-split": FIRST_SPLIT}


def scenario_named(name: str) -> Scenario:
    """The scenario called `name`, refusing loudly rather than founding a plain world (§8.7): a
    typo that silently dropped the goal would leave a player with nothing to aim at and no
    message saying why."""
    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f"unknown scenario {name!r}; choose one of: {', '.join(SCENARIOS)}"
        ) from None
