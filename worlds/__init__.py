"""Named worlds: the content a client asks for by name, never by building config itself (#35).

Scenarios are content too: a world plus a goal, opened by name (#317)."""

from worlds.demo import DEFAULT_WORLD, WORLDS, build_named_world, pacing_for
from worlds.scenarios import SCENARIOS, Scenario, scenario_named

__all__ = [
    "DEFAULT_WORLD",
    "SCENARIOS",
    "WORLDS",
    "Scenario",
    "build_named_world",
    "pacing_for",
    "scenario_named",
]
