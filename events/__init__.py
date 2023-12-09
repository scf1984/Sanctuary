from dataclasses import dataclass, field
from time import time
from typing import Callable

from world import EntityFetcher, EventHeap


@dataclass
class Event(EntityFetcher):
    action_function: Callable
    ts: float = field(default_factory=time)
    dt: float = 0.

    def __post_init__(self):
        EventHeap().put(self)

    def is_invalidated(self):
        return False

    def __lt__(self, other):
        return self.ts < other.ts

    def apply_event(self):
        if not self.is_invalidated():
            self.action_function()
