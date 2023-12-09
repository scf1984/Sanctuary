from time import time

from world import EntityFetcher


class Event(EntityFetcher):
    def __init__(self, action_function=None, ts=time(), dt=0):
        self.action_function = action_function
        self.ts = ts + dt

    def is_invalidated(self):
        return False

    def __lt__(self, other):
        return self.ts < other.ts

    def apply_event(self):
        if not self.is_invalidated():
            self.action_function()
