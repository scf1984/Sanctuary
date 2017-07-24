from heapq import heappush, heappop
from queue import Queue

import time

from entities import Bunny


class GENDER(object):
    MALE = 1
    FEMALE = 2
    UNISEX = 3
    ASEXUAL = 4
    SPAWN = 5


class Tile(object):
    ground_type = None
    permanent_object = None


class GroundType(object):
    pass


class Sand(GroundType):
    pass


class PermObject(object):
    pass


class Tree(PermObject):
    pass


class Event(object):
    action_function = None
    params = None
    ts = None

    def __lt__(self, other):
        return self.ts < other.ts

    def apply_event(self):
        self.action_function(self.params)


class EventHeap(object):
    def __init__(self):
        self.heap = []

    def peek(self):
        return self.heap[0] if len(self.heap) > 0 else None

    def pop(self):
        return heappop(self.heap)

    def put(self, event):
        if issubclass(event.__class__, Event):
            heappush(self.heap, event)
        else:
            raise ValueError('Got a non-Event pushed into the event heap: {0}'.format(event))


class World(object):
    def __init__(self, world_map):
        self.world_map = world_map
        self.grid = [[None for _ in range(100)] for _ in range(100)]
        self.animals = [Bunny() for _ in range(5)]
        self.event_heap = EventHeap()

    def update(self, dt):
        for animal in self.animals:
            animal.update(dt)

    def process_events(self):
        while self.event_heap.peek() and self.event_heap.peek().ts <= time.time():
            e = self.event_heap.pop()
            e.apply_event()

    def render(self):
        pass


class Universe(object):
    worlds = []
    pass


class WorldRunner(object):
    @staticmethod
    def run():
        import time
        loop_tick = 0.25
        world = World(world_map=[])
        prev_time = time.time()
        while True:
            now = time.time()
            dt = now - prev_time
            prev_time = now

            world.process_events()
            world.update(dt)
            world.render()

            sleep_time = max(loop_tick - (time.time() - prev_time), 0)
            print(str(sleep_time))
            time.sleep(sleep_time)


if __name__ == '__main__':
    WorldRunner.run()
