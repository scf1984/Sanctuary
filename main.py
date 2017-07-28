import time
from heapq import heappush, heappop

from entities import Bunny
from events import Event


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

    def add_event(self, event):
        if not issubclass(event.__class__, Event):
            raise TypeError('Tried to create an event from ' + event.__class__.__name__)
        self.event_heap.put(event)

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
    def __init__(self, **kwargs):
        self.world = World(kwargs)

    def run(self):
        import time
        loop_tick = 0.25
        prev_time = time.time()
        while True:
            now = time.time()
            dt = now - prev_time
            prev_time = now

            self.world.process_events()
            self.world.update(dt)
            self.world.render()

            sleep_time = max(loop_tick - (time.time() - prev_time), 0)
            print(str(sleep_time))
            time.sleep(sleep_time)


if __name__ == '__main__':
    WorldRunner(world_map=[]).run()
