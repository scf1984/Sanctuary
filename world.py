import time
from heapq import heappop, heappush

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
    def __init__(self, world_map, **kwargs):
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

    def render(self, canvas):
        canvas.create_line(0, 0, 0, 500, fill="green", dash=(10, 10), width=3)
        canvas.create_line(0, 0, 500, 0, fill="green", dash=(10, 10), width=3)
        canvas.create_line(0, 500, 500, 500, fill="green", dash=(10, 10), width=3)
        canvas.create_line(500, 0, 500, 500, fill="green", dash=(10, 10), width=3)
        for animal in self.animals:
            left, up, right, down = animal.location.bbox(5.0)
            canvas.create_oval(left, up, right, down, outline="red", fill="red", width=2)
        canvas.update()
