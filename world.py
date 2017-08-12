from time import time
from heapq import heappop, heappush
from itertools import product
from math import floor, ceil

from entities import Entity
from events import Event
from location import Location
from traits import SightRange, SightAngle


class InteractionGrid(object):
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.cells = dict()

    def add_entity(self, entity):
        x, y = (entity.location * (1 / self.cell_size)).coords
        i, j = ceil(x), ceil(y)
        cell = (i, j)
        if cell not in self.cells:
            self.cells[cell] = set()
        self.cells[cell].add(entity)

    def get_interactions(self, entities):
        for e in entities:
            e.entities_in_range.clear()
            self.add_entity(e)

        for (i, j) in self.cells.keys():
            neighbors = set.union(
                *[self.cells[c] for c in product([i + 1, i - 1, i], [j + 1, j - 1, j]) if c in self.cells])
            for e1 in self.cells[(i, j)]:
                for e2 in neighbors:
                    if e1 is not e2:
                        yield (e1, e2)


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
            raise TypeError('Got a non-Event pushed into the event heap: {0}'.format(event))


class World(object):
    def __init__(self, world_map, **kwargs):
        self.world_map = world_map
        self.grid = [[None for _ in range(100)] for _ in range(100)]
        self.entities = set()
        self.event_heap = EventHeap()

    def update(self, dt):
        for entity in self.entities:
            entity.update(dt)

    def add_event(self, event):
        if not issubclass(event.__class__, Event):
            raise TypeError('Tried to create an event from ' + event.__class__.__name__)
        self.event_heap.put(event)

    def entity_interactions(self):
        cell_size = max(a.traits[SightRange].value for a in self.entities) * 2 if len(self.entities) > 0 else 1000
        interactions = InteractionGrid(cell_size).get_interactions(self.entities)
        for e1, e2 in interactions:
            if Location.square_distance(e1.location, e2.location) < e1.traits[SightRange].value ** 2 and abs(
                    e1.get_velocity().angle(e1.location - e2.location)) < e1.traits[SightAngle].value:
                e1.entities_in_range.add(e2)
                e1.interact(e2)
            # TODO: Decide if interactions take place! (heading)

    def process_events(self):
        while self.event_heap.peek() and self.event_heap.peek().ts <= time():
            e = self.event_heap.pop()
            e.apply_event(self)

    def render(self, canvas):
        canvas.delete('all')
        canvas.create_line(0, 0, 0, 500, fill="green", width=10)
        canvas.create_line(0, 0, 500, 0, fill="green", width=10)
        canvas.create_line(0, 500, 500, 500, fill="green", width=10)
        canvas.create_line(500, 0, 500, 500, fill="green", width=10)
        for entity in self.entities:
            left, up, right, down = entity.location.bbox(0.00)
            canvas.create_oval(left, up, right, down, outline="red", fill="red", width=2)
        canvas.update()
