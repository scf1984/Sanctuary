import time
from heapq import heappop, heappush
from itertools import product
from math import floor, ceil

from events import Event
from traits import SightRange


class InteractionGrid(object):
    def __init__(self, cell_size):
        self.interactions = set()
        self.cell_size = cell_size
        self.cells = dict()

    def add_entity(self, entity):
        x, y = (entity.location * (1 / self.cell_size)).coords
        i, j = ceil(x), ceil(y)
        cell = (i, j)
        if cell not in self.cells:
            self.cells[cell] = set()

        neighbors = set.union(*[self.cells[c] for c in product([i+1, i-1, i], [j+1, j-1, j]) if c in self.cells])

        for e in neighbors:
            self.interactions.add((entity, e))
            self.interactions.add((e, entity))
        self.cells[cell].add(entity)

    def get_interactions(self, entities):
        for e in entities:
            self.add_entity(e)
        return self.interactions


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
    def __init__(self, world_map, animals):
        self.world_map = world_map
        self.grid = [[None for _ in range(100)] for _ in range(100)]
        self.animals = animals
        self.event_heap = EventHeap()

    def update(self, dt):
        for animal in self.animals:
            animal.update(dt)

    def add_event(self, event):
        if not issubclass(event.__class__, Event):
            raise TypeError('Tried to create an event from ' + event.__class__.__name__)
        self.event_heap.put(event)

    def find_animal_interactions(self):
        cell_size = max(a.traits[SightRange].value for a in self.animals) * 2
        interactions = InteractionGrid(cell_size).get_interactions(self.animals)
        for i in interactions:
            pass  # TODO: Decide if interactions take place!
        del interactions

    def process_events(self):
        while self.event_heap.peek() and self.event_heap.peek().ts <= time.time():
            e = self.event_heap.pop()
            e.apply_event()

    def render(self, canvas):
        canvas.delete('all')
        canvas.create_line(0, 0, 0, 500, fill="green", dash=(10, 10), width=3)
        canvas.create_line(0, 0, 500, 0, fill="green", dash=(10, 10), width=3)
        canvas.create_line(0, 500, 500, 500, fill="green", dash=(10, 10), width=3)
        canvas.create_line(500, 0, 500, 500, fill="green", dash=(10, 10), width=3)
        for animal in self.animals:
            left, up, right, down = animal.location.bbox(5.0)
            canvas.create_oval(left, up, right, down, outline="red", fill="red", width=2)
        canvas.update()
