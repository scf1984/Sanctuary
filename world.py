from enum import Enum
from heapq import heappop, heappush
from itertools import product
from math import ceil
from time import time
from warnings import warn

from traits import SightRange
from utils import Singleton, ilen


class InteractionGrid:
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

    def get_interactions(self):
        entities = tuple(Blackboard().iterate(Space.ENTITIES))
        for e in entities:
            self.add_entity(e)

        for (i, j) in self.cells.keys():
            neighbors = set.union(
                *[self.cells[c] for c in product([i + 1, i - 1, i], [j + 1, j - 1, j]) if c in self.cells])
            for e1 in self.cells[(i, j)]:
                for e2 in neighbors:
                    if e1 is not e2:
                        yield e1, e2


class EventHeap(metaclass=Singleton):
    def __init__(self):
        self.heap = []

    def peek(self):
        return self.heap[0] if len(self.heap) > 0 else None

    def pop(self):
        return heappop(self.heap)

    def put(self, event):
        heappush(self.heap, event)


class Space(Enum):
    ENTITIES = 'ENTITIES'


class Blackboard(metaclass=Singleton):
    def __init__(self):
        self.data = {k: {} for k in Space}

    def read(self, space: Space, key: str):
        return self.data[space][key]

    def write(self, space: Space, key: str, value):
        self.data[space][key] = value

    def iterate(self, space):
        return iter(self.data[space].values())


class World(metaclass=Singleton):

    def __init__(self, world_size=None):

        if world_size is None:
            warn('Using default world size.')
            world_size = (500, 500)

        self.world_map = world_size
        self.grid = [[None for _ in range(world_size[0])] for _ in range(world_size[1])]
        self.event_heap = EventHeap()
        Blackboard()

    @property
    def entities(self):
        return Blackboard().iterate(Space.ENTITIES)

    def update(self, dt):
        for entity in self.entities:
            entity.update(dt)

    def entity_interactions(self):
        cell_size = max(a.traits[SightRange].value for a in self.entities) * 2 if ilen(self.entities) > 0 else 1000
        interactions = InteractionGrid(cell_size).get_interactions()
        for e1, e2 in interactions:
            if e1.can_see(e2):
                e1.interact(e2)
            # TODO: Decide if interactions take place! (heading)

    def process_events(self):
        while self.event_heap.peek() and self.event_heap.peek().ts <= time():
            e = self.event_heap.pop()
            e.apply_event()

    def render(self, canvas):
        canvas.delete('all')
        canvas.create_line(0, 0, 0, 500, fill="green", width=10)
        canvas.create_line(0, 0, 500, 0, fill="green", width=10)
        canvas.create_line(0, 500, 500, 500, fill="green", width=10)
        canvas.create_line(500, 0, 500, 500, fill="green", width=10)
        for entity in Blackboard().iterate(Space.ENTITIES):
            left, up, right, down = entity.location.bbox(0.00)
            canvas.create_oval(left, up, right, down, outline="red", fill="red", width=2)
        canvas.update()


class EntityFetcher:
    entity_id: str

    @property
    def entity(self):
        return Blackboard().read(Space.ENTITIES, self.entity_id)
