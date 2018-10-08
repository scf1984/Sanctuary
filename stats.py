from events import CreateEntityEvent, ChangeEntityEvent
from helper import get_all_subclasses
from inclinations import Hungry, InHeat
from traits import Metabolism, AgingRate, PregnancyRate


class Stat(object):
    is_state_setter = False
    max_value = 100.0
    updater = None
    threshold = 80.0
    inclination = None
    name = None

    def __init__(self):
        self.value = 0

    def __gt__(self, other):
        return self.value > other.value

    def update(self, entity, world, dt=None, ds=None):
        if ds is None:
            ds = dt * (entity.traits[self.updater].value or 1.0)
        if (
                (self.value - self.threshold) * (self.value + ds - self.threshold) < 0
                or (self.value - self.threshold) == 0
        ):
            self.threshold_function(entity)
            if self.inclination not in entity.inclinations:
                entity.inclinations.add(self.inclination)
            else:
                entity.inclinations.remove(self.inclination)
        self.value += ds

    def threshold_function(self, entity):
        pass


class Age(Stat):
    is_state_setter = True
    threshold = 100
    name = 'age'
    updater = AgingRate

    def threshold_function(self, entity):
        entity.world.add_event(ChangeEntityEvent(entity, None))


class Hunger(Stat):
    is_state_setter = True
    name = 'hunger'
    updater = Metabolism
    inclination = Hungry


class Pregnancy(Stat):
    threshold = 100
    name = 'pregnancy'
    updater = PregnancyRate
    inclination = InHeat

    def threshold_function(self, entity):
        if self.value > self.threshold:
            return
        entity.world.add_event(CreateEntityEvent(entity.baby))
        entity.baby = None

    def update(self, entity, world, dt=None, ds=None):
        if entity.baby is not None:
            super().update(entity, world, dt, ds)


all_stats = {c.name: c for c in get_all_subclasses(Stat)}
