from events import CreateEntityEvent, ChangeEntityEvent
from utils import get_all_subclasses
from inclinations import Hungry, InHeat
from stats import Stat
from traits import Metabolism, AgingRate, PregnancyRate


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



