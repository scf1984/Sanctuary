from helper import get_all_subclasses
from traits import Metabolism, AgingRate


class Stat(object):
    is_state_setter = False
    max_value = 100.0
    updater = None
    name = None

    def __init__(self, stat_value):
        self.value = stat_value

    def __gt__(self, other):
        return self.value > other.value

    def update(self, animal, dt):
        if self.updater is not None:
            self.value += dt*animal.traits[self.updater].value
            print('Updated {0} by {1}, it\'s now {2}'.format(self.name, dt*animal.traits[self.updater].value, self.value ))


class Age(Stat):
    name = 'age'
    updater = AgingRate.name


class Hunger(Stat):
    is_state_setter = True
    name = 'hunger'
    updater = Metabolism.name


# class Health(Stat):
#     name = 'health'
#     pass


all_stats = {c.name: c for c in get_all_subclasses(Stat)}