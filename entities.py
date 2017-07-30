from copy import copy

from helper import Network
from location import Location
from states import Idle
from stats import all_stats, Hunger, Age
from traits import all_traits, Metabolism, AgingRate, ThirstRate, SightRange


class Entity(object):
    default_stats = {Age.name: Age(0), Hunger.name: Hunger(0)}
    default_traits = {}

    # Attempt at event-driven many body simulation:
    # def in_range_time(self, other):
    #     dr = self.location - other.location
    #     range_ = self.traits[SightRange].value
    #     if dr.square_magnitude() <= range_ ** 2:
    #         return 0
    #
    #     dv = self.get_velocity() - other.get_velocity()
    #     dvdr = dv * dr
    #     if dvdr >= 0:
    #         return None
    #
    #     d = dvdr ** 2 - (dv * dv) * (dr * dr - range_)
    #     if d < 0:
    #         return None
    #
    #     return -(dvdr + sqrt(d)) / (dv * dv)

    def __mul__(self, other):
        if self.__class__ != other.__class__:
            raise ValueError('Two different species tried to reproduce?!')
        else:
            animal_class = self.__class__
        return animal_class(location=self.location,
                            **{k: self.traits[v] * other.traits[v] for k, v in all_traits.items()})

    def is_dead(self):
        return False

    def change_state(self, new_state):
        self.state = new_state

    def __init__(self, **kwargs):
        self.eats = FoodChain.dict[self.__class__]
        self.stats = [kwargs.get(k, copy(self.__class__.default_stats.get(k))) for k in all_stats.keys()]
        self.traits = {k: kwargs.get(k, copy(self.__class__.default_traits.get(k))) for k in all_traits.values()}
        self.location = kwargs.get('location', Location(coords=(0, 0)))

        self.state = Idle(self)

    def update(self, dt):
        for i in range(len(self.stats)):
            self.stats[i].update(self, dt)
            if self.stats[i].is_state_setter and self.stats[i] > self.stats[0]:
                self.stats[i], self.stats[0] = self.stats[0], self.stats[i]
                print('Now the state should be ' + self.stats[0].name)
        self.state.update(dt)

    def get_velocity(self):
        return self.state.get_velocity()

    def interact(self, other):
        pass  # TODO: Interact with another animal!


class Grass(Entity):
    pass


class Bunny(Entity):
    default_traits = dict(
        {
            Metabolism: Metabolism(10),
            AgingRate: AgingRate(10),
            ThirstRate: ThirstRate(3),
            SightRange: SightRange(5)
        }
    )
    pass


class Wolf(Entity):
    pass


class FoodChain(Network):
    dict = {
        Entity: set(),
        Bunny: {Grass},
        Wolf: {Bunny}
    }


class GENDER(object):
    MALE = 1
    FEMALE = 2
    UNISEX = 3
    ASEXUAL = 4
    SPAWN = 5
    DOESNT_REPRODUCE = 6
