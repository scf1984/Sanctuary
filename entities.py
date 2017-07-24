from states import Idle
from stats import all_stats, Hunger, Age
from traits import all_traits, Metabolism, AgingRate
from copy import copy


class Entity(object):
    default_stats = {Age.name: Age(0)}
    default_traits = {AgingRate.name: AgingRate(1)}

    def __init__(self, **kwargs):
        self.eats = object.__getattribute__(FoodChain, self.__class__.__name__)
        self.stats = [kwargs.get(k, copy(self.__class__.default_stats.get(k))) for k in all_stats.keys()]
        self.traits = {k: kwargs.get(k, copy(self.__class__.default_traits.get(k))) for k in all_traits.keys()}

        self.state = Idle(self)
        self.target_animal = None

    def update(self, dt):
        for i in range(len(self.stats)):
            self.stats[i].update(self, dt)
            if self.stats[i].is_state_setter and self.stats[i] > self.stats[0]:
                self.stats[i], self.stats[0] = self.stats[0], self.stats[i]
                print('Now the state should be ' + self.stats[0].name)
        self.state.update(self, dt)


class Grass(Entity):
    pass


class Bunny(Entity):
    default_traits = dict(
        {
            Metabolism.name: Metabolism(2),
        },
        **Entity.default_traits
    )
    default_stats = dict(
        {
            Hunger.name: Hunger(0),
        },
        **Entity.default_stats
    )
    pass


class Wolf(Entity):
    pass


class FoodChain(object):
    Bunny = [Grass]
    Wolf = [Bunny]
