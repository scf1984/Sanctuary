from copy import copy

from entity_mixins import Sexual, ReproductionSystem, Asexual
from helper import Network
from states import Idle, StateTransitions
from stats import all_stats
from traits import all_traits, Metabolism, AgingRate, ThirstRate, SightRange, Fidgetiness, MaxSpeed, SightAngle, \
    PregnancyRate


class FoodChainLevel(object):
    def eats(self): return {}


class Aquavore(FoodChainLevel):
    def eats(self): return {}


class Herbivore(FoodChainLevel):
    def eats(self): return {Grass}


class Omnivore(FoodChainLevel):
    def eats(self): return {Grass, Bunny}


class Carnivore(FoodChainLevel):
    def eats(self): return {Bunny}


class SuperCarnivore(FoodChainLevel):
    def eats(self): return {}


class Entity(ReproductionSystem, FoodChainLevel, object):
    default_traits = {}

    def __mul__(self, other):
        if self.__class__ != other.__class__:
            raise ValueError('Two different species tried to reproduce?!')
        else:
            entity_class = self.__class__
        return entity_class(location=self.location, world=self.world,
                            **{k: self.traits[v] * other.traits[v] for k, v in all_traits.items()})

    def is_dead(self):
        return False

    def change_state(self, new_state):
        if StateTransitions.edge_exists(self.state.__class__, new_state.__class__):
            self.state = new_state
        else:
            raise ValueError('Illegal transition, from {0} to {1}'.format(self.state.__class__, new_state))

    def __init__(self, **kwargs):
        self.stats = {s: s() for s in all_stats.values()}
        self.traits = {k: kwargs.get(k, copy(self.__class__.default_traits.get(k))) for k in all_traits.values()}
        self.location = kwargs.get('location')

        self.baby = None
        self.state = Idle(self)
        self.inclinations = set()
        self.entities_in_range = set()

        self.world = kwargs.get('world')

        super().__init__()

    def update(self, dt):
        for k, v in self.stats.items():
            v.update(self, dt=dt, world=self.world)
        self.state.update(dt)

    def get_velocity(self):
        return self.state.get_velocity()

    def interact(self, other):
        self.interact_sex(other)

        pass  # TODO: Interact with another entity!




class Water(Entity):
    pass


class Grass(Entity, Asexual, Aquavore):
    pass


class Bunny(Entity, Sexual, Herbivore):
    default_traits = dict(
        {
            Metabolism: Metabolism(10),
            AgingRate: AgingRate(25),
            ThirstRate: ThirstRate(3),
            SightRange: SightRange(5),
            Fidgetiness: Fidgetiness(10),
            MaxSpeed: MaxSpeed(50),
            SightAngle: SightAngle(35),
            PregnancyRate: PregnancyRate(50)
        }
    )
    pass


class Wolf(Entity, Sexual, Carnivore):
    default_traits = dict(
        {
            Metabolism: Metabolism(10),
            AgingRate: AgingRate(10),
            ThirstRate: ThirstRate(3),
            SightRange: SightRange(5),
            Fidgetiness: Fidgetiness(10),
            MaxSpeed: MaxSpeed(50),
            SightAngle: SightAngle(35),
            PregnancyRate: PregnancyRate(5)
        }
    )
    pass


class FoodChain(Network):
    edges_dict = {
        Entity: set(),
        Bunny: {Grass},
        Wolf: {Bunny}
    }
