from abc import abstractmethod
from copy import copy
from uuid import uuid4

from reproduction_systems import ReproductionSystem, Spawn, TwoSexual
from location import Location
from utils import Network
from states import StateTransitions
from stats import Stat
from traits import all_traits, Metabolism, AgingRate, ThirstRate, SightRange, Fidgetiness, MaxSpeed, SightAngle, \
    PregnancyRate
from world import World, Blackboard, Space


class Entity:
    default_traits = {}
    reproduction_system: type(ReproductionSystem) | ReproductionSystem = None
    eats: set['Entity'] = None

    def __mul__(self, other):
        if not self.can_reproduce_with(other):
            raise ValueError('Two incompatible species tried to reproduce?!')

        entity_class = self.__class__
        return entity_class(location=(self.location + other.location) * 0.5,
                            traits={k: self.traits[v] * other.traits[v] for k, v in all_traits.items()})

    @abstractmethod
    def is_dead(self):
        pass

    def change_state(self, new_state):
        self.state = new_state

    def __init__(self, location, traits=None):
        self.entity_id = str(uuid4())
        Blackboard().write(Space.ENTITIES, self.entity_id, self)

        traits = traits or {}
        self.stats: dict[type(Stat): Stat] = {s: s() for s in Stat.all_stats.values()}
        self.traits = {k: traits.get(k, copy(self.default_traits.get(k))) for k in all_traits.values()}
        self.location = location

        self.baby = None
        self.state_network = StateTransitions(self)
        self.state = self.state_network.default(self.entity_id)
        self.inclinations = set()

        self.reproduction_system = self.reproduction_system(self.entity_id)

        self.world = World()

        super().__init__()

    def update(self, dt):
        for k, v in self.stats.items():
            v.update(None, self)
        self.state.update(dt)

    @property
    def velocity(self):
        return self.state.velocity

    @property
    def heading(self):
        return self.velocity.angle

    def can_eat(self, other):
        return other.__class__ in self.eats

    def is_eaten_by(self, other):
        return self.__class__ in other.eats

    def can_see(self, other):
        return (
            Location.square_distance(self.location, other.location) < self.traits[SightRange].value ** 2
            and abs(self.velocity.angle - (self.location - other.location).angle) < self.traits[SightAngle].value
        )

    def interact(self, other):
        if self.mates_with(other):
            self.reproduction_system.procreate(other)
        if self.can_eat(other):
            pass  # TODO: self will prowl/chase other.
        if self.is_eaten_by(other):
            pass  # TODO: self will avoid/escape other.

        pass  # TODO: Any other interactions?

    def can_reproduce_with(self, other):
        return self.__class__ == other.__class__

    def mates_with(self, other):
        return self.reproduction_system.mates_with(other)


class Water(Entity):
    def is_dead(self):
        pass


class Grass(Entity):
    reproduction_system = Spawn

    def is_dead(self):
        pass


class Bunny(Entity):
    reproduction_system = TwoSexual
    eats = {Grass}
    default_traits = {
        Metabolism: Metabolism(10),
        AgingRate: AgingRate(25),
        ThirstRate: ThirstRate(3),
        SightRange: SightRange(5),
        Fidgetiness: Fidgetiness(25),
        MaxSpeed: MaxSpeed(50),
        SightAngle: SightAngle(35),
        PregnancyRate: PregnancyRate(50)
    }

    def is_dead(self):
        pass


class Wolf(Entity):
    reproduction_system = TwoSexual
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

    def is_dead(self):
        pass


class FoodChain(Network):
    edges_dict = {
        Bunny: {Grass},
        Wolf: {Bunny}
    }
