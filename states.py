from abc import ABCMeta, abstractmethod
from random import random, uniform

from helper import Network
from location import Velocity, Location
from traits import MaxSpeed, Fidgetiness


class ABCState(metaclass=ABCMeta):
    def __init__(self, entity, destination=None):
        self.entity = entity
        self.destination = destination

    @property
    def velocity(self):
        if self.destination is None:
            return Velocity.random()
        else:
            return Velocity(((self.destination - self.entity.location).norm() * self.speed).coords)

    @abstractmethod
    def update(self, dt): pass



class Idle(ABCState):

    def __init__(self, entity):
        super().__init__(entity)
        self.next_walk_time = uniform(10, 500) / entity.traits[Fidgetiness].value

    def update(self, dt):
        self.next_walk_time -= dt
        if self.next_walk_time <= 0:
            self.entity.change_state(Walking(self.entity))


class Walking(ABCState):

    def __init__(self, entity, destination=None, speed=None):
        super().__init__(entity)
        self.destination = destination or entity.location + Location.random() * 25
        self.speed = speed or entity.traits[MaxSpeed].value * random() / 3.0

    def update(self, dt):
        self.entity.location = self.entity.location.go_to(self.destination, dt * self.speed)
        if self.entity.location is self.destination:
            self.entity.change_state(Idle(self.entity))


class Prowling(ABCState):
    pass


class Sexing(ABCState):
    pass


class Chasing(ABCState):
    pass


class Eating(ABCState):
    pass


class Drinking(ABCState):
    pass


class Panting(ABCState):
    pass


class Escaping(ABCState):
    pass


class Dying(ABCState):
    pass


class Dead(ABCState):
    pass


class StateTransitions(Network):
    default = Idle
    edges_dict = {
        Chasing: {Panting, Dying},
        Escaping: {Panting, Dying},
        Panting: {Idle, Dying},
        Idle: {Walking, Escaping, Prowling, Dying},
        Walking: {Idle, Escaping, Prowling, Dying},
        Prowling: {Chasing, Idle, Dying},
        Dying: {Dead}
    }
