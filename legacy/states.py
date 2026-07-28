from abc import ABCMeta, abstractmethod, ABC
from dataclasses import dataclass
from random import uniform

from location import Velocity, Location
from traits import Fidgetiness
from utils import Network, Vector
from world import Blackboard, Space


@dataclass
class ABCState(metaclass=ABCMeta):
    entity_id: str
    velocity = Vector((0.0, 0.0))

    @property
    def entity(self):
        return Blackboard().read(Space.ENTITIES, self.entity_id)

    @abstractmethod
    def update(self, dt): pass


@dataclass
class ABCMovementState(ABCState, ABC):
    destination: Location
    speed: float

    @property
    def velocity(self):
        if self.destination is None:
            return Velocity((0.0, 0.0))
        else:
            return Velocity(((self.destination - self.entity.location).norm() * self.speed).coords)

    def update(self, dt):
        self.entity.location = self.entity.location.go_to(self.destination, dt * self.speed)
        if self.entity.location is self.destination:
            self.entity.change_state(IdleState(self.entity_id))


@dataclass
class IdleState(ABCState):
    def __post_init__(self):
        self.next_walk_time = uniform(10, 500) / self.entity.traits[Fidgetiness].value

    def update(self, dt):
        self.next_walk_time -= dt
        if self.next_walk_time <= 0:
            self.entity.change_state(WalkingState(self.entity_id, self.entity.location + Location(1, -1), 10))


@dataclass
class WalkingState(ABCMovementState):
    pass


@dataclass
class ProwlingState(ABCState):
    pass


@dataclass
class SexingState(ABCState):
    pass


@dataclass
class ChasingState(ABCMovementState):
    pass


class EatingState(ABCState):
    pass


class DrinkingState(ABCState):
    pass


class PantingState(ABCState):
    pass


class EscapingState(ABCMovementState):
    pass


class DyingState(ABCState):
    pass


class DeadState(ABCState):
    pass


class StateTransitions(Network):
    default = IdleState
    edges_dict = {
        ChasingState: {PantingState, DyingState},
        EscapingState: {PantingState, DyingState},
        PantingState: {IdleState, DyingState},
        IdleState: {WalkingState, EscapingState, ProwlingState, DyingState},
        WalkingState: {IdleState, EscapingState, ProwlingState, DyingState},
        ProwlingState: {ChasingState, IdleState, DyingState},
        DyingState: {DeadState}
    }
