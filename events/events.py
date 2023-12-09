from dataclasses import dataclass, field
from time import time

from events import Event
from states import ABCState
from world import Blackboard, Space


@dataclass
class ChangeEntityStateEvent(Event):
    entity_id: str
    future_state: ABCState
    ts: float = field(default_factory=time)
    dt: float = field(default=0.0)

    def __post_init__(self):
        def action_function():
            self.entity.change_state(self.future_state)

        super().__init__(action_function, ts=ts, dt=dt)

    def is_invalidated(self):
        return self.entity.is_dead()


class CreateEntityEvent(Event):
    def __init__(self, entity, ts=time(), dt=0):
        def action_function():
            Blackboard().write(Space.ENTITIES, entity.entity_id, entity)

        super().__init__(action_function=action_function, ts=ts, dt=dt)

    def is_invalidated(self):
        return False


class ChangeEntityEvent(Event):
    def __init__(self, old_entity, new_entity, ts=time(), dt=0):
        def action_function(world):
            world.entities.remove(old_entity)
            if new_entity is not None:
                world.entities.add(new_entity)

        super().__init__(action_function=action_function, ts=ts, dt=dt)

    def is_invalidated(self):
        return False
