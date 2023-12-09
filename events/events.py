from dataclasses import dataclass, field
from time import time

from events import Event
from states import ABCState


@dataclass
class ChangeEntityStateEvent(Event):
    entity_id: str = None
    future_state: ABCState = None
    ts: float = field(default_factory=time)
    dt: float = field(default=0.0)

    def action_function(self):
        self.entity.change_state(self.future_state)

    def is_invalidated(self):
        return self.entity.is_dead()
