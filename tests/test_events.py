from unittest import TestCase

from entities import Bunny
from events.events import ChangeEntityStateEvent, CreateEntityEvent
from location import Location
from states import WalkingState
from world import World


class TestEvents(TestCase):
    def test_state_change(self):
        w = World()
        w.add_event(CreateEntityEvent(Bunny(location=Location(250, 250)) for _ in range(50)))
        for e in w.entities:
            w.add_event(ChangeEntityStateEvent(e.entity_id, WalkingState(e.entity_id, Location(1, 2), 1.0)))
            w.process_events()
            self.assertEqual(WalkingState, e.state.__class__)

