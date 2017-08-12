from unittest import TestCase

from entities import Bunny
from events import ChangeEntityStateEvent
from location import Location
from main import WorldRunner
from states import Walking
from world import World


class TestEvents(TestCase):
    def test_state_change(self):
        w = World(world_map=[])
        w.entities |= {Bunny(location=Location(250, 250)) for _ in range(50)}
        for e in w.entities:
            break
        w.add_event(ChangeEntityStateEvent(e, Walking))
        w.process_events()
        self.assertEqual(Walking, e.state.__class__)

