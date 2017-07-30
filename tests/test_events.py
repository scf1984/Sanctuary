from unittest import TestCase

from entities import Bunny
from events import ChangeAnimalStateEvent
from location import Location
from main import WorldRunner
from states import Walking
from world import World


class TestEvents(TestCase):
    def test_state_change(self):
        w = World(world_map=[], animals=[Bunny(location=Location(250, 250)) for _ in range(50)])
        wr = WorldRunner(world=w)
        wr.world.add_event(ChangeAnimalStateEvent(wr.world.animals[0], Walking))
        wr.world.process_events()
        self.assertEqual(Walking, wr.world.animals[0].state)

