from unittest import TestCase

from events import ChangeStateEvent
from main import WorldRunner
from states import Walking


class TestEvents(TestCase):
    def test_state_change(self):
        wr = WorldRunner(world_map=[])
        wr.world.add_event(ChangeStateEvent(wr.world.animals[0], Walking))
        wr.world.process_events()
        self.assertEqual(Walking, wr.world.animals[0].state)

