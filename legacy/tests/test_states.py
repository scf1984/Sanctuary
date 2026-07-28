from unittest import TestCase

from entities import Bunny
from location import Location, Velocity
from states import WalkingState


class TestStates(TestCase):
    def test_walking_velocity(self):
        b = Bunny(location=Location(0, 0))
        b.change_state(WalkingState(b.entity_id, Location(1, 2), 1))
        self.assertEqual(b.velocity, Velocity(1, 2).norm())
