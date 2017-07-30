from unittest import TestCase

from entities import Bunny
from location import Location, Velocity
from states import Walking


class TestStates(TestCase):
    def test_walking_velocity(self):
        b = Bunny(location=Location(0, 0))
        b.change_state(Walking(b, Location(1, 2), 1))
        self.assertEqual(b.get_velocity(), Velocity(1, 2).norm())
