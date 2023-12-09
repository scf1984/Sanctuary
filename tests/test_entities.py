from unittest import TestCase

from entities import Bunny, Wolf
from location import Location, Velocity
from states import WalkingState


class TestEntities(TestCase):
    def test_mul(self):
        bunny = Bunny(Location(0, 0))
        wolf = Wolf(Location(0, 0))
        with self.assertRaises(ValueError):
            bunny * wolf
        self.assertEqual((bunny * bunny).__class__, Bunny)

    def test_in_range_time(self):
        b1 = Bunny(location=Location(0, 0))
        b1.change_state(WalkingState(b1, 1, Location(1, 2)))
        self.assertEqual(b1.velocity(), Velocity(1, 2).norm())

        b2 = Bunny(location=Location(0, 0))
        b2.change_state(WalkingState(b2, 1, Location(1, 2)))
        self.assertEqual(b2.velocity(), Velocity(1, 2).norm())
