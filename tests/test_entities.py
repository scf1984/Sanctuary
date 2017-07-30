from unittest import TestCase

from entities import Bunny, Wolf
from location import Location, Velocity
from states import Walking


class TestEntities(TestCase):
    def test_mul(self):
        with self.assertRaises(ValueError):
            Bunny()*Wolf()
        self.assertEqual((Bunny()*Bunny()).__class__, Bunny)

    def test_in_range_time(self):
        b1 = Bunny(location=Location(0, 0))
        b1.change_state(Walking(b1, Location(1, 2), 1))
        self.assertEqual(b1.get_velocity(), Velocity(1, 2).norm())

        b2 = Bunny(location=Location(0, 0))
        b2.change_state(Walking(b2, Location(1, 2), 1))
        self.assertEqual(b2.get_velocity(), Velocity(1, 2).norm())





