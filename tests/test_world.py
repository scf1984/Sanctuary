from unittest import TestCase

from entities import Bunny
from location import Location
from world import InteractionGrid


class TraitWorld(TestCase):
    def test_interaction_grid(self):
        l1 = Location(100, 100)
        l2 = Location(200, 200)
        animals = {
            Bunny(location=l1),
            Bunny(location=l1)
        }

        i = InteractionGrid(10).get_interactions(animals)
        self.assertTrue((animals.pop(), animals.pop()) in i)

        animals = {
            Bunny(location=l1),
            Bunny(location=l2)
        }
        i = InteractionGrid(10).get_interactions(animals)
        self.assertFalse((animals.pop(), animals.pop()) in i)
