from unittest import TestCase

from entities import Bunny
from location import Location
from world import InteractionGrid


class TraitWorld(TestCase):
    def test_interaction_grid(self):
        b1 = Bunny(location=Location(100, 100))
        b2 = Bunny(location=Location(200, 200))
        b3 = Bunny(location=Location(210, 210))
        b4 = Bunny(location=Location(270, 270))
        entities = {b1, b2, b3, b4}
        i = list(InteractionGrid(50).get_interactions(entities))

        self.assertTrue((b2, b3) in i)
        self.assertTrue((b3, b4) in i)
        self.assertTrue((b1, b4) not in i)
