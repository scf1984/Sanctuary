from unittest import TestCase

from entities import Bunny, Wolf


class TestEntities(TestCase):
    def test_mul(self):
        with self.assertRaises(ValueError):
            Bunny()*Wolf()
        self.assertEqual((Bunny()*Bunny()).__class__, Bunny)