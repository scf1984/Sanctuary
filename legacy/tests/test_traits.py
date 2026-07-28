from random import seed
from unittest import TestCase

from traits import Metabolism, AgingRate


class TraitTests(TestCase):

    def test_trait(self):
        # basic qualities
        self.assertEqual(Metabolism.name, 'Metabolism')
        self.assertEqual(Metabolism.inherit_gain, 1.1)
        self.assertEqual(AgingRate(1).value, 1)

        # multiplication
        seed(1)
        self.assertAlmostEqual((Metabolism(1) * Metabolism(2)).value, 1.5469065003357403)
        with self.assertRaises(ValueError):
            Metabolism(1)*AgingRate(9)
