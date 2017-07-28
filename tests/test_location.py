from unittest import TestCase

from location import Location


class TestLocation(TestCase):
    l = Location(2.0, 2)

    def test_init(self):
        with self.assertRaises(ValueError):
            Location('a', 0.5)

    def test_eq(self):
        self.assertTrue(self.l == self.l)

    def test_add(self):
        self.assertEqual((Location(1, 1) + Location(2, 3)).coords, (3, 4))

    def test_subtract(self):
        self.assertEqual((Location(1, 1) - Location(2, 3)).coords, (-1, -2))

    def test_norm(self):
        self.assertAlmostEqual(self.l.norm() * self.l.norm(), 1.0)

    def test_mul(self):
        self.assertAlmostEqual(self.l * self.l, 8.0)

    def test_go_to(self):
        self.assertAlmostEqual(self.l.go_to(Location(0, 2), 1), Location(1, 2))
