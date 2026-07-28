from unittest import TestCase

from location import Location


class TestLocation(TestCase):


    def test_init(self):
        with self.assertRaises(ValueError):
            Location('a', 0.5)

    def test_eq(self):
        l = Location(2.0, 2)

        self.assertTrue(l == l)

    def test_add(self):
        self.assertEqual((Location(1, 1) + Location(2, 3)).coords, (3, 4))

    def test_subtract(self):
        self.assertEqual((Location(1, 1) - Location(2, 3)).coords, (-1, -2))

    def test_norm(self):
        l = Location(2.0, 2)
        self.assertAlmostEqual(l.norm() * l.norm(), 1.0)

    def test_mul(self):
        l = Location(2.0, 2)
        self.assertAlmostEqual(l * l, 8.0)

    def test_go_to(self):
        l = Location(2.0, 2)
        d = Location(0, 2)
        self.assertAlmostEqual(l.go_to(d, 1), Location(1, 2))

    def test_go_to_arrive(self):
        l = Location(2.0, 2)
        d = Location(0, 2)
        self.assertTrue(l.go_to(d, 10) is d)
