from math import acos, sqrt, pow
from random import uniform


def get_all_subclasses(cls):
    return [[c] + get_all_subclasses(c) if len(c.__subclasses__()) > 0 else c for c in cls.__subclasses__()]


class Network(object):
    edges_dict = {}
    default = None
    network_base_class = None

    def __getitem__(self, item):
        return self.edges_dict.get(item)

    @classmethod
    def edge_exists(cls, source, target):
        return target in cls.edges_dict[source]


class Vector(object):
    def __init__(self, coords, y=None):
        if not isinstance(coords, (int, float, tuple)) or not isinstance(y, (int, float)) and y is not None:
            raise ValueError('Tried to init location with something other than int/float.')
        if y is None:
            self.coords = coords
        else:
            self.coords = (coords, y)

    def __add__(self, other):
        return self.__class__(self.coords[0] + other.coords[0], self.coords[1] + other.coords[1])

    def __sub__(self, other):
        return self.__class__(self.coords[0] - other.coords[0], self.coords[1] - other.coords[1])

    def __eq__(self, other):
        return self.coords == other.coords

    def __mul__(self, other):
        c = self.__class__
        if self.__class__ != other.__class__ and issubclass(other.__class__, Vector):
            c = Vector
        if isinstance(other, (float, int)):
            return c(self.coords[0] * other, self.coords[1] * other)
        return sum((self.coords[0] * other.coords[0], self.coords[1] * other.coords[1]))

    def angle(self, other):
        nom = self * other
        den = sqrt(self.square_magnitude() * other.square_magnitude())
        if den == 0.0 or abs(nom) > abs(den):
            return 0.0
        return acos(nom / den)

    def norm(self):
        inverse_magnitude = self.square_magnitude()**-0.5
        return self * inverse_magnitude

    @classmethod
    def random(cls, x_range=None, y_range=None):
        if x_range is None:
            return cls(uniform(-1, 1), uniform(-1, 1)).norm()
        return cls(uniform(*x_range), uniform(*y_range))

    def square_magnitude(self):
        return self * self