import math
from random import uniform


def get_all_subclasses(cls):
    return [[c] + get_all_subclasses(c) if len(c.__subclasses__()) > 0 else c for c in cls.__subclasses__()]


class Network(object):
    dict = {}
    default = None
    network_base_class = None

    def __getitem__(self, item):
        return self.dict.get(item, self.default)

    @classmethod
    def edge_exists(cls, source, target):
        return target in dict[source]


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
        if isinstance(other, self.__class__):
            return sum((self.coords[0] * other.coords[0], self.coords[1] * other.coords[1]))

    def norm(self):
        inverse_magnitude = math.pow(self.square_magnitude(), -0.5)
        return self * inverse_magnitude

    @classmethod
    def random(cls, x_range, y_range):
        return cls(uniform(*x_range), uniform(*y_range))

    def square_magnitude(self):
        return self * self