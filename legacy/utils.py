from math import atan2
from random import uniform



def get_all_subclasses(cls):
    return [[c] + get_all_subclasses(c) if len(c.__subclasses__()) > 0 else c for c in cls.__subclasses__()]


class Network(object):
    edges_dict = {}
    default = None
    network_base_class = None

    def __init__(self, entity):
        self.entity = entity
        self.current_state = None

    def __getitem__(self, item):
        return self.edges_dict.get(item, None)

    def set_current_state(self, target):
        if self.current_state is None or target.__class__ in self.edges_dict[self.current_state.__class__]:
            self.current_state = target
        else:
            raise ValueError('Illegal transition, from {0} to {1}'
                             .format(self.current_state.__class__, target.__class__))


class Vector:
    __slots__ = ['coords']

    def __init__(self, coords, y=None):
        if not isinstance(coords, (int, float, tuple)) or not isinstance(y, (int, float)) and y is not None:
            raise ValueError('Tried to init location with something other than int/float/tuple.')
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
        if isinstance(other, (float, int)):
            return c(self.coords[0] * other, self.coords[1] * other)
        return sum((self.coords[0] * other.coords[0], self.coords[1] * other.coords[1]))

    @property
    def angle(self):
        return atan2(*self.coords)

    def norm(self):
        inverse_magnitude = self.square_magnitude ** -0.5
        return self * inverse_magnitude

    @classmethod
    def random(cls, x_range=None, y_range=None):
        if x_range is None:
            return cls(uniform(-1, 1), uniform(-1, 1)).norm()
        return cls(uniform(*x_range), uniform(*y_range))

    @property
    def square_magnitude(self):
        return self * self


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def ilen(generator):
    return sum(1 for _ in generator)
