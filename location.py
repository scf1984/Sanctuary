import math


class Location(object):
    def __init__(self, coords, y=None):
        if y is None:
            self.coords = coords
        else:
            self.coords = (coords, y)
        if not isinstance(coords, (int, float, tuple)) or not isinstance(y, (int, float)) and y is not None:
            raise ValueError('Tried to init location with something other than int/float.')

    def bbox(self, size):
        return (
            self.coords[0] - size,
            self.coords[1] + size,
            self.coords[0] + size,
            self.coords[1] - size
        )

    def __add__(self, other):
        return Location(tuple(a + b for a, b in zip(self.coords, other.coords)))

    def __sub__(self, other):
        return Location(tuple(a - b for a, b in zip(self.coords, other.coords)))

    def __eq__(self, other):
        return self.coords == other.coords

    def go_to(self, destination, dr):
        d = destination - self
        if d*d <= dr*dr:
            return destination
        else:
            return self + d.norm() * dr

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            return Location(tuple(a * other for a in self.coords))
        if isinstance(other, Location):
            return sum(tuple(a * b for a, b in zip(self.coords, other.coords)))

    def norm(self):
        inverse_magnitude = math.pow(self * self, -0.5)
        return self * inverse_magnitude
