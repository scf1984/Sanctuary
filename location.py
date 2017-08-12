import math

from helper import Vector


class Velocity(Vector):
    pass


class Location(Vector):
    def bbox(self, size):
        return (
            self.coords[0] - size,
            self.coords[1] + size,
            self.coords[0] + size,
            self.coords[1] - size
        )

    def go_to(self, destination, dr):
        d = destination - self
        if d * d <= dr * dr:
            return destination
        else:
            coords = (self + d.norm() * dr).coords
        return Location(coords)

    @staticmethod
    def square_distance(l1, l2):
        return (l1.coords[0] - l2.coords[0])**2 + (l1.coords[1] - l2.coords[1])**2

    @staticmethod
    def distance(l1, l2):
        return math.sqrt(Location.square_distance(l1, l2))
