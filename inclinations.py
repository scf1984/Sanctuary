from abc import ABCMeta

from math import sin


class ABCInclination(metaclass=ABCMeta):
    inclination = 0

    def update(self): pass
    def get_value(self): pass


class InHeat(ABCInclination):
    def __init__(self, period: float, phase: float):
        self.period = period
        self.phase = phase
        self.dt = 0

    def update(self, t: float=None, dt: float=None):
        if t is None:
            self.dt = dt
        else:
            self.dt = t
        self.inclination = sin(self.phase + self.period / self.dt)


class Hungry(ABCInclination):
    pass


class Thirsty(ABCInclination):
    pass


