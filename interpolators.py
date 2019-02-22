from abc import ABC, abstractmethod
from math import sin


class Interpolator(ABC):
    @abstractmethod
    def get_value(self, t):
        pass


class Sinusoidal(Interpolator):
    def __init__(self, phase, period, t0):
        self.phase = phase
        self.period = period
        self.t0 = t0

    def get_value(self, t):
        return (sin(self.phase + self.period * (t - self.t0)) + 1) / 2


class Linear(Interpolator):
    def __init__(self, t0, t1):
        self.t0 = t0
        self.t1 = t1

    def get_value(self, t):
        if t <= self.t0:
            return 0.0
        elif t >= self.t1:
            return 1.0
        else:
            return (t - self.t0) / (self.t1 - self.t0)
