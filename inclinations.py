from abc import abstractmethod, ABC

from interpolators import Interpolator


class ABCInclination(ABC):
    inclination = 0

    @abstractmethod
    def update(self, t):
        pass

    @abstractmethod
    def get_value(self, t):
        pass


class InHeat(ABCInclination):
    def get_value(self, t):
        self.interpolator.get_value(t)

    def __init__(self, interpolator: Interpolator):
        self.interpolator = interpolator

    def update(self, t: float):
        self.inclination = self.interpolator.get_value(t)


class Hungry(ABCInclination):
    def update(self, t):
        pass

    def get_value(self, t):
        pass


class Thirsty(ABCInclination):
    def update(self, t):
        pass

    def get_value(self, t):
        pass
