import math
from random import gauss

from utils import get_all_subclasses


class Trait:
    value = 0.0
    inherit_gain = 1.1

    @classmethod
    @property
    def name(cls):
        return cls.__name__

    def __init__(self, value):
        self.value = value

    def __mul__(self, other):
        if self.__class__ != other.__class__:
            raise ValueError("Cannot inherit two different traits!")
        mean = (self.value + other.value) / 2.0
        stddev = math.sqrt((self.value - mean) ** 2 + (other.value - mean) ** 2)
        min_value = min(self.value, other.value) / self.inherit_gain
        max_value = max(self.value, other.value) * self.inherit_gain
        while True:
            x = gauss(mean, stddev)
            if min_value <= x <= max_value:
                return self.__class__(value=x)


class Metabolism(Trait):
    # name = 'metabolism'
    pass


class AgingRate(Trait):
    # name = 'aging_rate'
    pass


class ThirstRate(Trait):
    # name = 'thirst_rate'
    pass


class MaxSpeed(Trait):
    # name = 'max_speed'
    pass


class SightRange(Trait):
    # name = 'sight_range'
    pass


class Fidgetiness(Trait):
    # name = 'Fidgetiness'
    pass


class SightAngle(Trait):
    # name = 'sight_angle'
    pass


#
#
# class SprintRange(Trait):
#     name = 'sprint_range'
#
#
# class Smell(Trait):
#     name = 'scent_sensitivity'
#
#
# class Scent(Trait):
#     name = 'scent_emit'
#

class PregnancyRate(Trait):
    # name = 'pregnancy_rate'
    pass

#
# class LifeExpectancy(Trait):
#     name = 'class_expectancy'
#
#
# class Meatiness(Trait):
#     name = 'meat'


# class Gender(Trait):
#     name = 'gender'
#     def __mul__(self, other):
#         return [self, other][random.randint(0, 1)]


all_traits = {c: c for c in get_all_subclasses(Trait)}
