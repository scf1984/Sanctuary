from random import choice


class ReproductionSystem(object):
    def possible_genders(self): return {}

    def interact_sex(self, other): pass


class Sexual(ReproductionSystem):
    def __init__(self):
        self.gender = choice(self.possible_genders())
        self.baby = None

    def possible_genders(self): return GENDER.MALE, GENDER.FEMALE

    def interact_sex(self, other):
        if other.__class__ == self.__class__:
            if self.gender == GENDER.FEMALE and other.gender == GENDER.MALE:
                # TODO: add random, avoid multiple sex immediately.
                self.have_sex(other)

    def have_sex(self, other):
        if GENDER.is_pregnable(self.gender) and self.baby is None:
            self.baby = self * other


class Unisexual(ReproductionSystem):
    @property
    def possible_genders(self):
        return GENDER.UNISEX,


class Asexual(ReproductionSystem):
    @property
    def possible_genders(self):
        return GENDER.ASEXUAL,


class GENDER(object):
    MALE = 1
    FEMALE = 2
    UNISEX = 3
    ASEXUAL = 4
    SPAWN = 5
    DOESNT_REPRODUCE = 6

    @staticmethod
    def is_pregnable(g):
        return g in {GENDER.FEMALE, GENDER.UNISEX}
