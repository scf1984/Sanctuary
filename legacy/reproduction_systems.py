from dataclasses import dataclass, field
from enum import Enum
from world import Blackboard, Space


class ProcreateMethod(Enum):
    ASEXUAL = 0
    SEXUAL = 1
    LEECHING = 2


@dataclass
class ReproductionSystem:
    entity_id: str = None
    genders: set = field(init=False, default_factory=set)
    same_species: bool = field(init=False, default=True)
    more_species: set = field(init=False, default_factory=set)

    def procreate(self, other):
        pass

    def mates_with(self, other):
        if self.same_species and Blackboard().read(Space.ENTITIES, self.entity_id).__class__ is other.__class__:
            return True
        if other.__class__ in self.more_species:
            return True
        return False


class Gender(Enum):
    MALE = 1
    FEMALE = 2
    UNISEX = 3
    ASEXUAL = 4

    @staticmethod
    def is_pregnable(g):
        return g in {Gender.FEMALE, Gender.UNISEX}


class TwoSexual(ReproductionSystem):
    genders = {Gender.MALE, Gender.FEMALE}


class OneGenderSexual(ReproductionSystem):
    genders = {Gender.UNISEX}


class Asexual(ReproductionSystem):
    genders = Gender.ASEXUAL


class Spawn(ReproductionSystem):
    pass
