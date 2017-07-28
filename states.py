from abc import ABCMeta, abstractmethod

from helper import Network


class ABCState(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self): pass
    @abstractmethod
    def update(self, animal, dt): pass

    pass


class Idle(ABCState):
    def __init__(self, animal):
        pass

    def update(self, animal, dt):
        pass


class Walking(ABCState):
    pass


class Prowling(ABCState):
    pass


class Chasing(ABCState):
    pass


class Panting(ABCState):
    pass


class Escaping(ABCState):
    pass


class InHeat(ABCState):
    pass


class Hungry(ABCState):
    pass


class Thirsty(ABCState):
    pass


class Injured(ABCState):
    pass


class StateTransitions(Network):
    dict = {
        Chasing: {Panting},
        Escaping: {Panting}
    }
