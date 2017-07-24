from abc import ABCMeta, abstractmethod


class ABCState(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self): pass

    @abstractmethod
    def update(self, animal, dt): pass

    pass


class Idle(ABCState):
    def __init__(self, animal):
        super().__init__()


    def update(self, animal, dt):
        pass


class Walking(ABCState):
    pass


class Prowling(ABCState):
    pass


class Chasing(ABCState):
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
