from abc import ABCMeta, abstractmethod

from helper import Network


class ABCState(metaclass=ABCMeta):
    def __init__(self, animal):
        self.animal = animal

    @abstractmethod
    def update(self, dt): pass

    pass


class Idle(ABCState):
    def __init__(self, animal):
        super().__init__(animal)

    def update(self, dt):
        pass


class Walking(ABCState):
    def __init__(self, animal, destination, speed):
        super().__init__(animal)
        self.destination = destination
        self.speed = speed

    def update(self, dt):
        self.animal.location.go_to(self.destination, dt * self.speed)
        if self.animal.location == self.destination:
            self.animal.change_state(Idle(self.animal))


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
        Escaping: {Panting},
        Panting: {Walking}
    }
