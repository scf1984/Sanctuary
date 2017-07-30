import time

from location import Location


class Event(object):
    def __init__(self, action_function=None, ts=time.time(), dt=0):
        self.action_function = action_function
        self.ts = ts + dt

    def is_invalidated(self):
        return False

    def __lt__(self, other):
        return self.ts < other.ts

    def apply_event(self):
        if not self.is_invalidated():
            self.action_function()

# Attempt at event-driven interaction between animals
# class AnimalEnteredRangeEvent(Event):
#     interaction_range = 0
#
#     def __init__(self, animal1, animal2, action_function=None, **params):
#         self.animal1 = animal1
#         self.animal2 = animal2
#         if action_function is None:
#             def action_function():
#                 pass  # TODO
#
#         super().__init__(action_function=action_function, **params)
#
#     def is_invalidated(self):
#         if Location.square_distance(self.animal1.location, self.animal2.location) > self.interaction_range ** 2:
#             return True


class ChangeAnimalStateEvent(Event):
    def __init__(self, animal, future_state, ts=time.time(), dt=0):
        self.animal = animal
        self.future_state = future_state

        def action_function():
            self.animal.change_state(self.future_state)

        super().__init__(action_function, ts=ts, dt=dt)

    def is_invalidated(self):
        return self.animal.is_dead()
