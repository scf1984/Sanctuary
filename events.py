from time import time


class Event(object):
    def __init__(self, action_function=None, ts=time(), dt=0):
        self.action_function = action_function
        self.ts = ts + dt

    def is_invalidated(self):
        return False

    def __lt__(self, other):
        return self.ts < other.ts

    def apply_event(self, world):
        if not self.is_invalidated():
            self.action_function(world)


class ChangeEntityStateEvent(Event):
    def __init__(self, entity, future_state, ts=time(), dt=0):
        self.entity = entity
        self.future_state = future_state

        def action_function(world):
            self.entity.change_state(self.future_state(entity))

        super().__init__(action_function, ts=ts, dt=dt)

    def is_invalidated(self):
        return self.entity.is_dead()


class CreateEntityEvent(Event):
    def __init__(self, entity, ts=time(), dt=0):

        def action_function(world):
            world.entities.add(entity)

        super().__init__(action_function=action_function, ts=ts, dt=dt)

    def is_invalidated(self):
        return False


class ChangeEntityEvent(Event):
    def __init__(self, old_entity, new_entity, ts=time(), dt=0):

        def action_function(world):
            world.entities.remove(old_entity)
            if new_entity is not None:
                world.entities.add(new_entity)

        super().__init__(action_function=action_function, ts=ts, dt=dt)

    def is_invalidated(self):
        return False
