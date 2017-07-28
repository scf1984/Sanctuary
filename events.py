import time


class Event(object):
    def __init__(self, action_function, params, ts):
        self.action_function = action_function
        self.params = params
        self.ts = ts

    def is_invalidated(self):
        return False

    def __lt__(self, other):
        return self.ts < other.ts

    def apply_event(self):
        if not self.is_invalidated():
            self.action_function(**self.params)


class ChangeStateEvent(Event):
    def __init__(self, animal, future_state, ts=time.time(), dt=None):
        if dt is not None:
            ts += dt
        self.params = {'animal': animal,
                       'next': future_state}

        def action_function(**params):
            params['animal'].change_state(params['next'])

        super().__init__(action_function, self.params, ts)

    def is_invalidated(self):
        return self.params['animal'].is_dead()
