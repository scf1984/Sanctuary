from utils import get_all_subclasses


class Stat:
    is_state_setter = False
    max_value = 100.0
    updater = None
    threshold = 80.0
    inclination = None
    name = None

    def __init__(self):
        self.value = 0

    def __gt__(self, other):
        return self.value > other.value

    @classmethod
    @property
    def all_stats(cls) -> dict:
        return {c: c for c in get_all_subclasses(Stat)}

    def update(self, entity, world, dt=None, ds=None):
        if ds is None:
            ds = dt * (entity.traits[self.updater].value or 1.0)
        if (
                (self.value - self.threshold) * (self.value + ds - self.threshold) < 0
                or (self.value - self.threshold) == 0
        ):
            self.threshold_function(entity)
            if self.inclination not in entity.inclinations:
                entity.inclinations.add(self.inclination)
            else:
                entity.inclinations.remove(self.inclination)
        self.value += ds

    def threshold_function(self, entity):
        pass
