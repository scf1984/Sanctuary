class World:
    _size = 42

    @classmethod
    @property
    def size(cls):
        return cls._size


def test_classmethod_property_chains():
    assert World.size == 42
