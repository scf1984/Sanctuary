def get_all_subclasses(cls):
    return [c for c in cls.__subclasses__()]


class Network(object):
    dict = {}
    default = None
    network_base_class = None

    def __getitem__(self, item):
        return self.dict.get(item, self.default)

    @classmethod
    def edge_exists(cls, source, target):
        return target in dict[source]
