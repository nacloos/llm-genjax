
registry = {}


def make(id):
    return registry[id]


def register(id, obj=None):
    if obj is None:
        # decorator
        def _register(obj):
            registry[id] = obj
            return obj
        return _register
    else:
        registry[id] = obj
        return obj
