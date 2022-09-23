from numpy import ndarray, array, bincount

from .description import Description

class Collector:
    _count: int = 0
    def __init__(self):
        self._entries: dict[Description, dict[tuple[int, int], list[int]]] = {}
        self._is_open: bool = False
        self._id = Collector._count
        Collector._count += 1

    def __hash__(self) -> int:
        return self._id

    def open(self, *args):
        for instance in args:
            if callable(getattr(instance, "_bind_collector", None)):
                instance._bind_collector(self)
        self._is_open = True
    
    def close(self):
        self._is_open = False

    def put(self, d: Description, step: int, state: int) -> None:
        if d not in self._entries:
            self._entries[d] = {}
        
        for (step_, state_), chunk in self._entries[d].items():
            if step_ + len(chunk) == step:
                chunk.append(state)
                break
        else:
            self._entries[d][(step, state)] = [state]

    def count(self) -> ndarray: 
        return bincount(self._history)