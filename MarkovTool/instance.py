from dataclasses import dataclass, field
from copy import deepcopy, copy
from typing import Callable, Iterable
from typing_extensions import Self
from numpy import ndarray, cumsum
from numpy.random import Generator, default_rng
from itertools import islice

from .description import Description
from .stat import Collector

class Endless(Iterable):
    _count = 0
    def __init__(self, description: Description) -> None:
        self._description = description
        self.state: int = self._pick_initial_state()
        self._old_state: int = -1
        self._step: int = 0
        self._state_rng: Generator = default_rng(self._description.my_seed)
    
        self._collectors: set[Collector] = set()

        self._id = Endless._count
        Endless._count += 1

    @property
    def state(self) -> int:
        return self._old_state

    @state.setter 
    def state(self, value: int) -> None:
        if value >= 0 and value < self._description.dimension:
            self._state = value
        else:
            raise ValueError(f'State should be int in range [0, {self._description.dimension})')

    def __eq__(self, other: Self) -> bool:
        return self._description == other._description \
           and self._step == other._step \
           and self._state == self._state

    def __deepcopy__(self, memo: dict) -> Self:
        result = Endless.__new__(Endless)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            match k:
                case "_description":
                    setattr(result, k, v)
                case _:
                    setattr(result, k, deepcopy(v, memo))
        return result

    def _bind_collector(self, collector: Collector) -> None:
        self._collectors.add(collector)

    def _unbind_collector(self, collector: Collector) -> None:
        self._collectors.remove(collector)            

    def _emit(self, step: int, state: int) -> None:
        if len(self._collectors) == 0:
            return
        else:
            for collector in self._collectors:
                if collector._is_open:
                    collector.put(self._description, self._id, step, state)

    def _pick_initial_state(self) -> int:
        if isinstance(self._description.initial_state, ndarray):
            rng = default_rng(None)
            pick: float = rng.random()
            accumulated = cumsum(self._description.initial_state)
            for i, value in enumerate(accumulated):
                if pick < value:
                    return i
            else: raise ValueError('pick is higher than the last element of accumulated')    
        else:
            return self._description.initial_state

    def _pick_next_state(self) -> int:
        pick: float = self._state_rng.random()
        accumulated = cumsum(self._description.matrix[self._state])
        for i, value in enumerate(accumulated):
            if pick < value:
                return i
        else: 
            print(pick ,accumulated)
            raise ValueError('pick is higher than the last element of accumulated')

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        self._old_state = self._state
        self._emit(self._step, self._old_state)
        self._state = self._pick_next_state()
        self._step += 1
        return self._old_state

    def take(self, n: int = None) -> list:
        if n is None:
            return list(self)
        else:
            return list(islice(self, n))

    def skip(self, n: int = None) -> None:
        if n is None:
            for _ in self: pass
        else:
            next(islice(self, n, n), None)

    def branch(self, **kwargs) -> Self:
        new = copy(self)
        new._id = Endless._count
        Endless._count += 1
        new._state_rng = deepcopy(self._state_rng)
        if '_state' in kwargs:
            new.state = kwargs['_state']
            del kwargs['_state']
        if kwargs:
            new._description = copy(self._description)
            for k, v in kwargs.items():
                setattr(new._description, k, v)
        return new 

class Finite(Endless):
    def __init__(self, description: Description,
                 stop_predicate: Callable[[Self], bool] = lambda self: False):
        super().__init__(description)
        self._stop_predicate = stop_predicate
            
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        if self._stop_predicate(self):
            raise StopIteration
        return super().__next__()