from dataclasses import dataclass, field
from copy import deepcopy
from typing import Callable
from typing_extensions import Self
from numpy import ndarray, cumsum
from numpy.random import Generator, default_rng
from itertools import islice

from .description import Description, Variation

@dataclass()
class Endless:
    _description: Description | Variation
    _state: int = field(default = -1, init = False)
    _old_state: int = field(default = -1, init = False)
    _step: int = field(default = 0, init = False)
    _state_rng: Generator = field(default = None, init = False)

    @property
    def state(self) -> int:
        return self._old_state

    @state.setter 
    def state(self, value: int) -> None:
        if value >= 0 and value < self._description.dimension:
            self._state = value
        else:
            raise ValueError(f'State should be int in range [0, {self._description.dimension})')

    def __post_init__(self):
        self._state_rng = default_rng(self._description.my_seed)
        self.state = self._pick_initial_state()

    def __deepcopy__(self, memo: dict):
        result = Endless.__new__(Endless)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

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
        else: raise ValueError('pick is higher than the last element of accumulated')

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        self._old_state = self._state
        self._state = self._pick_next_state()
        self._step += 1
        return self._old_state

    def take(self, n: int) -> list:
        return list(islice(self, n))

    def skip(self, n: int) -> None:
        next(islice(self, n, n), None)

@dataclass
class Finite(Endless):
    _stop_predicate: Callable[[Self], bool] = field(
        default = lambda self: False,
        init = True)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        if self._stop_predicate(self):
            raise StopIteration
        return super().__next__()