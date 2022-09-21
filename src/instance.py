from dataclasses import dataclass, field
from numpy import ndarray, cumsum
from numpy.random import Generator, default_rng

from description import Description

@dataclass
class Endless:
    __description: Description
    _state: int = field(-1, init=False)
    _old_state: int = field(-1, init=False)
    _step: int = field(0, init=False)
    _state_rng: Generator = field(None, init=False)

    def __post_init__(self):
        self.state = self._pick_initial_state()
        self._state_rng = default_rng(self.__description.my_seed)

    @property
    def state(self) -> int:
        return self._old_state

    def _pick_initial_state(self) -> int:
        if isinstance(self.__description._initial_state, ndarray):
            pick: float = self._state_rng.random()
            accumulated = cumsum(self.__description._initial_state)
            for i, value in enumerate(accumulated):
                if pick < value:
                    return i
            else: raise ValueError('pick is higher than the last element of accumulated')    
        else:
            return self._initial_state

    def _pick_next_state(self) -> int:
        pick: float = self._state_rng.random()
        accumulated = cumsum(self.__description._matrix[self._state])
        for i, value in enumerate(accumulated):
            if pick < value:
                return i
        else: raise ValueError('pick is higher than the last element of accumulated')
    
    def __iter__(self):
        return self

    def __next__(self) -> int:
        self._old_state = self._state
        self._state = self._pick_next_state()
        self._step += 1
        return self._old_state