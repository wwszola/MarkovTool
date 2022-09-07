from typing import Iterator
from typing_extensions import Self
import numpy as np
from random import seed, random
from pathlib import Path

class MarkovChain(Iterator):
    def __init__(self, dimension: int = 0, max_steps: int = 10, my_seed: int = 0) -> None:
        if dimension >= 0:
            self._dimension: int = dimension
        else:
            raise ValueError("Dimension must be >= 0")

        self._stochastic_matrix: np.array = None
        self._state: int = 0

        self._step: int = 0
        self._max_steps: int = max_steps

        self._my_seed: int = my_seed

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        self._state = value

    @property
    def my_seed(self) -> int:
        return self._my_seed

    @my_seed.setter
    def my_seed(self, value: int) -> None:
        seed(value)
        self._my_seed = value

    def __iter__(self) -> Self:
        self._step = 0
        seed(self._my_seed)
        return self

    def __next__(self) -> int:
        if self._state >= 0 and self._step < self._max_steps:
            old: int = self._state
            self._state = self._pick_next_state()
            self._step += 1
            return old
        else:
            raise StopIteration

    def _pick_next_state(self) -> int:
        pick: float = random()
        accumulated = np.add.accumulate(self._stochastic_matrix[self._state])
        for i, value in enumerate(accumulated):
            if pick < value:
                return i
        raise StopIteration
        
    @staticmethod
    def txt_load(filepath: Path, *args, **kwargs) -> Self:
        '''Creates object from .txt file
        File must be formatted in a following way:
        0   dimension 
        1   state 0 weights     0.1, ..., 0.2,
        . . .        
        '''
        with filepath.open('r') as file:
            dim = int(file.readline())
            try:
                object = MarkovChain(dim, *args, **kwargs)
                object._stochastic_matrix = np.loadtxt(file, dtype=np.float32, ndmin=2, skiprows=0, delimiter=',')
                object._verify_matrix()
            except ValueError as err:
                print(f'Failed loading data from {filepath}')
                print(err)

        return object

    def _verify_matrix(self) -> None:
        '''Verify that _stochastich_matrix is right-stochastic matrix
        '''
        if np.allclose(np.sum(self._stochastic_matrix, 1), 1.0):
            pass
        else:
            raise ValueError('Matrix is not right-stochastic matric')
