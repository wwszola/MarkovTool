from typing import Iterator
from typing_extensions import Self
import numpy as np
from random import seed, random
from pathlib import Path

class MarkovChain(Iterator):
    def __init__(self, dimension: int = 0, initial_state: int = 0, max_steps: int = 10, my_seed: int = 0, iter_reset: bool = True) -> None:
        if dimension >= 0:
            self._dimension: int = dimension
        else:
            raise ValueError("Dimension must be >= 0")

        self._stochastic_matrix: np.array = None
        self._state: int = initial_state
        self._initial_state: int = initial_state

        self._step: int = 0
        self._max_steps: int = max_steps

        self._my_seed: int = my_seed
        self._iter_reset: bool = iter_reset

        self._count = np.zeros(dimension, dtype=np.int32)

    @property
    def max_steps(self) -> int:
        return self._max_steps
    
    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self._max_steps = value

    @property
    def my_seed(self) -> int:
        return self._my_seed

    @my_seed.setter
    def my_seed(self, value: int) -> None:
        seed(value)
        self._my_seed = value
    
    @property
    def initial_state(self) -> int:
        return self._initial_state
    
    @initial_state.setter
    def initial_state(self, value) -> None:
        self._initial_state = value

    @property
    def count(self):
        '''States occurence count normalized to sum=1.0 
        '''
        sum = np.sum(self._count)
        if sum == 0:
            return np.zeros_like(self.count)
        else:
            return self._count/np.sum(self._count)

    def __iter__(self) -> Self:
        if self._iter_reset:
            self._step = 0
            self._count[:] = 0
            seed(self._my_seed)
        return self

    def __next__(self) -> int:
        if self._state >= 0:
            if self._step != 0 and self._step % self.max_steps == 0:
                raise StopIteration
            old: int = self._state
            self._count[old] += 1
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
