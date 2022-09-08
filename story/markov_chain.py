from typing import Iterator, List, Tuple
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
        self.initial_state = initial_state

        self._step: int = 0
        self._iter_step: int = 0
        self.max_steps = max_steps

        self.my_seed = my_seed
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
    def iter_reset(self) -> bool:
        return self._iter_reset

    @iter_reset.setter
    def iter_reset(self, value: bool) -> None:
        self._iter_reset = value
            
    @property
    def initial_state(self) -> int:
        return self._initial_state
    
    @initial_state.setter
    def initial_state(self, value) -> None:
        self._initial_state = value

    @property
    def count(self) -> np.array:
        '''Histogram normalized to sum=1.0 
        '''
        sum = np.sum(self._count)
        if sum == 0:
            return np.zeros_like(self._count)
        else:
            return self._count/np.sum(self._count)

    def __iter__(self) -> Self:
        if self._iter_reset:
            self.reset()
        self._iter_step = self._step
        return self

    def __next__(self) -> int:
        if self._state >= 0 and self._step < self._iter_step + self.max_steps:
            
            old: int = self._state
            self._count[old] += 1
            self._state = self._pick_next_state()
            self._step += 1
            return old
        else:
            raise StopIteration
    
    def reset(self) -> None:
        self._step = 0
        self._state = self._initial_state
        self._count[:] = 0
        seed(self._my_seed)

    def run(self, record: bool = False) -> List[int]:
        '''Runs max_steps number of states
        If record is true returns list of states the chain went through
        If record is false returns None
        '''
        tape: List[int] = None
        if record:
            tape = []
        for out in self:
            if record:
                tape.append(out)
        return tape

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