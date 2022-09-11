from typing import Dict, Iterator, List
from typing_extensions import Self
import numpy as np
from numpy.random import Generator, PCG64
from pathlib import Path
from time import time_ns
from copy import deepcopy
from contextlib import contextmanager

class MarkovChain(Iterator):
    _static_rng_seed = time_ns()
    _static_rng = Generator(PCG64(_static_rng_seed))
    @staticmethod
    def reset_static_rng(a: int = None) -> None:
        if a is not None:
           MarkovChain._static_rng_seed = a
        MarkovChain._static_rng = Generator(PCG64(MarkovChain._static_rng_seed))

    def __init__(self, dimension: int = 0, initial_state: int | List = 0, max_steps: int = 10, my_seed: int = 0, iter_reset: bool = True) -> None:
        if dimension >= 0:
            self._dimension: int = dimension
        else:
            raise ValueError("Dimension must be >= 0")

        self._stochastic_matrix: np.ndarray = None
        self.initial_state = initial_state
        self._state: int = self._pick_initial_state()

        self._step: int = 0
        self._iter_step: int = 0
        self.max_steps = max_steps

        self._state_rng: Generator = None
        self.my_seed = my_seed
        self.iter_reset: bool = iter_reset

        self._count = np.zeros(dimension, dtype=np.int32)

    @property
    def matrix(self) -> np.ndarray:
        return self._stochastic_matrix

    @matrix.setter
    def matrix(self, matrix: np.ndarray) -> None:
        '''Verifies that _stochastich_matrix is right-stochastic matrix
        '''
        if np.allclose(np.sum(matrix, 1), 1.0):
            self._stochastic_matrix = matrix
        else:
            raise ValueError('Matrix is not right-stochastic matric')

    @property
    def state(self) -> int:
        return self._state

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
        self._state_rng = Generator(PCG64(value))
        self._my_seed = value
    
    @property
    def iter_reset(self) -> bool:
        return self._iter_reset

    @iter_reset.setter
    def iter_reset(self, value: bool) -> None:
        self._iter_reset = value
            
    @property
    def initial_state(self) -> int | List:
        return self._initial_state
    
    @initial_state.setter
    def initial_state(self, value: int | List | np.ndarray) -> None:
        if isinstance(value, (List, np.ndarray)) and np.allclose(np.sum(value), 1.0):
            self._initial_state = np.array(value, dtype=np.float32)
        elif isinstance(value, int):
            self._initial_state = value

    @property
    def count(self) -> np.ndarray:
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
        if self._step < self._iter_step + self.max_steps:
            old: int = self._state
            self._count[old] += 1
            self._state = self._pick_next_state()
            self._step += 1
            return old
        else:
            raise StopIteration
    
    def reset(self) -> None:
        self._step = 0
        self._state = self._pick_initial_state()
        self._count[:] = 0
        self.my_seed = self.my_seed

    def __deepcopy__(self, memo: Dict) -> Self:
        result = MarkovChain.__new__(MarkovChain)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
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

    def _pick_initial_state(self) -> int:
        if isinstance(self._initial_state, np.ndarray):
            pick: float = MarkovChain._static_rng.random()
            accumulated = np.add.accumulate(self._initial_state)
            for i, value in enumerate(accumulated):
                if pick < value:
                    return i
        else:
            return self._initial_state

    def _pick_next_state(self) -> int:
        pick: float = self._state_rng.random()
        accumulated = np.add.accumulate(self._stochastic_matrix[self._state])
        for i, value in enumerate(accumulated):
            if pick < value:
                return i
        raise StopIteration

    @staticmethod
    def txt_load(filepath: Path, **kwargs) -> Self:
        '''Creates object from .txt file
        File must be formatted in a following way:
        0   dimension 
        1   state 0 weights     0.1, ..., 0.2,
        . . .        
        '''
        object: MarkovChain = None
        with filepath.open('r') as file:
            dim = int(file.readline())
            try:
                object = MarkovChain(dim, **kwargs)
                object.matrix = np.loadtxt(file, dtype=np.float32, ndmin=2, skiprows=0, delimiter=',')
            except ValueError as err:
                print(f'Failed loading data from {filepath}')
                print(err)

        return object
    
    @staticmethod
    def from_array(matrix: np.ndarray, **kwargs) -> Self:
        object: MarkovChain = None
        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
            dim = matrix.shape[0]
            try:                
                object = MarkovChain(dim, **kwargs)
                object.matrix = (matrix)
            except ValueError as err:
                print(f'Failed loading data from array')
        return object            

    @staticmethod
    def random(dimension: int, **kwargs) -> Self:
        matrix = MarkovChain._static_rng.random((dimension, dimension))
        matrix /= matrix.sum(1)[:, np.newaxis]
        if "initial_state" not in kwargs:
            initial_state = MarkovChain._static_rng.random(dimension)
            initial_state /= initial_state.sum()[np.newaxis]
            kwargs["initial_state"] = initial_state
        return MarkovChain.from_array(matrix, **kwargs)

    @contextmanager
    def simulation(self, **kwargs) -> Self:
        copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(copy, k, v)
        yield copy
        del copy