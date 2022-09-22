from dataclasses import dataclass, field, asdict
from numpy import array, ndarray, float32, allclose, newaxis
from numpy.random import Generator, default_rng
from copy import copy
from typing_extensions import Self

@dataclass
class Description:
    _dimension: int = None
    _my_seed: int = None
    _matrix: ndarray = field(default = None, init = False)
    _initial_state: int | ndarray = field(default = None, init = False)

    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        else:
            raise ValueError('Dimension isn\'t defined yet')

    @property
    def my_seed(self) -> int:
        return self._my_seed

    @my_seed.setter
    def my_seed(self, value: int) -> None:
        self._my_seed = value

    @property
    def matrix(self) -> ndarray:
        return copy(self._matrix)

    @matrix.setter
    def matrix(self, value: ndarray) -> None:
        if value is not None:
            self._matrix = self._verify_matrix(value)

    def _verify_matrix(self, value: list[list[float]] | ndarray) -> ndarray:
        value = array(value, dtype=float32)
        if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
            raise ValueError('Matrix should be an array NxN in size')

        if self._dimension is not None and value.shape[0] != self._dimension:
            raise ValueError('Matrix dimension should be equal to the original dimension')
        
        value /= value.sum(1)[:, newaxis]

        if not allclose(value.sum(1), 1.0):
            raise ValueError('Matrix should be a right-stochastic matrix')
        
        return value

    @property
    def initial_state(self) -> int | ndarray:
        return copy(self._initial_state)
    
    @initial_state.setter
    def initial_state(self, value: int | list | ndarray) -> None:
        self._initial_state = self._verify_initial_state(value)

    def _verify_initial_state(self, value: int | list | ndarray) -> int | ndarray:
        if isinstance(value, (list, ndarray)):
            if len(value) != self._dimension:
                raise ValueError('Initial state dimension should be equal to the original dimension')

            value /= value.sum()[newaxis]         

            if allclose(value.sum(), 1.0):
                return array(value, dtype = float32)
            else:
                raise ValueError('Initial state probabilities should normalize to sum 1.0')
        
        elif isinstance(value, int):
            if value < 0 or value >= self._dimension:
                raise ValueError(f'Invalid initial state: int {value}')
            else:
                return value
        else:
            raise TypeError('Initial state should be the type of either int, list or numpy.ndarray')

    def variant(self, **kwargs) -> Self:
        result = copy(self)
        for name, value in kwargs.items():
            if hasattr(result, name):
                setattr(result, name, value)
        return result

    @staticmethod
    def from_array(matrix: ndarray, initial_state: ndarray) -> Self:
        object: Description = None
        try:
            object = Description()
            object.matrix = matrix
            object._dimension = object.matrix.shape[0]
            object.initial_state = initial_state
        except ValueError as err:
            print(f'Failed loading data from an array')
            print(err)
        return object   

    @staticmethod
    def random(dimension: int, seed_: int = None) -> Self:
        rng = default_rng(seed_)
        matrix = rng.random((dimension, dimension))
        initial_state = rng.random(dimension)
        object = Description.from_array(matrix, initial_state)
        object.my_seed = seed_
        return object