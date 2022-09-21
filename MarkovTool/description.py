from dataclasses import dataclass, field, asdict
from numpy import array, ndarray, float32, allclose, newaxis
from numpy.random import Generator, default_rng
from copy import copy

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
    def matrix(self, matrix: ndarray) -> None:
        if matrix is not None:
            self._matrix = self._verify_matrix(matrix)

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

    @staticmethod
    def from_array(matrix: ndarray, initial_state: ndarray):
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
    def random(dimension: int, seed_: int = None):
        rng = default_rng(seed_)
        matrix = rng.random((dimension, dimension))
        initial_state = rng.random(dimension)
        object = Description.from_array(matrix, initial_state)
        object.my_seed = seed_
        return object

@dataclass
class Variation:
    _parent: Description = field(default = None)
    _changes: dict = field(default_factory = dict)

    def __setattr__(self, __name: str, __value):
        try:
            super().__setattr__(__name, __value)
        except AttributeError:
            self._changes[__name] = __value

    def __getattribute__(self, __name: str):
        result = None
        try:
            result = super().__getattribute__(__name)
        except AttributeError:
            if __name in self._changes:
                result = self._changes[__name]
            else:
                result = self._parent.__getattribute__(__name)
        finally:
            return result