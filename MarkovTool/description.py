from numpy import array, ndarray, float32, allclose, newaxis, cumsum
from numpy.random import default_rng, Generator
from copy import copy
from typing_extensions import Self

class Description():
    """Class describing a process

    Static:
    _count: int = 0
        number of all instances created
    _gen_id() -> int

    Properties:
    shape: tuple[int]
        input x output size of state space  
    my_seed: int
        value which seeds the instance of a process
    matrix: ndarray 
        probability matrix for a process
    initial_state: int | ndarray
        initial distribution of a process

    Attributes:
    _id: int
    _shape: tuple[int]
    _my_seed: int = None
    _matrix: ndarray = None
    _matrix_cumsum: ndarray = None
        precalculated values for picking algorithm
    _initial_state: int | ndarray = None
    _initial_state_cum_sum: ndarray = None
        precalculated values for picking algorithm

    Methods:
    __init__(self, shape, my_seed, matrix, initial_state)
        constructor setting shape, my_seed, matrix and initial_state
    _verify_matrix(self, value: list[list[float]] | ndarray) -> ndarray
        returns verified copy of the matrix
    _verify_initial_state(self, value: int | ndarray) -> int | ndarray
        returns verified copy of an initial state
    __hash__(self) -> int
        returns self._id
    _initial(self, pick: float) -> int
        defines rule for initial state
    _transition(self, state: int, pick: float) -> int
        defines rule for the next state
    variant(self, **kwargs) -> Self
        returns modified copy
    fill_random(self, seed_: int) -> Generator:
        generates random matrix 
    """
    
    _count: int = 0
    @staticmethod
    def _gen_id() -> int:
        """returns new unique id"""
        id = Description._count
        Description._count += 1
        return id
    
    def __init__(self, shape: tuple[int] = None, my_seed: int = None, 
                 matrix: ndarray = None, initial_state: int | ndarray = None):
        """constructor setting shape, my_seed, matrix, initial_state

        all parameters default to None
        """
        self._id = Description._gen_id()
        self._shape: tuple[int] = None
        self.shape = shape
        self._my_seed: int = None
        self.my_seed: int = my_seed
        self._matrix: ndarray = None
        self._matrix_cumsum: ndarray = None
        self.matrix = matrix
        self._initial_state: int | ndarray = None
        self._initial_state_cumsum: ndarray = None
        self.initial_state = initial_state

    def __str__(self) -> str:
        name = type(self).__name__
        shape = str(self.shape)
        return f'{name: >8}:{self._id: <8}\n{self.my_seed: >4} {shape: >8}' 

    @property
    def shape(self) -> tuple[int]:
        """input x output size of state space
        
        raises ValueError"""
        if self._shape is not None:
            return self._shape
        else:
            raise ValueError('Shape isn\'t defined yet')

    @shape.setter
    def shape(self, value: tuple[int]):
        """shape property setter

        raises ValueError
        """        
        if self._shape is not None:
            raise ValueError('Shape may be set only once')

        if value is None or value[0] is None or value[1] is None:
            return

        if value[0] > 0 and value[1] > 0:
            self._shape = value
        else:
            raise ValueError('Shape must be tuple of two positive integers')


    @property
    def my_seed(self) -> int:
        """value which seeds the instance of a process
        
        if None, every instance will have different behaviour
        """
        return self._my_seed

    @my_seed.setter
    def my_seed(self, value: int) -> None:
        """my_seed property setter"""
        self._my_seed = value

    @property
    def matrix(self) -> ndarray:
        """probability matrix for a process

        raises ValueError
        """
        if self._matrix is not None:
            return copy(self._matrix)
        else:
            raise ValueError('Matrix isn\'t defined yet')

    @matrix.setter
    def matrix(self, value: ndarray) -> None:
        """matrix property setter
        also calculates self._matrix_cumsum

        raises ValueError
        """
        if value is None:
            return
        
        self._matrix = self._verify_matrix(value)
        self._matrix_cumsum = cumsum(self._matrix, axis=1)

    def _verify_matrix(self, value: list[list[float]] | ndarray) -> ndarray:
        """returns verified copy of the matrix
        
        values are normalized so the sum of every row is equal to 1.0
        raises ValueError 
        """
        try:
            value = array(value, dtype=float32)

            if value.shape != self.shape:
                raise ValueError('Matrix dimension should be equal to the original dimension')
            
            value /= value.sum(1)[:, newaxis]

            if not allclose(value.sum(1), 1.0):
                raise ValueError('Matrix should be a right-stochastic matrix')
            
            return value
        
        except ValueError as err:
            raise err

    @property
    def initial_state(self) -> int | ndarray:
        """initial distribution of a process

        raises ValueError
        """
        if self._initial_state is not None:
            return copy(self._initial_state)
        else:
            raise ValueError('Initial state isn\' defined yet')
    
    @initial_state.setter
    def initial_state(self, value: int | list | ndarray) -> None:
        """initial_state property setter
        also calculates self._initial_state_cumsum

        raises ValueError
        """
        if value is None:
            return

        self._initial_state = self._verify_initial_state(value)
        self._initial_state_cumsum = cumsum(self._initial_state)

    def _verify_initial_state(self, value: int | list | ndarray) -> int | ndarray:
        """returns verified copy of the initial_state

        value: int 
            should be in range [0, self.shape[0])
        value: list | ndarray
            after normalization sum over rows should be close to 1.0

        raises ValueError, TypeError
        """
        try:
            if isinstance(value, (list, ndarray)):
                if len(value) != self.shape[0]:
                    raise ValueError('Initial state size should be input size')

                value /= value.sum()[newaxis]         

                if allclose(value.sum(), 1.0):
                    return array(value, dtype = float32)
                else:
                    raise ValueError('Initial state probabilities should normalize to sum 1.0')
            
            elif isinstance(value, int):
                if value < 0 or value >= self.shape[0]:
                    raise ValueError(f'Invalid initial state: int {value}')
                else:
                    return value
            else:
                raise TypeError('Initial state should be the type of either int, list or numpy.ndarray')
        except (ValueError, TypeError) as err:
            print(value)
            raise err

    def __hash__(self) -> int:
        """returns self._id"""
        return self._id
    
    def _initial(self, pick: float) -> int:
        """defines rule for initial state

        Parameters:
        pick: float
            should be random value uniform in range [0, 1)
        """

        if isinstance(self.initial_state, ndarray):
            accumulated = self._initial_state_cumsum
            for i, value in enumerate(accumulated):
                if pick < value:
                    return i
            else: 
                raise ValueError('pick is higher than the last element of accumulated')    
        else:
            return self.initial_state

    def _transition(self, state: int, pick: float) -> int:
        """defines rule for the next state

        Parameters:
        state: int 
            next state may depend on a given state
        pick: float
            should be random value uniform in range [0, 1)
        """
        accumulated = self._matrix_cumsum[state]
        for i, value in enumerate(accumulated):
            if pick < value:
                return i
        else: 
            print(pick, accumulated)
            raise ValueError('pick is higher than the last element of accumulated')

    def variant(self, **kwargs) -> Self:
        """returns modified copy

        pass property name and desired value as keyword arguments
        """
        result = copy(self)
        self._id = Description._gen_id()
        for name, value in kwargs.items():
            if hasattr(result, name):
                setattr(result, name, value)
        return result

    def fill_random(self, seed_: int = None, rng: Generator = None) -> Self:
        """generates random matrix 
        
        Using seed_ as a sequence for rng
        Generates valid matrix values
        Returns self
        """
        if rng is None:
            rng = default_rng(seed_)
        self.matrix = rng.random(self.shape)
        self.initial_state = rng.random(self.shape[0])
        return self
    
class Markov(Description):
    """Dataclass describing a Markov process
    
    Properties:
    dimension: int
        size of the state space

    Methods:
    __init__(self, dimension, my_seed, matrix, initial_state)
        constructor setting dimension, matrix, my_seed, initial_state

    This exists just for making sure input and output sizes are the same
    """

    def __init__(self, dimension: int = None, my_seed: int = None, 
                 matrix: ndarray = None, initial_state: int | ndarray = None):
        """constructor setting dimension and my_seed"""
        super().__init__((dimension, dimension), my_seed, matrix, initial_state)

    @property
    def dimension(self) -> int:
        """size of the state space

        calls Description.shape getter
        raises ValueError
        """
        try:
            return self.shape[0]
        except ValueError as err:
            raise err        

    @dimension.setter
    def dimension(self, value: int) -> None:
        """dimension property setter
        
        calls Description.shape setter
        raises ValueError
        """
        try:
            self.shape = (value, value)
        except ValueError as err:
            raise err
