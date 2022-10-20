from numpy import array, ndarray, float32, allclose, newaxis, cumsum
from numpy.random import default_rng
from copy import copy
from typing_extensions import Self

class Description:
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

    Attributes:
    _id: int
    _shape: tuple[int]
    _my_seed: int = None

    Methods:
    __init__(self, shape: tuple[int], my_seed: int)
        constructor setting shape and my_seed 
    __hash__(self) -> int
        returns self._id
    variant(self, **kwargs) -> Self
        returns modified copy
    """
    
    _count: int = 0
    @staticmethod
    def _gen_id() -> int:
        """returns new unique id"""
        id = Description._count
        Description._count += 1
        return id
    
    def __init__(self, shape: tuple[int] = None, my_seed: int = None):
        """constructor setting shape and my_seed

        all parameters default to None
        """
        self._shape: tuple[int] = None
        self.shape = shape
        self._my_seed: int = None
        self.my_seed: int = my_seed
        self._id = Description._gen_id()

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

    def __hash__(self) -> int:
        """returns self._id"""
        return self._id
    
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

    
class Markov(Description):
    """Dataclass describing a Markov process
    
    Static:
        returns new unique id
    from_array(matrix: ndarray, initial_state: int | ndarray) -> Self:
        creates object setting all but my_seed property
    random(dimension: int, seed_: int = None) -> Self
        generates random process setting all properties 

    Properties:
    dimension: int
        size of the state space
    matrix: ndarray 
        transition matrix for a process
    initial_state: int | ndarray
        initial distribution of a process

    Attributes:
    _matrix: ndarray = None
    _matrix_cumsum: ndarray = None
        precalculated values for picking algorithm
    _initial_state: int | ndarray = None
    _initial_state_cum_sum: ndarray = None
        precalculated values for picking algorithm

    Methods:
    __init__(self, dimension: int, my_seed: int)
        constructor setting dimension and my_seed
    _verify_matrix(self, value: ndarray) -> ndarray
        returns verified copy of a matrix
    _verify_initial_state(self, value: int | ndarray) -> int | ndarray
        returns verified copy of an initial state
    
    """

    def __init__(self, dimension: int = None, my_seed: int = None):
        """constructor setting dimension and my_seed"""
        super().__init__((dimension, dimension), my_seed)
        self._matrix: ndarray = None
        self._matrix_cumsum: ndarray = None
        self._initial_state: int | ndarray = None
        self._initial_state_cumsum: ndarray = None

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

    @property
    def matrix(self) -> ndarray:
        """transition matrix for a process

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
        self._matrix = self._verify_matrix(value)
        self._matrix_cumsum = cumsum(self._matrix, axis=1)

    def _verify_matrix(self, value: list[list[float]] | ndarray) -> ndarray:
        """returns verified copy of the matrix
        
        value after normalization should satisfy Markov property
        raises ValueError 
        """
        try:
            value = array(value, dtype=float32)
            if len(value.shape) != 2 or value.shape[0] != value.shape[1]:
                raise ValueError('Matrix should be an array NxN in size')

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
        self._initial_state = self._verify_initial_state(value)
        self._initial_state_cumsum = cumsum(self._initial_state)

    def _verify_initial_state(self, value: int | list | ndarray) -> int | ndarray:
        """returns verified copy of the initial_state

        value: int 
            should be in range [0, self._dimension)
        value: list | ndarray
            after normalization sum over value should be close to 1.0

        raises ValueError, TypeError
        """
        try:
            if isinstance(value, (list, ndarray)):
                if len(value) != self.dimension:
                    raise ValueError('Initial state dimension should be equal to the original dimension')

                value /= value.sum()[newaxis]         

                if allclose(value.sum(), 1.0):
                    return array(value, dtype = float32)
                else:
                    raise ValueError('Initial state probabilities should normalize to sum 1.0')
            
            elif isinstance(value, int):
                if value < 0 or value >= self.dimension:
                    raise ValueError(f'Invalid initial state: int {value}')
                else:
                    return value
            else:
                raise TypeError('Initial state should be the type of either int, list or numpy.ndarray')
        except (ValueError, TypeError) as err:
            raise err
        
    @staticmethod
    def from_array(matrix: ndarray, initial_state: ndarray) -> Self:
        """creates object setting all but my_seed property
        
        Dimension is derived from shape of the matrix

        Returns:
            Markov if created succesfully
            None if failed to set valid properties
        """
        object: Markov = None
        try:
            object = Markov()
            object.shape = matrix.shape
            object.matrix = matrix
            object.initial_state = initial_state
        except (ValueError, TypeError) as err:
            print(f'Failed loading data from an array')
            print(err)
        return object   

    @staticmethod
    def random(dimension: int, seed_: int = None) -> Self:
        """generates random process setting all properties 
        
        Using seed_ as a sequence for rng
        Generates valid matrix and initial_state values

        Returns:
            Markov if created succesfully
            None if failed to set valid properties
        """
        rng = default_rng(seed_)
        matrix = rng.random((dimension, dimension))
        initial_state = rng.random(dimension)
        object = Markov.from_array(matrix, initial_state)
        if object is not None:
            object.my_seed = seed_
        return object