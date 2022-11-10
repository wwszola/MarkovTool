from copy import deepcopy, copy
from typing import Callable, Iterator, Iterable, Hashable
from typing_extensions import Self
from numpy.random import Generator, default_rng
from itertools import islice

from .description import Stochastic
from .stat import Collector


class Instance(Iterator):
    """Representing a process specific to a backend
    A backend needs to be hashable

    Static:
    _count: int = 0
        number of all instances created
    _gen_id() -> int
        returns new unique id

    Properties:
    has_stopped: bool
        True if StopIteration has been raised
    state: int
        last state generated
        .setter assign to self._forced_state 
        on next next the effect executes
    
    Attributes:
    _backend: Hashable = None
        tag representing specific process
        while extending Instance use that for generating states
    _has_stopped: bool = False
    _state: int = -1
        last state generated
    _forced_state: int | Generator = None
        is assigned a value in state.setter
    _step: int = 0
        number of times __next__ has been called
    _collectors: set[Collector] = set()
    _id: int

    Methods:
    __init__(self, backend: Hashable = None)
        constructor creating new instance
    __hash__(self) -> int
        returns self._id
    __eq__(self) -> bool
        uses hash equality
    __str__(self) -> str
    __repr__(self) -> str
    _verify_state(self, value: int | Iterable[int]) -> int | Generator
        returns veirified state value
    _bind_collector(self, collector: Collector) -> None
        adds collector to self._collectors
    _unbind_collector(self, collector: Collector) -> None
        remove collector from self._collectors
    _entry(self) -> dict
        creates entry for emitting
    _emit(self, step: int, state: int) -> None
        put a new entry in all from self._collectors
    __iter__(self) -> Self
        return self
    __next__(self) -> int
        pick a new state, emit and return it 
    take(self, n: int = None) -> list:
        returns list of next n generated states
    skip(self, n: int = None) -> None:
        advance the iterator n steps
    branch(self, **kwargs) -> Self:
        returns a modified copy
    """

    _count: int = 0
    @staticmethod
    def _gen_id() -> int:
        """returns new unique id"""
        id = Instance._count
        Instance._count += 1
        return id
    
    def __init__(self, backend: Hashable = None) -> None:
        """constructor creating new instance"""
        self._backend: Hashable = backend
        self._has_stopped: bool = False
        self._state: int = None
        self._forced_state: int = None
        self._step: int = 0
    
        self._collectors: set[Collector] = set()

        self._id = Instance._gen_id()

    def __hash__(self) -> int:
        """Returns self._id"""
        return self._id    

    def __eq__(self, other: Self) -> bool:
        """uses hash equality"""
        return type(self) == type(other) and hash(self) == hash(other)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        name = type(self).__name__
        return f'{name}(_id: {self._id}, _backend: {self._backend}, step: {self._step}, state: {self._state})'

    @property
    def has_stopped(self) -> bool:
        """True if StopIteration has been raised
        """
        return self._has_stopped

    @property
    def state(self) -> int:
        """returns last state generated"""
        if self._state is not None:
            return self._state
        else:
            raise ValueError('No state generated yet')

    @state.setter 
    def state(self, value: int | Iterable[int]) -> None:
        """assigns value to temporary self._forced_state which is
        assigned to self._state in the next call to __next__
        """
        value = self._verify_state(value)
        self._forced_state = value

    def _verify_state(self, value: int | Iterable[int]) -> int | Generator:
        """returns argument value unchanged"""
        if isinstance(value, int):
            return value
        elif isinstance(value, Iterable):
            return (_ for _ in value)
        else:
            return None

    def _bind_collector(self, collector: Collector) -> None:
        """adds collector to self._collectors"""
        self._collectors.add(collector)

    def _unbind_collector(self, collector: Collector) -> None:
        """remove collector from self._collectors"""
        self._collectors.remove(collector)            

    def _entry(self) -> dict:
        """creates entry for emitting"""
        return {'backend': self._backend, 'instance': self, 'step': self._step, 'state': self.state}

    def _emit(self) -> None:
        """put a new entry in all from self._collectors"""
        if len(self._collectors) == 0:
            return
        else:
            closed = []
            for collector in self._collectors:
                if collector._is_open:
                    collector.put(**self._entry())
                else:
                    closed.append(collector)
            for collector in closed:
                self._unbind_collector(collector)
                    
    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        """pick a new state, emit and return it

        extend this method, like that:
        self._state = ...
        return super().__next__()

        if _forced_state is int, use that value
        if _forced_state is Generator, use next value
        """
        if isinstance(self._forced_state, int):
            self._state = self._forced_state
            self._forced_state = None
        elif isinstance(self._forced_state, Generator):
            value = next(self._forced_state, None)
            if not value:
                self._forced_state = None
            else:
                self._state = value

        self._emit()
        self._step += 1
        return self._state    
        
    def take(self, n: int = None) -> list:
        """returns list of next n generated states
        
        if n is None generate until self is not exhausted
        """
        if n is None:
            return list(self)
        else:
            return list(islice(self, n))

    def skip(self, n: int = None) -> None:
        """advance the iterator n steps

        if n is None consume whole iterator
        """
        if n is None:
            for _ in self: pass
        else:
            next(islice(self, n, n), None)

    def branch(self, **kwargs) -> Self:
        """returns a modified copy
        
        branched instances preserve collectors, thus emitting without binding
        """
        new = copy(self)
        new._id = Instance._gen_id()
        if 'state' in kwargs:
            new.state = kwargs['state']
            del kwargs['state']

        closed = []
        for c in self._collectors:
            if c._is_open:
                c._redirect(self, new, self._entry()['backend'])
            else:
                closed.append(c)
        for c in closed:
            self._unbind_collector(c)
                    
        return new 

class Endless(Instance):
    """inherits from Instance, represents a stochastic process

    Attributes:
    _backend: Stochastic
        extends Instance._backend
        describe stochastic behaviour of the process
    _state_rng: numpy.random.Generator = numpy.random.default_rng()
        rng used for transitions
    
    Methods:
    __init__(self, description: Stochastic)
        constructor creating new instance from description, extends Instance.__init__
    _verify_state(self, value: int) -> int
        returns verified value
    _pick_initial_state(self) -> int
        uses self._backend._initial as rule 
    _pick_next_state(self) -> int
        uses self._backend._transition as rule, calling with self.state
    _entry(self) -> dict
        extends Instance._entry, assigns description
    __next__(self) -> int
        extends Instance.__next__
    branch(self, **kwargs) -> Self
        extends Instance.branch, assigns _state_rng and correct description
    """
    def __init__(self, description: Stochastic) -> None:
        super().__init__(description)
        self._state_rng: Generator = default_rng(self._backend.my_seed)

    def _verify_state(self, value: int) -> int:
        """return verified state"""
        if value < 0 or value >= self._backend.shape[0]:
            raise ValueError(f'State should be int in range [0, {self._backend.dimension})')
        return value

    def _pick_initial_state(self) -> int:
        """uses self._backend._initial as rule 
        
        uses numpy.random.default_rng(None)
        """
        rng = default_rng(None)
        pick: float = rng.random()
        return self._backend._initial(pick)

    def _pick_next_state(self) -> int:
        """uses self._backend._transition as rule, calling with self.state 

        uses self._state_rng
        """
        pick: float = self._state_rng.random()
        return self._backend._transition(self._state, pick)

    def __next__(self) -> int:
        """extends Instance.__next__

        first state is picked from initial distribution
        next states are picked from self._state_rng
        """
        if self._step == 0:
            self._state = self._pick_initial_state()
        else:
            self._state = self._pick_next_state()

        return super().__next__()
        
    def branch(self, **kwargs) -> Self:
        """extends Instance.branch, assigns _state_rng and correct description
        
        new _state_rng will be a deepcopy of self._state_rng
        pass property name and desired value as keyword arguments
        use properties from Description to assign a variant description
        """
        new = super().branch(**kwargs)
        new._state_rng = deepcopy(self._state_rng)
        kwargs = dict(((k, v) for k, v in kwargs.items() if hasattr(self._backend, k)))
        if kwargs:
            new._backend = self._backend.variant(**kwargs)
        return new

class Finite(Endless):
    """Class inheriting from Endless implementing a stop condition

    Attributes:
    _stop_predicate: Callable[[Finite], bool]
        if called returns True, raise StopIteration
    
    Methods:
    __init__(self, description, stop_predicate: Callable = ...)
        extends Endless.__init__ and sets stop_predicate
    __iter__(self) -> Self:
        extends Endless.__iter__
    __next__(self) -> int:
        check self._stop_predicate and call Endless.__next__
    """
    def __init__(self, description: Stochastic,
                 stop_predicate: Callable[[Self], bool] = lambda self: False):
        """extends Endless.__init__ and sets stop_predicate
        
        Parameters:
        description: Description
            passed to Endless.__init__
        stop_predicate: Callable[[Self], bool]
            defaults to lambda self: False
        """
        super().__init__(description)
        self._stop_predicate = stop_predicate
            
    def __iter__(self) -> Self:
        """extends Endless.__iter__"""
        return super().__iter__()

    def __next__(self) -> int:
        """check self._stop_predicate and call Endless.__next__"""
        if self._step > 0 and self._stop_predicate(self):
            self._has_stopped = True
            raise StopIteration
        return super().__next__()

class Dependent(Endless):
    """inherits from Endless, process with Instance object as input
    uses self._input.state for transitions
    
    Properties:
    input: Instance
        an instance whose state is used as an input state

    Attributes:
    _input: Instance

    Methods:
    __init__(self, description: Stochastic, parent: Instance)
        constructor setting description and parent
    _pick_initial_state() -> int
        overwrites Endless._pick_initial_state, call self._pick_next_state
    _pick_next_state()
        overwrites Endless._pick_next_state, uses transition with self.parent.state
    """
    def __init__(self, description: Stochastic, input: Instance) -> None:
        super().__init__(description)
        self._input: Instance = None
        self.input = input

    @property
    def input(self) -> Instance:
        """an instance whose state is used as an input state"""
        return self._input

    @input.setter
    def input(self, value: Instance):
        """input setter

        checks that self input space matches value output space
        raises ValueError otherwise
        """
        if value is None:
            return
        
        if self._backend.shape[0] == value._backend.shape[1]:
            self._parent = value
        else:
            raise ValueError("Parent output size should match input size")

    def _pick_initial_state(self) -> int:
        """overwrites Endless._pick_initial_state, call self._pick_next_state
        """
        return self._pick_next_state()

    def _pick_next_state(self) -> int:
        """overwrites Endless._pick_next_state, uses transition with self.parent.state
        """
        pick: float = self._state_rng.random()
        return self._backend._transition(self._parent.state, pick)
    
    def __next__(self) -> int:
        if self._parent.has_stopped:
            self._has_stopped = True
            raise StopIteration()
        
        return super().__next__() 
