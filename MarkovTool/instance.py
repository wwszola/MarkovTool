from copy import deepcopy, copy
from typing import Callable, Iterable
from typing_extensions import Self
from numpy import ndarray, cumsum
from numpy.random import Generator, default_rng
from itertools import islice

from .description import Description
from .stat import Collector

class Endless(Iterable):
    """Base instance inherited from Iterable

    Static:
    _count: int = 0
        number of all instances created
    _gen_id() -> int
        returns new unique id

    Properties:
    state: int
        last state generated
        .setter assign to self._forced_state 
        on next next the effect executes
    
    Attributes:
    _description: Description
    _state: int = -1
        last state generated
    _forced_state: int = None
        is assigned a value in state.setter
    _step: int = 0
        number of times __next__ has been called
    _state_rng: numpy.random.Generator = numpy.random.default_rng()
        rng used for transitions
    _collectors: set[Collector] = set()
    _id: int

    Methods:
    __init__(self, description: Description)
        constructor creating new instance from description
    __eq__(self, other: Self) -> bool
        equal if _description, _step and _state are all equal 
    _bind_collector(self, collector: Collector) -> None
        adds collector to self._collectors
    _unbind_collector(self, collector: Collector) -> None
        remove collector from self._collectors
    _emit(self, step: int, state: int) -> None
        put a new entry in all from self._collectors
    _pick_initial_state(self) -> int
        uses self._description._initial as rule 
    _pick_next_state(self) -> int
        uses self._description._transition as rule, calling with self.state
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
        id = Endless._count
        Endless._count += 1
        return id
    
    def __init__(self, description: Description) -> None:
        """constructor creating new instance from description"""
        self._description = description
        self._state: int = -1
        self._forced_state: int = None
        self._step: int = 0
        self._state_rng: Generator = default_rng(self._description.my_seed)
    
        self._collectors: set[Collector] = set()

        self._id = Endless._gen_id()

    @property
    def state(self) -> int:
        """returns last state generated"""
        if self._state >= 0:
            return self._state
        else:
            raise ValueError('No state generated yet')

    @state.setter 
    def state(self, value: int) -> None:
        """assigns value to temporary self._forced_state which is
        assigned to self._state in the next call to __next__
        """
        if value >= 0 and value < self._description.dimension:
            self._forced_state = value
        else:
            raise ValueError(f'State should be int in range [0, {self._description.dimension})')

    def __eq__(self, other: Self) -> bool:
        """equal if _description, _step and _state are all equal"""
        return self._description == other._description \
           and self._step == other._step \
           and self._state == other._state

    def _bind_collector(self, collector: Collector) -> None:
        """adds collector to self._collectors"""
        self._collectors.add(collector)

    def _unbind_collector(self, collector: Collector) -> None:
        """remove collector from self._collectors"""
        self._collectors.remove(collector)            

    def _emit(self, step: int, state: int) -> None:
        """put a new entry in all from self._collectors"""
        if len(self._collectors) == 0:
            return
        else:
            closed = []
            for collector in self._collectors:
                if collector._is_open:
                    collector.put(self._description, self._id, step, state)
                else:
                    closed.append(collector)
            for collector in closed:
                self._unbind_collector(collector)
                    
    def _pick_initial_state(self) -> int:
        """uses self._description._initial as rule 
        
        uses numpy.random.default_rng(None)
        """
        rng = default_rng(None)
        pick: float = rng.random()
        return self._description._initial(pick)

    def _pick_next_state(self) -> int:
        """uses self._description._transition as rule, calling with self.state 

        uses self._state_rng
        """
        pick: float = self._state_rng.random()
        return self._description._transition(self._state, pick)

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        """pick a new state, emit and return it

        first state is picked from initial distribution
        next states are picked from self._state_rng
        if _forced_state is not None, use that instead
        """
        if self._step == 0:
            self._state = self._pick_initial_state()
        elif self._forced_state is not None:
            self._state = self._forced_state
            self._forced_state = None
            self._state_rng.random() # consume value for continuity
        else:
            self._state = self._pick_next_state()
        self._emit(self._step, self._state)
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
        
        new _state_rng will be a deepcopy of self._state_rng
        pass property name and desired value as keyword arguments
        use properties from Description to assign a variant description
        branched instances preserve collectors, thus emitting without binding
        """
        new = copy(self)
        new._id = Endless._gen_id()
        new._state_rng = deepcopy(self._state_rng)
        if 'state' in kwargs:
            new.state = kwargs['state']
            del kwargs['state']
        if kwargs:
            new._description = self._description.variant(**kwargs)
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
    def __init__(self, description: Description,
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
            raise StopIteration
        return super().__next__()

class Dependent(Endless):
    """inherits from Endless
    uses self._parent.state as an input
    
    Properties:
    parent: Endless
        an instance whose state is used as an input state

    Attributes:
    _parent: Endless

    Methods:
    __init__(self, description: Description, parent: Endless)
        constructor setting description and parent
    _pick_initial_state() -> int
        overwrites Endless._pick_initial_state, call self._pick_next_state
    _pick_next_state()
        overwrites Endless._pick_next_state, uses transition with self.parent.state
    """
    def __init__(self, description: Description, parent: Endless) -> None:
        super().__init__(description)
        self._parent: Endless = None
        self.parent = parent

    @property
    def parent(self) -> Endless:
        """an instance whose state is used as an input state"""
        return self._parent

    @parent.setter
    def parent(self, value: Endless):
        """parent setter

        checks that self input space matches value output space
        raises ValueError otherwise
        """
        if value is None:
            return
        
        if self._description.shape[0] == value._description.shape[1]:
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
        return self._description._transition(self._parent.state, pick)