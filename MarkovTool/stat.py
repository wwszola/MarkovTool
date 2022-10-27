from typing import Hashable
from numpy import ndarray, array, bincount

class Collector:
    """Gathers states emitted by instances

    Static:
    _count: int = 0
        number of all instances created
    _gen_id() -> int
        return new unique id

    Attributes:
    _entries: dict
        nested dictionary recording all states, see put for details
    _is_open: bool = True
        if set to False no entries are accepted 
    _id: int

    Methods:
    __init__(self, *instances)
        calls self.open(*instances)  
    __hash__(self) -> int
        returns self._id
    open(self, *instances)
        start accepting entries and bind self to instances
    close(self)
        stop accepting entries
    put(self, desc: Description, id: int, step, state) -> bool
        try to make a new entry
    
    Valid instances:
    Instances bound by passing to __init__ or open,
    must implement method _bind_collector(self, collector)
    Instances should emit their state by calling put
    """
    _count: int = 0
    @staticmethod
    def _gen_id() -> int:
        """return new unique id"""
        id = Collector._count
        Collector._count += 1
        return id

    def __init__(self, *instances):
        """Calls self.open(*instances)  
        
        *instances
            see Valid instances in Collector.__doc__ 
        """
        self._entries: dict[Hashable, dict[tuple[id, int], list[int]]] = {}
        self._partial: dict[tuple[id, int], list[int]] = {}
        self._is_open: bool = True
        self._id = Collector._gen_id()
        self.open(*instances)

    def __hash__(self) -> int:
        """Returns self._id"""
        return self._id

    def open(self, *instances):
        """Start accepting entries and bind to instances
        
        *instances
            see Valid instances in Collector.__doc__ 
        """
        self._is_open = True
        for instance in instances:
            if callable(getattr(instance, "_bind_collector", None)):
                instance._bind_collector(self)
    
    def close(self):
        """Stop accepting entries"""
        self._is_open = False

    def put(self, id: int, step: int, state: int, backend = None) -> bool:
        """Try to make a new entry

        Makes sure no duplicates are present.
        The entry is accepted if self._is_open == True,
        and one of the following is also True:
        - a chunk exists that was started by the same instance,
        and awaits new value at the correct step
        - no value in any chunk exists that could be 
        matched correctly to the entry
        Parameters:
        TODO
        Returns:
        True if the entry has been accepted, False otherwise 
        """
        if not self._is_open:
            return False        

        group: dict = None

        if backend is None:
            group = self._partial
        else:
            if backend not in self._entries:
                self._entries[backend] = dict()
            group = self._entries[backend]

        for (id_, step_), chunk in group.items():
            if id == id_:            
                if step_ + len(chunk) == step:
                    chunk.append(state)
                    return True
            elif step_ <= step < step_ + len(chunk) and chunk[step - step_] == state:
                return False

        group[(id, step)] = [state]

        return True