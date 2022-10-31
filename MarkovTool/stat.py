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
        self._entries: dict[Hashable, dict[int, list[list[str | int]]]] = {}
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

    def _redirect(self, a: int, b: int, backend: Hashable) -> int:
        group = self._entries.get(backend, None)
        if not group or a not in group:
            return False
        tape = group.setdefault(b, ['_'])
        tape.append(['REF', a, self._length(a, backend)])
        return False
    
    def _length(self, id: int, backend: Hashable) -> int:
        group = self._entries.get(backend, None)
        if not group:
            return None, id
        
        counter, d = 0, 0
        for x in group[id]:
            match x[0]:
                case 'RAW':
                    d = len(x) - 1
                case 'REF':
                    d = x[2]
                case _:
                    d = 0
            counter += d
        return counter

    def _retrieve(self, id: int, step: id, backend: Hashable) -> tuple[int, int]:
        group = self._entries.get(backend, None)
        if not group:
            return None, id
        
        counter, d = 0, 0
        it = iter(group[id])
        x = next(it, None)
        while x:
            match x[0]:
                case 'RAW':
                    d = len(x) - 1
                    if counter + d > step:
                        return x[step - counter + 1], id
                case 'REF':
                    d = x[2]
                    if counter + d > step:
                        id = x[1]              
                        it = iter(group[id])
                case _:
                    d = 0
            counter += d
            x = next(it, None)
        return None, id

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

        if backend is None:
            backend = '__EMPTY__'

        group = self._entries.setdefault(backend, dict())
        tape = group.setdefault(id, [['_']])

        matches = list(filter(lambda entry: entry[0] is not None and entry[0] == state, 
                             [self._retrieve(id, step, backend) for id in group.keys()]))

        if matches:
            if tape[-1][0] == 'REF' and tape[-1][1] in [m[1] for m in matches]:
                tape[-1][2] += 1
            else:
                tape.append(['REF', matches[0][1], 1])
            return False
        else:
            if tape[-1][0] == 'RAW':
                tape[-1].append(state)
            else:
                tape.append(['RAW', state])
            return True