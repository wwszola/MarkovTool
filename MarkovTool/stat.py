from dataclasses import dataclass, field
from enum import Enum
from typing import Hashable, Generator
from itertools import islice

class ChunkType(Enum):
    """Enum class, use this as a pattern for matching"""
    RAW = 0
    REF = 1

@dataclass
class ChunkRaw:
    """dataclass storing raw values in a chunk
    
    Attributes:
    start: int
        step of the first state in a chunk
    data: list[int]
    type_: ChunkType
    """
    start: int
    data: list[int]
    type_: ChunkType = field(default = ChunkType.RAW, init = False)

@dataclass
class ChunkRef:
    """dataclass referencing a ChunkRaw
    
    Attributes:
    start: int
        step of the first state in a chunk
    point_to: ChunkRaw
        chunk of data referenced
    length: int
        how much is referenced
    type_: ChunkType
    """
    start: int
    point_to: ChunkRaw
    length: int
    type_: ChunkType = field(default = ChunkType.REF, init = False)

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
    _match(self, group: dict, step: int, state: int) -> tuple[ChunkRaw, Hashable]
        searches for the chunk containing specific value on correct step
    put(self, instance: Hashable, step: int, state: int, backend: Hashable = None) -> bool
        try to make a new entry
    redirect(self, src: Hashable, dst: Hashable) -> bool
        copy entries of src as emitted by dst
    length(self, id: int) -> int
        returns step of the last entry put by an instance
    retrieve(self, instance: Hashable, step: int) -> int
        returns state put by an instance on given step
    playback(self, instance: Hashable) -> Generator:
        returns Generator yielding states put by an instance
    count(self, instance, windows = (1), step_range = None) -> dict[tuple, int]
        count number of occurences of patterns

    Valid instances:
    Instances bound by passing to __init__ or open, 
    may implement method _bind_collector(self, collector),
    must implement method _entry(self) returning dict with keys 
    {'instance', 'step', 'state', 'backend'} where value for 'backend' may be None
    Instances may emit their state by calling put with _entry() result as keyword arguments
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
        self._entries: dict[Hashable, dict[Hashable, list[ChunkRef | ChunkRaw]]] = {}
        self._entries['__EMPTY__'] = {}
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

    def _match(self, group: dict[Hashable, list[ChunkRef | ChunkRaw]], step: int, state: int) -> tuple[ChunkRaw, Hashable]:
        """searches for the chunk containing specific value on correct step
        
        Parameters:
        group: dict
            dict for group of instances
        step: int
        state: int

        Returns:
        tuple[ChunkRaw, Hashable] 
            chunk and the instance it belongs to
        tuple[None, None]
            no match found
        """
        for instance, tape in group.items():
            for chunk in tape:
                if chunk.type_ == ChunkType.RAW and chunk.start + len(chunk.data) > step:
                    return (chunk, instance)

        return (None, None)

    def put(self, instance: Hashable, step: int, state: int, backend: Hashable = None) -> bool:
        """Try to make a new entry
        Makes sure no duplicates are present for instances from given backend
        
        Parameters:
        instance: Hashable
            see valid instances in Collector.__doc__
        step: int
        state: int
        backend: Hashable = None
            tag representing specific process

        Returns:
        True:
            the entry has been accepted
        False:
            collector is closed or duplicate
        """
        if not self._is_open:
            return False        

        if backend is None:
            backend = '__EMPTY__'
        group = self._entries.setdefault(backend, dict())

        raw_match, instance_ = self._match(group, step, state)
        if instance_ == instance:
            return False

        tape = group.setdefault(instance, [])

        if not tape:
            if raw_match:
                tape.append(ChunkRef(step, raw_match, 1))
            else:
                tape.append(ChunkRaw(step, [state]))
            return True

        last = tape[-1]
        match last.type_, bool(raw_match):
            case [ChunkType.REF, True]:
                if last.point_to is raw_match:
                    last.length += 1
            case [ChunkType.RAW, True]:
                tape.append(ChunkRef(step, raw_match, 1))
            case [ChunkType.REF, False]:
                tape.append(ChunkRaw(step, [state]))
            case [ChunkType.RAW, False]:
                last.data.append(state)
            
        return True

    def redirect(self, src: Hashable, dst: Hashable) -> bool:
        """copy entries of src as emitted by dst

        see valid instances in Collector.__doc__    
        Parameters:
        src: Hashable
            instance to copy entries from
        dst: Hashable
            instance to put entries into
        
        Returns:
        True:
            success
        False: 
            'backend' value for instances' entries are not equal
            src instance hasn't put an entry
        """
        src_backend = src._entry()['backend']
        dst_backend = src._entry()['backend']
        group = self._entries.get(src_backend, None)
        if not group or src not in group or src_backend != dst_backend:
            return False

        src_tape = group[src]
        dst_tape = group.setdefault(dst, [])
        for chunk in src_tape:
            match chunk.type_:
                case ChunkType.REF:
                    dst_tape.append(chunk)
                case ChunkType.RAW:
                    dst_tape.append(ChunkRef(chunk.start, chunk, len(chunk.data)))
        return True
    
    def length(self, instance: Hashable) -> int:
        """returns step of the last entry put by an instance
        returns None if instance hasn't put an entry

        Parameters:
        instance: Hashable
            see valid instances in Collector.__doc__
        """
        backend = instance._entry()['backend']
        if backend is None:
            backend = '__EMPTY__'

        group = self._entries.get(backend, None)
        if not group:
            return None
        
        tape = group.get(instance, None)
        if not group:
            return None

        last = tape[-1]
        match last.type_:
            case ChunkType.RAW:
                return last.start + len(last.data)
            case ChunkType.RAW:
                return last.start + last.length

    def retrieve(self, instance: Hashable, step: int) -> int:
        """returns state put by an instance on given step
        returns None if instance hasn't put an entry at that step
        
        Parameters:
        instance: Hashable
            see valid instances in Collector.__doc__
        step: int
        """
        backend = instance._entry()['backend']
        if backend is None:
            backend = '__EMPTY__'
        
        group = self._entries.get(backend, None)
        if not group:
            return None
        
        tape = group.get(instance, None)
        if not group:
            return None
        
        for chunk in tape:
            match chunk.type_:
                case ChunkType.RAW:
                    if chunk.start + len(chunk.data) > step:
                        return chunk.data[step - chunk.start]
                case ChunkType.REF:
                    if chunk.start + chunk.length > step:
                        return chunk.point_to.data[step - chunk.start]

        return None

    def playback(self, instance: Hashable) -> Generator:
        """returns Generator yielding states put by an instance
        returns None if instance hasn't put an entry 

        Parameters:
        instance: Hashable
            see valid instances in Collector.__doc__
        
        """
        backend = instance._entry()['backend']
        if backend is None:
            backend = '__EMPTY__'
        
        group = self._entries.get(backend, None)
        if not group:
            return None

        tape = group.get(instance, None)
        if not tape:
            return None

        for chunk in tape:
            match chunk.type_:
                case ChunkType.RAW:
                    for state in chunk.data:
                        yield state
                case ChunkType.REF:
                    raw = chunk.point_to
                    start = chunk.start - raw.start
                    for state in raw.data[start : start + chunk.length]:
                        yield state
        return None

    def count(self, instance: Hashable, windows: tuple[int] = (1, ), step_range: tuple[int, int] = None) -> dict[tuple, int]:
        """count number of occurences of patterns

        Parameters:
        instance: Hashable
            see valid instances in Collector.__doc__
        windows: tuple[int] = (1,)
            lengths of patterns which are taken into account
        step_range: tuple[int, int] = None
            specifies range that counting is done over
            if step_range is None whole playback is iterated

        Returns:
        dict[tuple, int]
            histogram of all patterns with specified lengths
        None
            if instance hasn't put an entry in specified range
        """
        history = self.playback(instance)
        if not history:
            return None

        if step_range:
            history = islice(history, step_range[0], step_range[1])
        history = list(history)
        if len(history) == 0:
            return None
        
        result: dict[tuple, int] = dict()
        for width in windows:
            for i in range(len(history) - width + 1):
                pattern = tuple(history[i : i + width])
                c = result.setdefault(pattern, 0)
                result[pattern] = c + 1
        return result