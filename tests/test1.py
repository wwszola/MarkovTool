import pytest

from functools import reduce

from MarkovTool.stat import Collector

def n_collectors(n, *args):
    if n == 1:
        return Collector(*args)
    else:
        return [n_collectors(1) for _ in range(n)]

def n_dummy_instances(n):
    class Dummy:
        def __init__(self):
            self.is_bound = False

        def _bind_collector(self, c):
            self.is_bound = True

    if n == 1:
        return Dummy()
    else:
        return [n_dummy_instances(1) for _ in range(n)]

class TestCollector:
        def test_gen_id_and_count(self):
            assert Collector._count == 0
            _N = 10
            expected_ids = list(range(Collector._count, 
                                  Collector._count + _N))
            collectors = n_collectors(_N)
            ids = [c._id for c in collectors]
            
            assert expected_ids == ids 
            assert Collector._count == collectors[0]._id + _N

        def test_hashable(self):
            _N = 10
            collectors = n_collectors(_N)
            hashes = [hash(c) for c in collectors]
            assert len(set(hashes)) == _N
            assert reduce(
                lambda x, y: x and y,
                map(lambda a: a == a, collectors))
            assert reduce(
                lambda x, y: x and y,
                map(lambda a, b: a < b and not a >= b,
                    collectors, collectors))

        def test_open_close(self):
            _N = 10
            instances = n_dummy_instances(_N)
            
            collector = n_collectors(1, instances)
            assert reduce(
                lambda x, y: x and y,
                map(lambda a: a.is_bound, instances))
            
            collector.close(instances)
            assert reduce(
                lambda x, y: x and y,
                map(lambda a: not a.is_bound, instances))