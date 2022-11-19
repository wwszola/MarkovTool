from math import floor
from typing import Generator
from collections import OrderedDict
from itertools import cycle

from .instance import Instance

# usage in examples/example8.py
class Model():
    def __init__(self, *nodes: tuple[Instance, list[int]]):
        self._firing_order: OrderedDict[Instance, Generator] = OrderedDict()
        for instance, pattern in nodes:
            self._firing_order[instance] = cycle(pattern)

        self._tick = 0

    def forward(self, ticks: int):
        end = self._tick + ticks
        while self._tick < end:
            for instance, fire in self._firing_order.items():
                if next(fire):
                    next(instance, None)
            self._tick += 1