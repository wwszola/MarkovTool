from MarkovTool import *
from itertools import pairwise

class XORInstance(Instance):
    def __init__(self, x1_inst: Instance, x2_inst: Instance):
        backend = (x1_inst, x2_inst)
        super().__init__(backend)
    
    def __next__(self):
        x1 = bool(self._backend[0].state)
        x2 = bool(self._backend[1].state)
        self._state = int(x1 ^ x2)
        return super().__next__()

x1_mat = [[0, 1], [1, 1]]
x1_desc = Markov(2, matrix = x1_mat, initial_state = 0, my_seed = 3)
x1 = Endless(x1_desc)

x2_mat = [[1, 1], [1, 0]]
x2_desc = Markov(2, matrix = x2_mat, initial_state = 0, my_seed = 1)
x2 = Endless(x2_desc)

y = XORInstance(x1, x2)

c = Collector(x1, x2, y)
m = Model((x1, [1]), (x2, [1]), (y, [1]))
m.forward(4096)
# print(*c.playback(x1))
# print(*c.playback(x2))
# print(*c.playback(y))

training = Markov(2, initial_state = 0)
training.fit(pairwise(c.playback(y)))
execution = Endless(training)
c.open(execution)
execution.skip(4096)
# print(*c.playback(execution))

print(c.count(y, (1, 2)))
print(c.count(execution, (1, 2)))
