from MarkovTool import Markov
from MarkovTool import Endless, Dependent
from MarkovTool import Model, Collector

process1 = Markov(3).fill_random()
process1.initial_state = 0

mat = process1.matrix
# for i in range(process1.dimension): mat[i, i] = 0
process2 = process1.variant(matrix = mat)

node1 = Endless(process1)
node2 = Dependent(process2, node1)
c = Collector(node1, node2)

m = Model((node1, [1]), (node2, [1]))
m.forward(32)

diff = map(
    lambda s1, s2: s2 if s1 != s2 else None,
    c.playback(node1), c.playback(node2)
)
print(list(c.playback(node1)))
print(list(diff))