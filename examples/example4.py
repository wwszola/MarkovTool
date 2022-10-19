from MarkovTool import Markov, Finite, Collector

d = Markov.random(3, seed_ = 11)
f = Finite(d, lambda self: self._step >= 10)

c = Collector(f)
f.skip(3)
g = f.branch()
g.state = 1
f.skip(3), g.skip(6)
c.close()
print(c._entries) 