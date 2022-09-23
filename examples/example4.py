from MarkovTool import Description, Finite, Collector

d = Description.random(3)
f = Finite(d, lambda self: self._step >= 10)
f.state = 1
g = f.branch()
g.state = 0

c = Collector()
c.open(f, g)
print(f.take(10), f._step)
print(g.take(10), g._step)
c.close()
print(c._entries)   