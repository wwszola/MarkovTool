from MarkovTool import Markov, Finite, Collector

d = Markov(8, my_seed = 7).fill_random()
f = Finite(d, lambda self: self._step >= 15)
c = Collector(f)

f.skip(5)
g = f.branch(state = 1)
f.skip(5), g.skip(5)
g.state = 4
f.skip(), g.skip()
c.close()
print(list(c.playback(f, d)))
print(list(c.playback(g, d)))