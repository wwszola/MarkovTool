from MarkovTool import *
from numpy import eye


# signal_mat = []
signal_desc = Markov(4).fill_random(2)
signal = Endless(signal_desc)

activation_desc = Stochastic((4, 2)).fill_random(21)
activation = Dependent(activation_desc, signal)

c = Collector(signal, activation)

m = Model(
    (signal, [1, 0, 0, 0]), 
    (activation, [1, 1, 1, 1])
)
m.forward(32)
print(*c.playback(signal), sep='\t')
print(*c.playback(activation))