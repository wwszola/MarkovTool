from MarkovTool import *

from numpy import identity, ones, triu

source = Markov(
    dimension = 3, 
    my_seed = 1,
    matrix = 1.0 - identity(3), # states not repeating
    initial_state = 0)

emmision1 = Description(
    shape = (3, 5),
    my_seed = 2,
    matrix = triu(ones((3, 5)), 1)) # emitted state > input state always

s = Finite(source, lambda self: self._step > 10) 
e1 = Dependent(emmision1, s)

# this should be called as a model
for v1, v2 in zip(s, e1):    
    print(v1, v2)
