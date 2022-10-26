from MarkovTool import *

from numpy import identity, ones, triu, array
from numpy.random import random
from itertools import islice

source = Markov(
    dimension = 3, 
    my_seed = 1,
    matrix = 1.0 - identity(3), # states not repeating
    initial_state = 0)

emmision1 = Description(
    shape = (3, 5),
    my_seed = 2,
    matrix = triu(ones((3, 5)), 1)) # emitted state > input state always

s = Endless(source) 
e1 = Dependent(emmision1, s)

c = Collector(s, e1)

# this should be called as a model
for _ in islice(zip(s, e1), 0, 30):    
    pass

print(c._entries)
c.close()