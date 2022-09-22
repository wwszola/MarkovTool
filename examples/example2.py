from MarkovTool import Description, Variation, Endless
from itertools import cycle     

process = Description.random(4, seed_=0)

print('Beginning of these should be different from each other')
for _ in range(3):
    print(Endless(process).take(10))

variation = Variation(process, {"initial_state": 1})

print('All of these should be the same.')
for _ in range(3):
    print(Endless(variation).take(10))
