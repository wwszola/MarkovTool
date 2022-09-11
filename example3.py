from story import MarkovChain
from copy import deepcopy

chain = MarkovChain.random(4, iter_reset = False)
print(chain.run(True))

with chain.simulation(max_steps=15) as chain_copy:
    print(chain.run(True))
    print(chain_copy.run(True))

