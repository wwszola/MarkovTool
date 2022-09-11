from story import MarkovChain
from copy import deepcopy

chain = MarkovChain.random(4, iter_reset = False)
print(chain.run(True))

chain_copy = deepcopy(chain)

print(chain_copy.run(True))

print(chain.run(True))
