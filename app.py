from markov_chain import MarkovChain

from pathlib import Path

models_folder = Path('models')

chain1 = MarkovChain.txt_load(models_folder/'model1.txt', my_seed = 0)

print("-".join([str(state) for state in chain1]))

chain1.state, chain1.my_seed = 0, 0

print("-".join([str(state) for state in chain1]))
