from markov_chain import MarkovChain

from statistics import stdev
from pathlib import Path
import numpy as np

models_folder = Path('models')

chain1 = MarkovChain.txt_load(
    models_folder/'model1.txt', 
    initial_state=0,
    max_steps=100,
    my_seed=0,
    iter_reset=True)

# print("-".join([str(state) for state in chain1]))

k = 5
for i in range(k):
    chain1.my_seed = i
    chain1.run()
    print(chain1.count, stdev(chain1.count))