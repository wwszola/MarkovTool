from markov_chain import MarkovChain

from statistics import stdev
from pathlib import Path
import numpy as np

models_folder = Path('models')

chain1 = MarkovChain.txt_load(
    models_folder/'model1.txt', 
    initial_state=0,
    my_seed=0,
    iter_reset=True)

# # print("-".join([str(state) for state in chain1]))

def setup1() -> None:
    k = 3
    for i in range(k):
        chain1.max_steps = (i+1)*3
        print(chain1.run(record=True))

setup1()

chain1.iter_reset = False
chain1.reset()

setup1()