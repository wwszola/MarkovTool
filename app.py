from markov_chain import MarkovChain

from pathlib import Path

models_folder = Path('models')

chain1 = MarkovChain.txt_load(
    models_folder/'model1.txt', 
    my_seed=0,
    initial_state=0,
    iter_reset=False)

# print("-".join([str(state) for state in chain1]))

chain1.my_seed = 17
chain1.state = 0
for _ in chain1:
    pass
print(chain1.count)

# chain1.my_seed = 18
# chain1.my_seed = 17
chain1.state = 0
for _ in chain1:
    pass
print(chain1.count)