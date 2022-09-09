from typing import List
from time import time_ns
import numpy as np
from random import randrange

from story import MarkovChain

def round(dim: int = 3):
    chain = MarkovChain.random(dimension=dim, initial_state=0, my_seed=randrange(128), iter_reset=False)
    print(f"my seed: {chain.my_seed}")
    chain.max_steps = 5
    history: List = chain.run(True)
    chain.max_steps = 30

    guess_matrix = chain.matrix.copy()
    for i in range(dim):
        a, b = randrange(dim), randrange(dim)
        print(a, b)
        guess_matrix[a][b] = np.nan

    for state in chain:
        with np.printoptions(precision=2, floatmode='fixed', nanstr='X'):
            print(f'This is the stochastic matrix of the machine')
            print(guess_matrix)
            print(f'These are last five steps of the machine')
            print(history)
        
        break


game_running = True
input_buffer: str = ''
while(game_running):
    input_buffer = input('Do you want to play a game? (y/n)') + 'n'
    choice: str = input_buffer.lower()[0]
    if choice[0] == 'y':
        round()
