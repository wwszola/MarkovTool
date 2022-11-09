import itertools
from sys import stdout
from time import sleep
from MarkovTool import *

from numpy import identity, ones, triu
from copy import copy
from os import system
from time import time
# loop of 0, 1, 2, 3
mat_clock4 = [
    [0, 1, 0, 0], 
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0]
]
clock = Markov(
    dimension = 4, 
    matrix = mat_clock4,
    initial_state = 0)
time_process = Endless(clock) 

# applying probabilisitc noise to the clock
small_noise = [
    [15,  1,  1,  1],
    [ 1, 15,  1,  1],
    [ 1,  1, 15,  1],
    [ 1,  1,  1, 15],
]
experiment = Markov(
    dimension = 4,
    my_seed = 1,
    matrix = small_noise)
result = Dependent(experiment, time_process) # input of this instance is a state from time_process
collector = Collector(time_process, result)

FPS = 6
back = ['[', ' ', ' ', ' ', ' ', ']']
for t, *rs in itertools.zip_longest(time_process, *[result]*3): # generate 3 results based on single input state
    time_render = copy(back)
    time_render[t+1] = 'X'

    for r in rs:
        sys_time = time()
        system('cls||clear')
        result_render = copy(back)
        result_render[r+1] = 'X'

        print(''.join(time_render))
        print(''.join(result_render))
        stdout.flush()
        sleep(1/FPS - (time() - sys_time))