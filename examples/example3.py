from story import MarkovChain

MarkovChain.normalize = True
chain = MarkovChain.random(4, max_steps = 15, my_seed = 0, iter_reset = True)
print('iter_reset=True')
print('Original process:', chain.run(True), sep='\n')

with chain.simulation(initial_state = 3) as chain1:
    print('Process with set initial_state = 3:', chain1.run(True), sep='\n')
    with chain1.simulation(my_seed = 7) as chain2:
        print('Process with set initial state and new seed:', chain2.run(True), sep='\n')

new_matrix = chain.matrix
new_matrix[:, 1] = 0.0
new_initial_state = chain.initial_state
new_initial_state[1] = 0.0
with chain.simulation(matrix = new_matrix, initial_state = new_initial_state) as chain1:
    print('Process with cut out all transitions to state 1:', chain1.run(True), sep='\n')