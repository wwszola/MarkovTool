from story import MarkovChain


chain = MarkovChain.random(4, max_steps = 10, my_seed = 0, iter_reset = True)
with chain.simulation(initial_state = 3) as chain1:
    print('iter_reset=True', 'Original process:', chain.run(True), sep='\n')
    print('Process with set initial_state:', chain1.run(True), sep='\n')
    with chain1.simulation(my_seed = 7) as chain2:
        print('Process with set initial state and new seed:', chain2.run(True), sep='\n')