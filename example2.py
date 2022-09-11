from story import MarkovChain

MarkovChain.reset_static_rng(0)

chain = MarkovChain.random(
    dimension=3, 
    iter_reset=True)

# Some of these should be different from each other
for _ in range(3):
    print(chain.run(True))

print('\n')

# All of these should be the same
for _ in range(3):
    # Although iter_reset is True, call reset_static_rng to get identical first state
    MarkovChain.reset_static_rng()
    print(chain.run(True))

'''using static rng for picking initial state allows 
chains with single and multiple initial_state to give identical processes
'''
