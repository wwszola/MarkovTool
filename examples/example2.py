from story import MarkovChain

MarkovChain.reset_static_rng(0)

chain = MarkovChain.random(
    dimension=4, 
    iter_reset=True)

print('Beginning of these should be different from each other')
for _ in range(3):
    print(chain.run(True))

print('All of these should be the same. Every iteration reset_static_rng is called')
for _ in range(3):
    # Although iter_reset is True, call reset_static_rng to get identical first state
    MarkovChain.reset_static_rng()
    print(chain.run(True))

'''using static rng for picking initial state allows 
chains with single and multiple initial_state to give identical processes
'''
