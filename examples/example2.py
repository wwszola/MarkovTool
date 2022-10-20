from MarkovTool import Markov, Endless


process: Markov = Markov(6, my_seed = 5).fill_random()

print('Beginning of these should be different from each other')
for _ in range(3):
    print(Endless(process).take(10))

variant = process.variant(initial_state = 0)

print('All of these should be the same.')
for _ in range(3):
    print(Endless(variant).take(10))
