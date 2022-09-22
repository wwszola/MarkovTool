from MarkovTool import Description, Endless


process: Description = Description.random(4, seed_=0)

print('Beginning of these should be different from each other')
for _ in range(3):
    print(Endless(process).take(10))

variant = process.variant(initial_state = 0)

print('All of these should be the same.')
for _ in range(3):
    print(Endless(variant).take(10))
