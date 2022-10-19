from MarkovTool import Markov, Endless


p1 = Markov.random(4, seed_ = 1)
p1.initial_state = 0

f = Endless(p1)
first_5_steps = f.take(5) 

print('Original process:')
print(first_5_steps, f.branch().take(10))

print('The process branched at 5th step with forced state=3:')
print(first_5_steps, f.branch(state = 3).take(10))

print('The process branched at 5th step with no possible transitions to state 1:')
new_matrix = p1.matrix
new_matrix[:, 1] = 0.0
print(first_5_steps, f.branch(matrix = new_matrix).take(10))