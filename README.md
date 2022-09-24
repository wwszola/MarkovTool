## How to use in a virtual environment
1. Clone the project
```
git clone https://github.com/wwszola/interactive_story
cd interactive_story
```
2. Create, activate and prepare a virtual environment
```
python -m venv venv
./venv/Scripts/activate

pip install -r requirements.txt
python setup.py install
```
3. Run some examples
```
python examples/example2.py
```
4. Deactivate the virtual environment after use
```
deactivate
```

## Get started
### First run
```
from MarkovTool import Description, Endless
process = Description.random(3)

print(Endless(process).take(10))
```
```
[0, 0, 2, 2, 0, 1, 1, 1, 1, 2]
```
Static method `random`, given the dimension, generates values for Markov matrix and initial probability vector. Constructor `Endless` creates a new running instance. We generate first 10 states and print them to the terminal.

### Stay on the path
```
for _ in range(3): 
    print(Endless(process).take(10))
```
```
[2, 2, 2, 0, 1, 0, 1, 0, 2, 0]
[2, 0, 1, 0, 2, 0, 2, 0, 1, 0]
[0, 1, 2, 0, 2, 2, 0, 2, 0, 2]
```
The same description generates different processes. That happens because `my_seed` property of `Description` defaults to `None`, generating unique RNGs.
Set this property to make sure the process behaves in repeatable manner.
```
process.my_seed = 0
for _ in range(3): 
    print(Endless(process).take(10))
```
```
[2, 2, 0, 0, 0, 2, 2, 2, 2, 2]
[0, 2, 0, 0, 0, 2, 2, 2, 2, 2]
[1, 2, 0, 0, 0, 2, 2, 2, 2, 2]
```
To ensure the first state is always the same set `initial_state` to an `int` value.
```
process.initial_state = 1
for _ in range(3): 
    print(Endless(d).take(10))
```
```
[1, 2, 0, 0, 0, 2, 2, 2, 2, 2]
[1, 2, 0, 0, 0, 2, 2, 2, 2, 2]
[1, 2, 0, 0, 0, 2, 2, 2, 2, 2]
```
See also `Description.random`, `Endless.skip`

### Manipulation
```
process = Description.random(2)
process.initial_state = 0
p_mat = process.matrix
# removing transition from state 1 to state 0 entirely 
p_mat[1, 0] = 0.0
variation = process.variant(matrix = p_mat)
print(Endless(process).take(10))
print(Endless(variation).take(10))
```
```
[0, 0, 1, 1, 0, 1, 1, 0, 0, 1]
[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
```
Call method `variant` to create a copy of the description. Pass properties you wish to change as keyword arguments.

### Parallel worlds
You may run instances with different properties while keeping others to create complex behaviour. First, generate 5 steps from `Endless` instance.
```
process = Description.random(7)
process.my_seed = 17
instance = Endless(process)
print(instance.take(5))
print(instance.branch(_state=4).take(10))
print(instance.take(10))
```
```
[4, 6, 1, 5, 3]
```
Next, run second unique process, which has `_state` set to 4, by calling `branch`. Both of them eventually converge to the same output. Instances produces the same states at the same steps, but only given the same previous state.
```
print(instance.branch(state=4).take(10))
print(instance.take(10))
```
```
[4, 1, 4, 5, 5, 0, 2, 4, 0, 6]
[1, 3, 2, 4, 5, 0, 2, 4, 0, 6]
```
### `Collector` section
`Collector` allows to gather all instances running in a manner that takes into a consideration parallel branches, but no duplicates will be present. 
Present in stat module

### A game?
__HAS NOT BEEN REFACTORED YET__
- example `examples/game.py` presents simple game where you bet and claim rewards for guessing the state

## Notes
- parallel is itertools.zip_longest??
- just use itertools to get the result you want
## TODO
- Collector.close unbinding instances
- docstrings: 
- - include info that branched instances are automatically emiiting to the collectors
- tests - behaviours we need to preserve:
- - start with examples
- package distribution using setuptools
- stat module: count, ...
- load from file: we want json
- refactor game example
- import clean-up
- manual instance