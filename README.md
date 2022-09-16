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

python install -r requirements.txt
python setup.py install
```
3. Run some examples
```
python examples/example1.py
```
4. Deactivate the virtual environment after use
```
deactivate
```
## Project currently includes:
### `MarkovChain` implementation:
- Static methods `txt_load`, `from_array`, `random` for creating `MarkovChain` objects
- `MarkovChain` inherits from `Iterator`, so use it as you wish :)
- `matrix: np.ndarray` needs to be a [Markov matrix](https://en.wikipedia.org/wiki/Stochastic_matrix)
- call `print(MarkovChain.random(3).run(record=True))` to generate your first process
### Reusing `MarkovChain`
- setting `iter_reset = True` resets `step`, generates first state, seeds `_state_rng`, clears `count`
- in order to extend or merge processes set `iter_reset = False`, and adjust desired properites
- call `reset` after `iter_reset = False` to discard the previous process
### Initial state
- Setting `initial_state` with a 1-D `list | np.ndarray` of probabilities allows to randomly choose the first state; uses `_static_rng` as the generator
- Setting with an `int` makes sure this is always your first state
### Setting seed
- Calling static method `reset_static_rng` generates identical `matrix` (in `random`) and first `_state`
- Setting `my_seed` creates identical process, but unique for every first state

## TODO
- state translator( a dict ?)
- comment doc clean-up, ?import clean-up
- advance steps, manual and ?fast
- create tests 
- statistics example