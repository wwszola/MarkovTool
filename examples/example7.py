from itertools import pairwise, islice
import numpy as np
from typing import OrderedDict

from MarkovTool import Markov, Endless, Collector

errors = OrderedDict()
for dim in range(3, 6):
    d1 = Markov(dim, 0).fill_random(0)
    d1.initial_state = 0
    f = Endless(d1)

    base = 3
    low_exponent, exponent = 4, 7
    history = f.skip(base ** exponent)

    d2 = Markov(d1.dimension, 0, initial_state = 0)
    errors[dim] = OrderedDict()
    for p in range(low_exponent, exponent + 1):
        data = pairwise(islice(history, base ** p))
        d2.fit(data, (0.0, 1.0))    
        wrongs = 0
        for old, new in zip(islice(history, base ** p), Endless(d2)):
            if old != new:
                wrongs += 1
        error = d2._matrix - d1._matrix
        error = np.sum(error ** 2) / d2.dimension ** 2
        errors[dim][p] = error, wrongs

header = [f'{base ** p}' for p in range(low_exponent, exponent + 1)]
print('\t', ('\t'*2).join(header), sep = '') 
for dim_, errors_ in errors.items():
    line = f'{dim_}:\t'
    sections = []
    for error, wrongs in errors_.values():
        sections.append('{:.3E} {}'.format(error, wrongs))
    line += '\t'.join(sections)
    print(line)        