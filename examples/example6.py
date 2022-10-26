from concurrent.futures import process
from MarkovTool import Description, Dependent

from itertools import islice

process = Description((3, 3), my_seed=5).fill_random()

# tree connection instead of graph model should be sufficient
# won't work with shape checking in init
node1 = Dependent(process, None)
node2 = Dependent(process, node1)
node1._parent = node2
node2._state = 0

for v1, v2 in islice(zip(node1, node2), 0, 15):    
    print(v1, v2)