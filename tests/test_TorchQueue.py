import numpy as np
import torch
from utility.torch_queue import TorchQueue


def test_TorchQueue():
    init = np.zeros((5, 3, 3))
    queue = TorchQueue(init)

    f = lambda x, y: torch.tensor(x**2 + y**2)
    rng = np.arange(3)
    X, Y = np.meshgrid(rng, rng)
    
    for i in range(9):
        queue.insert(f(i*X, i*Y))
    
    print(queue[None], end="\n\n")
    print(queue[1])