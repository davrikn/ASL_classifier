import torch
import numpy as np


class TorchQueue:

    """
    Inplace queue in which adding new values will overwrite the value
    at the position of head. Retrieving a specific index will return
    the given index displaced from the head index.
    For example, if the queue is [1, 2, 3, 4], and it is represented
    as [3, 4, 1, 2] (head at 2), then queue[1] will return the array at
    index 2+1 = 3, i.e. 2
    """

    def __init__(self, init: torch.Tensor) -> None:
        """
        buffer (int): Length of queue
        init (Array): Array to be initialized to
        """
        # Init must be a numpy array with at least two dimensions:
        assert len(init.shape) > 1 

        if isinstance(init, np.ndarray):
            init = torch.tensor(init)
        elif isinstance(init, torch.Tensor):
            pass
        else:
            raise RuntimeError("Invalid init type")

        # Initializing class variables:
        self.__n = len(init)
        self.__inst_shape = init[0].shape
        self.__queue = init
        self.__head = 0
    

    def __len__(self) -> int:
        return self.__n

    
    def __getitem__(self, ind: int = None) -> torch.Tensor:
        if ind is None:
            return self.__queue
        elif isinstance(ind, int):
            ind = (self.__head + ind) % self.__n
            return self.__queue[ind]
        else:
            raise IndexError("Invalid index type")
    

    def insert(self, val: torch.Tensor) -> None:
        """
        Updates the queue by setting a value at head.
        """
        assert val.shape == self.__inst_shape
        self.__queue[self.__head] = val
        self.__head = (self.__head + 1) % self.__n