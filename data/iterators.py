import torch
import itertools
import numpy as np

class Grouped_Iterator:
    """Wrapper around an iterable that returns groups (chunks) of items.

    Args:
        iterable (iterable): iterable to wrap
        chunk_size (int): size of each chunk
    """

    def __init__(self, iterable, len_iter, chunk_size):
        self.length = int(np.ceil(len_iter / float(chunk_size)))
        self.iterable = iterable
        self.chunk_size = chunk_size

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        chunk = []
        try:
            for i in range(self.chunk_size):
                chunk.append(next(self.iterable))
        except StopIteration as e:
            if len(chunk) == 0:
                raise e
        return chunk




