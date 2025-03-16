import numpy as np
from collections import namedtuple

def mysoftmax(arr: list):
    arr = np.array(arr)
    sum_of_exps = sum(np.exp(elem) for elem in arr)
    arr = np.exp(arr) / sum_of_exps
    return arr


topk_namedtuple = namedtuple('topk_namedtuple', ['values', 'indices'])

def topk(array: np.ndarray, k: int, largest: bool = True) -> topk_namedtuple:
    """Returns the k largest/smallest elements and corresponding indices 
    from an array-like input.

    Parameters
    ----------
    array : np.ndarray or list
        the array-like input
    k : int
        the k in "top-k" 
    largest ： bool, optional
        controls whether to return largest or smallest elements        

    Returns
    -------
    namedtuple[values, indices]
        Returns the :attr:`k` largest/smallest elements and corresponding indices 
        of the given :attr:`array`

    Example
    -------
    >>> array = [5, 3, 7, 2, 1]
    >>> topk(array, 2)
    >>> topk_namedtuple(values=array([7, 5]), indices=array([2, 0], dtype=int64))

    >>> topk(array, 2, largest=False)
    >>> topk_namedtuple(values=array([1, 2]), indices=array([4, 3], dtype=int64))

    >>> array = [[1, 2], [3, 4], [5, 6]]
    >>> topk(array, 2)
    >>> topk_namedtuple(values=array([6, 5]), indices=(array([2, 2], dtype=int64), array([1, 0], dtype=int64)))
    """

    array = np.asarray(array)
    flat = array.ravel()

    if largest:
        indices = np.argpartition(flat, -k)[-k:]
        argsort = np.argsort(-flat[indices])
    else:
        indices = np.argpartition(flat, k)[:k]
        argsort = np.argsort(flat[indices])

    indices = indices[argsort]
    values = flat[indices]
    indices = np.unravel_index(indices, array.shape)
    if len(indices) == 1:
        indices, = indices
    return topk_namedtuple(values=values, indices=indices)