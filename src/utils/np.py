from functools import lru_cache, wraps

import numpy as np


def np_cache(*lru_args, array_argument_index=0, **lru_kwargs):
    """
    https://gist.github.com/Susensio/61f4fee01150caaac1e10fc5f005eb75
    LRU cache implementation for functions whose parameter at ``array_argument_index`` is a numpy array of dimensions <= 2

    Example:
    >>> from sem_env.utils.cache import np_cache
    >>> array = np.array([[1, 2, 3], [4, 5, 6]])
    >>> @np_cache(maxsize=256)
    ... def multiply(array, factor):
    ...     return factor * array
    >>> multiply(array, 2)
    >>> multiply(array, 2)
    >>> multiply.cache_info()
    CacheInfo(hits=1, misses=1, maxsize=256, currsize=1)
    """

    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            np_array = args[array_argument_index]
            if len(np_array.shape) > 2:
                raise RuntimeError(
                    f"np_cache is currently only supported for arrays of dim. less than 3 but got shape: {np_array.shape}"
                )
            hashable_array = tuple(map(tuple, np_array))
            args = list(args)
            args[array_argument_index] = hashable_array
            return cached_wrapper(*args, **kwargs)

        @lru_cache(*lru_args, **lru_kwargs)
        def cached_wrapper(*args, **kwargs):
            hashable_array = args[array_argument_index]
            array = np.array(hashable_array)
            args = list(args)
            args[array_argument_index] = array
            return function(*args, **kwargs)

        # copy lru_cache attributes over too
        wrapper.cache_info = cached_wrapper.cache_info
        wrapper.cache_clear = cached_wrapper.cache_clear
        return wrapper

    return decorator
