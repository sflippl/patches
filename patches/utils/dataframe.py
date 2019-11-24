"""Utilities for converting arrays into dataframes.
"""

import functools

import numpy as np
import pandas as pd

def array_to_dataframe(array):
    try:
        dims = array.shape
        flat_array = array.flatten()
    except AttributeError as e:
        raise TypeError('array must be a numpy array or a similar object.') from e
    dct_flat = {
        "dim%d"%i: np.array(
            np.repeat(
                range(dims[i]), functools.reduce(lambda x,y:x*y, dims[i+1:], 1)
            ).tolist() * functools.reduce(lambda x,y:x*y, dims[:i], 1)
        ) for i in range(len(dims))
    }
    dct_flat['array'] = flat_array
    df_flat = pd.DataFrame(dct_flat)
    return df_flat
