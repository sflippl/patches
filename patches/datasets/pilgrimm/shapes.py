"""Define different shapes.
"""

import numpy as np

def horizontal(n):
    if n<2:
        raise ValueError('n must be at least 2.')
    return np.array([[1.]*n])

def vertical(n):
    if n<2:
        raise ValueError('n must be at least 2.')
    return np.array([[1.]]*n)

def diagonal(n):
    if n<2:
        raise ValueError('n must be at least 2.')
    return np.identity(n)

def antidiagonal(n):
    if n<2:
        raise ValueError('n must be at least 2.')
    return np.identity(n)[range(n-1,-1,-1)]
