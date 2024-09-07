"""
Author: Baljyot Singh Parmar
"""

import numpy as np
import matplotlib.image as mpimg


def read_file(file_loc):
    """
    Parameters
        ----------

    file_loc : str
                path to the file

    Returns
        -------

    Array-like
                the array is 2D array of the pixel locations
    """
    img = mpimg.imread(file_loc)
    return img


def reshape_col2d(arr, permutations):
    """
    Docstring for reshape_col2d
    This function reshapes a 2D array by permuting the columns in the order specified by permutations

    Parameters:
    -----------
    arr : numpy array
        The array to be reshaped
    permutations : list of integers
        The permutations to be applied to the columns of arr

    Returns:
    --------
    numpy array
        The reshaped array

    NOTES:
    ------
    Sometimes this breaks i have no idea why. Use with caution

    """
    # Check that permutations is a list of integers
    if not isinstance(permutations, list):
        raise TypeError("permutations must be a list")
    if not all([isinstance(i, int) for i in permutations]):
        raise TypeError("permutations must be a list of integers")
    # Check that permutations is a permutation of np.arange(len(permutations))
    idx = np.empty_like(permutations)
    idx[permutations] = np.arange(len(permutations))
    arr[:] = arr[:, idx]
    return arr


def rescale_range(x, min_x, max_x, a, b):
    """https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range"""
    if min_x >= max_x:
        raise ValueError("min_x={} is not less than max_x={}".format(min_x, max_x))
    if a >= b:
        raise ValueError("a={} is not less than b={}".format(a, b))
    return ((b - a) * (x - min_x) / (max_x - min_x)) + a
