# Random Functions that help with anything
import numpy as np


def check_type(arr, wanted_type):
    """
    Check the type of all elements in a NumPy array.

    Parameters:
    - values: NumPy array
    - wanted_type: The desired type for all elements in the array

    Returns:
    - True if all elements have the specified type, False otherwise
    """
    # Check if the input is a NumPy array
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    # Use vectorized comparison to check the type of all elements at once
    return np.all(np.vectorize(lambda x: type(x) == wanted_type)(arr))
