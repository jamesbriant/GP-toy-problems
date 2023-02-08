import numpy as np

def convert_to_nparray(values) -> np.ndarray:
    """
    Converts input to a numpy array.

    Args:
    -----
        input: np.ndarray, list of floats or float
            to be converted
    Returns:
    --------
        numpy array with values from input
    """
    if not isinstance(values, np.ndarray):
        if not isinstance(values, list):
            values = [values]
        return np.array(values)
    return values