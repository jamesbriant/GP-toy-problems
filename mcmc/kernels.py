import numpy as np

def log_RBF(
    lengthscale: float,
    x: float,
    xprime: np.ndarray,
) -> np.ndarray:
    """Calculates the logarithm of the squared exponential function of given
    lengthscale using at a singular x location but multiple xprime locations.

    """
    output = (x - xprime)**2
    return -output/(2*lengthscale**2)