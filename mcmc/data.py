import numpy as np
from dataclasses import dataclass

@dataclass(repr=False)
class Data:
    """
    """
    x_c: np.ndarray
    t: np.ndarray
    y: np.ndarray
    x_f: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        self.d = np.hstack([self.y, self.z])

        self.y_n = self.y.shape[0]
        self.z_n = self.z.shape[0]