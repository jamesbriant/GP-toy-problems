from dataclasses import dataclass, field

import numpy as np

@dataclass
class Chain:
    """
    """
    chain_length: int
    parameter_names: set
    _chain: dict = field(init=False)
    _index: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._chain = {parameter_name: np.zeros(self.chain_length) for parameter_name in self.parameter_names}

    @property
    def chain(self) -> dict:
        return self._chain

    def update(self, new_values: dict) -> None:
        for param_name, value in new_values.items():
            self._chain[param_name][self._index] = value
        self._index += 1