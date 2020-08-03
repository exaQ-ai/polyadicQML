"""Module for auto-supervised clustering -- and classification.
"""

import numpy as np
from scipy.spatial.distance import cdist

from .quantumModel import quantumModel, circuitML


class QMeans(quantumModel):
    def __init__(
        self,
        circuit: circuitML, nclasses: int,
        **kwargs
    ):
        super().__init__(circuit, **kwargs)

        self.means = np.empty((nclasses, 2**circuit.nbqbits))

    def predict_proba(self, X: np.ndarray, params=None) -> np.ndarray:
        dists = self.run_circuit(X, params)
        return cdist(dists, self.means, metric="cityblock")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=0)

    def fit(
        self, input_train, target_train, batch_size=None, **kwargs
    ):
        pass
