"""Module for auto-supervised clustering -- and classification.
"""

import numpy as np
from scipy.spatial.distance import cdist

from .quantumModel import quantumModel, circuitML
from .utility import stable_softmax


class QMeans(quantumModel):
    def __init__(
        self,
        circuit: circuitML, nclasses: int,
        **kwargs
    ):
        super().__init__(circuit, **kwargs)

        self.means = np.empty((nclasses, 2**circuit.nbqbits))

    def dists(self, X: np.ndarray, params=None) -> np.ndarray:
        probs = self.run_circuit(X, params)
        return cdist(probs, self.means, metric="euclidean")

    def predict_proba(self, X: np.ndarray, params=None) -> np.ndarray:
        return stable_softmax(
            - self.dists(X, params),    # NOTE negative distances
            axis=1
        )

    def __update_centers__(self, probs, labels=None):
        if labels is None:
            labels = np.argmin(
                cdist(probs, self.means, metric="euclidean"),
                axis=1
            )

        for i in range(len(self.means)):
            self.means[i] = np.mean(
                probs[labels == i],
                axis=0
            )

    def __update_params__(self, probs, labels=None, **kwargs):
        pass

    def fit(
        self, input_train, target_train, batch_size=None, **kwargs
    ):
        pass
