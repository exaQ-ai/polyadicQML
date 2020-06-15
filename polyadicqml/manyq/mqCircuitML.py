"""Implementeation of quantum circuit for ML using manyQ simulator.
"""
from ..circuitML import circuitML

import numpy as np

class mqCircuitML(circuitML):
    def __init__(self, circuitBuilder, nbqbits):
        super().__init__(circuitBuilder, nbqbits)

    def run(self, X, params, shots=None, job_size=None):
        _X = X.T 
        _params = np.hstack(len(X)* (params.reshape(-1,1),))
        result = self.make_circuit(_X, _params, shots)
        return result(shots).T

    def make_circuit(self, x, params, shots=None):
        """Generate the circuit corresponding to input `x` and `params`.

        Parameters
        ----------
        x : vector-like
            Input sample
        params : vector-like
            Parameter vector.
        shots : int, optional
            Number of shots, by default None

        Returns
        -------
        quantum circuit
        """
        raise NotImplementedError
    