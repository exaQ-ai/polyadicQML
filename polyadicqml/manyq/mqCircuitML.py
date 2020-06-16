"""Implementeation of quantum circuit for ML using manyQ simulator.
"""
from ..circuitML import circuitML

import numpy as np

from .manyqBdr import manyqBdr

class mqCircuitML(circuitML):
    def __init__(self, make_circuit, nbqbits, nbparams, cbuilder=manyqBdr):
        super().__init__(make_circuit, nbqbits, nbparams, cbuilder)

    def run(self, X, params, shots=None, job_size=None):
        _X = X.T 
        _params = np.hstack(len(X)* (params.reshape(-1,1),))
        result = self.make_circuit(self, _X, _params, shots)

        return result(shots).T