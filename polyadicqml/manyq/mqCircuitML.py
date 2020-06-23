"""Implementeation of quantum circuit for ML using manyQ simulator.
"""
from ..circuitML import circuitML

import numpy as np

from .manyqBdr import manyqBdr

class mqCircuitML(circuitML):
    """Quantum ML circuit interface for manyq simulator.
    Provides a unified interface to run multiple parametric circuits with different input and model parameters. 

    Parameters
    ----------
    make_circuit : callable of signature self.make_circuit
        Function to generate the circuit corresponding to input `x` and `params`.
    nbqbits : int
        Number of qubits.
    nbparams : int
        Number of parameters.
    cbuilder : circuitBuilder, optional
        Circuit builder, by default manyqBuilder

    Raises
    ------
    ValueError
        If both `noise_model` and `noise_backend` are provided.
    """
    def __init__(self, make_circuit, nbqbits, nbparams, cbuilder=manyqBdr):
        super().__init__(make_circuit, nbqbits, nbparams, cbuilder)

    def __single_run__(self, X, params, shots=None):
        batch_size = 1 if len(X.shape) < 2 else len(X)
        
        _X = X.T 
        _params = np.hstack(batch_size* (params.reshape(-1,1),))
        result = self.make_circuit(self.circuitBuilder(self.nbqbits, batch_size=batch_size),
                                   _X, _params, shots)

        return result(shots).T if batch_size > 1 else result.ravel()

    def run(self, X, params, shots=None, job_size=None):
        if job_size:
            raise NotImplementedError

        return self.__single_run__(X, params, shots)