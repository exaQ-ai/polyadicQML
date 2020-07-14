"""Implementeation of quantum circuit for ML using manyQ simulator.
"""
from ..circuitML import circuitML

import numpy as np

# TODO: verify import
__gpu_available__ = True
try:
    from cupy import asarray, hstack
except ModuleNotFoundError:
    __gpu_available__ = False

from .mqBuilder import mqBuilder


class mqCircuitML(circuitML):
    """Quantum ML circuit interface for manyq simulator.
    Provides a unified interface to run multiple parametric circuits with
    different input and model parameters.

    Parameters
    ----------
    make_circuit : callable of signature self.make_circuit
        Function to generate the circuit corresponding to input `x` and
        `params`.
    nbqbits : int
        Number of qubits.
    nbparams : int
        Number of parameters.
    cbuilder : circuitBuilder, optional
        Circuit builder, by default mqBuilder

    Attributes
    ----------
    nbqbits : int
        Number of qubits.
    nbparams : int
        Number of parameters.

    Raises
    ------
    ValueError
        If both `noise_model` and `noise_backend` are provided.
    """
    def __init__(
        self, make_circuit, nbqbits, nbparams, gpu=False, cbuilder=mqBuilder
    ):
        super().__init__(make_circuit, nbqbits, nbparams, cbuilder)
        if gpu and not __gpu_available__:
            raise ModuleNotFoundError(
                "No module named 'cupy', install for gpu support."
            )
        self.__gpu = gpu

    def __verify_builder__(self, cbuilder):
        bdr = cbuilder(1, 1)
        if not isinstance(bdr, mqBuilder):
            raise TypeError(
                    f"The circuit builder class is not compatible: provided \
                        {cbuilder} expected {mqBuilder}"
            )

    def __single_run__(self, X, params, nbshots=None):
        batch_size = 1 if len(X.shape) < 2 else len(X)

        _X = X.T
        _params = None
        if self.__gpu:
            _X = asarray(_X)
            _params = hstack(batch_size * (asarray(params).reshape(-1, 1),))
        else:
            _params = np.hstack(batch_size * (params.reshape(-1, 1),))

        bdr = self.make_circuit(
            self._circuitBuilder(
                self.nbqbits, batch_size=batch_size, gpu=self.__gpu
            ),
            _X, _params
        )
        if nbshots: bdr.measure_all()

        result = bdr.circuit()(nbshots)

        return result.T if batch_size > 1 else result.ravel()

    def run(self, X, params, nbshots=None, job_size=None):
        if job_size:
            raise NotImplementedError

        return self.__single_run__(X, params, nbshots)

    def gpu(self):
        """Switch to cupy.
        """
        if not __gpu_available__:
            raise ModuleNotFoundError(
                "No module named 'cupy', install for gpu support."
            )
        self.__gpu = True

    def cpu(self):
        """Switch to numpy.
        """
        self.__gpu = False
