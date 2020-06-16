"""Circuit builders for the manyQ simulator"""
from ..circuitBuilder import circuitBuilder

import manyq as mq
from numpy import pi

class manyqBdr(circuitBuilder):
    """Abstract class for Qiskit-circuits builders. 
    """
    def __init__(self, nbqbits, job_size, *args, gpu=False, **kwargs):
        super().__init__(nbqbits)
        mq.initQreg(nbqbits, job_size, gpu=gpu)


    def __run_circuit__(self, shots=None):
        if not shots:
            return mq.measureAll()
        else:
            return mq.makeShots(shots)

    def circuit(self):
        """Return the built circuit.

        Returns
        -------
        quantum circuit
        """
        return self.__run_circuit__

    def measure_all(self):
        """Add measurement.
        """
        mq.measureAll()

    def alldiam(self, idx=None):
        """Add X-rotation of pi/2.

        Parameters
        ----------
        idx : iterable, optional
            Indices on which to apply the rotation, by default None. If None, apply to all qubits.
        """
        if idx is None:
            idx = range(self.nbqbits)
        for i in idx:
            mq.SX(i)

    def input(self, idx, theta):
        """Add input gate. It is a Y-rotation of angle $\pi - \theta$

        Parameters
        ----------
        idx : Union[iterable, int]
            Index[-ices] of qubits on which to input theta.
        theta : Union[list-like, float]
            Parameter[s] to input. Has to have the same length as `idx`.
        """
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                mq.RZ(self.qr[i], theta[p])
                mq.SX(i)
        else:
            mq.RZ(self.qr[idx], theta)
            mq.SX(idx)

    def allin(self, x):
        """Add input gate to all qubits and input x.

        Parameters
        ----------
        x : list-like
            Parameters to input.
        """
        for i in range(self.nbqbits):
            mq.RZ(i, x[i])
            mq.SX(i)

    def cc(self, a, b):
        """Add CC gate between qubits `a` and `b`

        Parameters
        ----------
        a : int
            Control qubit
        b : int
            Target qubit.
        """
        mq.CZ(a, b)
        mq.SX(a)
        mq.SX(b)