"""Circuit builders for the manyQ simulator"""
from ..circuitBuilder import circuitBuilder

import manyq as mq
from numpy import pi

class manyqBdr(circuitBuilder):
    """Builder for circuits to be run on manyQ simulator. 
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
        mq.measureAll()

    def alldiam(self, idx=None):
        if idx is None:
            idx = range(self.nbqbits)
        for i in idx:
            mq.SX(i)

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                mq.RZ(self.qr[i], theta[p])
                mq.SX(i)
        else:
            mq.RZ(self.qr[idx], theta)
            mq.SX(idx)

    def allin(self, x):
        for i in range(self.nbqbits):
            mq.RZ(i, x[i])
            mq.SX(i)

    def cc(self, a, b):
        mq.CZ(a, b)
        mq.SX(a)
        mq.SX(b)