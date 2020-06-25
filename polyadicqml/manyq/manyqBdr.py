"""Circuit builders for the manyQ simulator"""
from ..circuitBuilder import circuitBuilder

import manyq as mq
from numpy import pi

class manyqBuilder(circuitBuilder):
    """Builder for circuits to be run on manyQ simulator. 
    """
    def __init__(self, nbqbits, batch_size, *args, gpu=False, **kwargs):
        super().__init__(nbqbits)
        mq.initQreg(nbqbits, batch_size, gpu=gpu)


    def __run_circuit__(self, nbshots=None):
        if not nbshots:
            return mq.measureAll()
        else:
            return mq.makeShots(nbshots)

    def __call__(self, nbshots=None):
        return self.__run_circuit__(nbshots)

    def circuit(self):
        """Return the built circuit.

        Returns
        -------
        quantum circuit
        """
        return self

    def measure_all(self):
        mq.measureAll()

        return self

    def alldiam(self, idx=None):
        if idx is None:
            idx = range(self.nbqbits)
        for i in idx:
            mq.SX(i)

        return self

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                mq.SX(i)
                mq.RZ(self.qr[i], theta[p])
                mq.SX(i)
        else:
            mq.SX(idx)
            mq.RZ(self.qr[idx], theta)
            mq.SX(idx)

        return self

    def allin(self, theta):
        for i in range(self.nbqbits):
            mq.SX(i)
            mq.RZ(i, theta[i])
            mq.SX(i)

        return self

    def cz(self, a, b):
        mq.CZ(a, b)

        return self