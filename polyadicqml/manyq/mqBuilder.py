"""Circuit builders for the manyQ simulator"""
from ..circuitBuilder import circuitBuilder

import manyq as mq


class mqBuilder(circuitBuilder):
    """Builder for circuits to be run on manyQ simulator.
    """
    def __init__(self, nbqbits, batch_size, *args, gpu=False, **kwargs):
        super().__init__(nbqbits)
        mq.initQreg(nbqbits, batch_size, gpu=gpu)

        self.__txt = ""

    def __run_circuit__(self, nbshots=None):
        if not nbshots:
            return mq.measureAll()
        else:
            return mq.makeShots(nbshots)

    def __call__(self, nbshots=None):
        return self.__run_circuit__(nbshots)

    def __str__(self):
        return self.__txt

    def __repr__(self):
        return "mqBuilder(" + str(self) + ")"

    def circuit(self):
        """Return the built circuit.

        Returns
        -------
        quantum circuit
        """
        return self

    ##############################
    # GATES

    def measure_all(self):
        mq.measureAll()

        self.__txt += "measure_all()"

        return self

    def alldiam(self, idx=None):
        if idx is None:
            idx = range(self.nbqbits)
        if isinstance(idx, int):
            idx = [idx]
        for i in idx:
            self.__verify_index__(i)
            mq.SX(i)

            self.__txt += f"SX({i})"

        return self

    def __single_input(self, idx, theta):
        self.__verify_index__(idx)
        mq.SX(idx)
        mq.RZ(idx, theta)
        mq.SX(idx)

        self.__txt += f"SX({idx})RZ({idx},{theta})SX({idx})"

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                self.__single_input(i, theta[p])
        else:
            self.__single_input(idx, theta)

        return self

    def allin(self, theta):
        for i in range(self.nbqbits):
            self.__single_input(i, theta[i])

        return self

    def cz(self, a, b):
        self.__verify_index__(a)
        self.__verify_index__(b)

        mq.CZ(a, b)

        self.__txt += f"CZ({a},{b})"

        return self
