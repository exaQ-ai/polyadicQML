"""Circuit builders for the qiskit/IBMQ interface"""
from qiskit import QuantumRegister, QuantumCircuit
from numpy import pi

from ..circuitBuilder import circuitBuilder

class __qiskitGeneralBuilder__(circuitBuilder):
    """Abstract class for Qiskit-circuits builders. 
    """
    def __init__(self, nbqbits, *args, **kwargs):
        super().__init__(nbqbits)
        self.qr = QuantumRegister(self.nbqbits, 'qr')
        self.qc = QuantumCircuit(self.qr)

    def circuit(self):
        """Return the built circuit.

        Returns
        -------
        qiskit.QuantumCircuit
        """
        return self.qc

    def measure_all(self):
        self.qc.measure_all()

        return self


class qiskitBuilder(__qiskitGeneralBuilder__):
    """Qiskit-circuits builder using rx, rz and cz gates.
    """
    def __init__(self, nbqbits, *args, **kwargs):
        super().__init__(nbqbits)
    
    def alldiam(self, idx=None):
        if idx is None:
            self.qc.rx(pi/2, self.qr)
        else:
            for i in idx:
                self.qc.rx(pi/2, self.qr[i])

        return self

    # def inputY(self, idx, theta):
    #     if isinstance(idx, list):
    #         for p, i in enumerate(idx):
    #             self.qc.ry(pi - theta[p], self.qr[i])
    #     else:
    #         self.qc.ry(pi - theta, self.qr[idx])
    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                self.qc.rx(pi/2, self.qr[i])
                self.qc.rz(theta[p], self.qr[i])
                self.qc.rx(pi/2, self.qr[i])
        else:
            self.qc.rx(pi/2, self.qr[idx])
            self.qc.rz(theta, self.qr[idx])
            self.qc.rx(pi/2, self.qr[idx])

        return self

    # def allinY(self, theta):
    #     for i, qb in enumerate(self.qr):
    #         self.qc.ry(pi - theta[i], qb)
    def allin(self, x):
        self.qc.rx(pi/2, self.qr)
        for i, qb in enumerate(self.qr):
            self.qc.rz(x[i], qb)
        self.qc.rx(pi/2, self.qr)

        return self

    def cz(self, a, b):
        self.qc.cz(self.qr[a], self.qr[b])

        return self

class ibmqNativeBuilder(__qiskitGeneralBuilder__):
    """Qiskit-circuits builder using IBMQ native gates (u1, u2, and cz).
    """
    def __init__(self, nbqbits, *args, **kwargs):
        super().__init__(nbqbits)

    def alldiam(self, idx=None):
        if idx is None:
            self.qc.u2(-pi/2, pi/2, self.qr)
        else:
            for i in idx:
                self.qc.u2(-pi/2, pi/2, self.qr[i])

        return self

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                self.qc.u2(-pi/2, pi/2, self.qr[i])
                self.qc.u1(theta[p], self.qr[i])
                self.qc.u2(-pi/2, pi/2, self.qr[i])
        else:
            self.qc.u2(-pi/2, pi/2, self.qr[idx])
            self.qc.u1(theta, self.qr[idx])
            self.qc.u2(-pi/2, pi/2, self.qr[idx])

        return self

    def allin(self, theta):
        self.qc.u2(-pi/2, pi/2, self.qr)
        for i, qb in enumerate(self.qr):
            self.qc.u1(theta[i], qb)

        self.qc.u2(-pi/2, pi/2, self.qr)

        return self

    def cz(self, a, b):
        self.qc.cz(self.qr[a], self.qr[b])

        return self
