"""Circuit builders for the qiskit/IBMQ interface"""
from qiskit import QuantumRegister, QuantumCircuit
from numpy import pi

from ..circuitBuilder import circuitBuilder

class __qiskitGeneralBuilder__(circuitBuilder):
    """Abstract class for Qiskit-circuits builders. 
    """
    def __init__(self, nbqbits):
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
        """Add measurement.
        """
        self.qc.measure_all()

    def alldiam(self, idx=None):
        """Add X-rotation of pi/2.

        Parameters
        ----------
        idx : iterable, optional
            Indices on which to apply the rotation, by default None. If None, apply to all qubits.
        """
        raise NotImplementedError

    def input(self, idx, theta):
        """Add input gate. It is a Y-rotation of angle $\pi - \theta$

        Parameters
        ----------
        idx : Union[iterable, int]
            Index[-ices] of qubits on which to input theta.
        theta : Union[list-like, float]
            Parameter[s] to input. Has to have the same length as `idx`.
        """
        raise NotImplementedError

    def allin(self, x):
        """Add input gate to all qubits and input x.

        Parameters
        ----------
        x : list-like
            Parameters to input.
        """
        raise NotImplementedError

    def cc(self, a, b):
        """Add CC gate between qubits `a` and `b`

        Parameters
        ----------
        a : int
            Control qubit
        b : int
            Target qubit.
        """
        raise NotImplementedError

class qiskitBuilder(__qiskitGeneralBuilder__):
    """Qiskit-circuits builder using rx, rz and cz gates.
    """
    def __init__(self, nbqbits):
        super().__init__(nbqbits)
    
    def alldiam(self, idx=None):
        if idx is None:
            self.qc.rx(pi/2, self.qr)
        else:
            for i in idx:
                self.qc.rx(pi/2, self.qr[i])

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                self.qc.rz(theta[p], self.qr[i])
                self.qc.rx(pi/2, self.qr[i])
        else:
            self.qc.rz(theta, self.qr[idx])
            self.qc.rx(pi/2, self.qr[idx])

    def allin(self, x):
        for i, qb in enumerate(self.qr):
            self.qc.rz(x[i], qb)

        self.qc.rx(pi/2, self.qr)

    def cc(self, a, b):
        self.qc.cz(self.qr[a], self.qr[b])
        self.qc.rx(pi/2, [self.qr[a], self.qr[b]])

class ibmqNativeBuilder(__qiskitGeneralBuilder__):
    """Qiskit-circuits builder using IBMQ native gates (u1, u2, and cx).
    """
    def __init__(self, nbqbits):
        super().__init__(nbqbits)

    def alldiam(self, idx=None):
        if idx is None:
            self.qc.u2(-pi/2, pi/2, self.qr)
        else:
            for i in idx:
                self.qc.u2(-pi/2, pi/2, self.qr[i])

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                self.qc.u1(theta[p], self.qr[i])
                self.qc.u2(-pi/2, pi/2, self.qr[i])
        else:
            self.qc.u1(theta, self.qr[idx])
            self.qc.u2(-pi/2, pi/2, self.qr[idx])

    def allin(self, x):
        for i, qb in enumerate(self.qr):
            self.qc.u1(x[i], qb)

        self.qc.u2(-pi/2, pi/2, self.qr)

    def cc(self, a, b):
        self.qc.u2(-pi/2, pi/2, self.qr[b])
        self.qc.u1(pi/2, self.qr[b])
        self.qc.cx(self.qr[a], self.qr[b])
        self.qc.u1(-pi/2, self.qr[b])
        self.qc.u2(-pi/2, pi/2, self.qr[a])
