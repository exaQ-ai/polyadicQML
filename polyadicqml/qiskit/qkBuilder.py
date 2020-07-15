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
            Built circuit
        """
        return self.qc

    def measure_all(self):
        self.qc.measure_all()

        return self


class qkBuilder(__qiskitGeneralBuilder__):
    """Qiskit-circuits builder using rx, rz and cz gates.

    Parameters
    ----------
    nbqbits : int
        Number of qubits.
    """
    def __init__(self, nbqbits, *args, **kwargs):
        super().__init__(nbqbits)

    def alldiam(self, idx=None):
        if idx is None:
            self.qc.rx(pi/2, self.qr)
        else:
            if isinstance(idx, int):
                idx = [idx]
            for i in idx:
                self.__verify_index__(i)
                self.qc.rx(pi/2, self.qr[i])

        return self

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                self.__verify_index__(i)

                self.qc.rx(pi/2, self.qr[i])
                self.qc.rz(theta[p], self.qr[i])
                self.qc.rx(pi/2, self.qr[i])
        else:
            self.__verify_index__(idx)

            self.qc.rx(pi/2, self.qr[idx])
            self.qc.rz(theta, self.qr[idx])
            self.qc.rx(pi/2, self.qr[idx])

        return self

    def allin(self, theta):
        self.qc.rx(pi/2, self.qr)
        for i, qb in enumerate(self.qr):
            self.qc.rz(theta[i], qb)
        self.qc.rx(pi/2, self.qr)

        return self

    def cz(self, a, b):
        self.__verify_index__(a)
        self.__verify_index__(b)

        self.qc.cz(self.qr[a], self.qr[b])
        return self


class ibmqNativeBuilder(__qiskitGeneralBuilder__):
    """Qiskit-circuits builder using IBMQ native gates (u1, u2, and cz).

    Parameters
    ----------
    nbqbits : int
        Number of qubits.
    """
    def __init__(self, nbqbits, *args, **kwargs):
        super().__init__(nbqbits)

    def alldiam(self, idx=None):
        if idx is None:
            self.qc.u2(-pi/2, pi/2, self.qr)
        else:
            if isinstance(idx, int):
                idx = [idx]
            for i in idx:
                self.__verify_index__(i)
                self.qc.u2(-pi/2, pi/2, self.qr[i])

        return self

    def input(self, idx, theta):
        if isinstance(idx, list):
            for p, i in enumerate(idx):
                self.__verify_index__(i)

                self.qc.u2(-pi/2, pi/2, self.qr[i])
                self.qc.u1(theta[p], self.qr[i])
                self.qc.u2(-pi/2, pi/2, self.qr[i])
        else:
            self.__verify_index__(idx)

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
        self.__verify_index__(a)
        self.__verify_index__(b)

        self.qc.cz(self.qr[a], self.qr[b])
        return self


class qkParallelBuilder(__qiskitGeneralBuilder__):
    """Qiskit-circuits builder for running parallel on same QPU.
    Uses rx, rz and cz gates.

    Parameters
    ----------
    nbqbits : int
        Number of qubits.
    tot_nbqbits : int, optional
        Number if qubits in the QPU, by default None. If None, this is
        equivalent to :class:`qkBuilder`
    """
    def __init__(self, nbqbits, tot_nbqbits=None, *args, **kwargs):
        super().__init__(tot_nbqbits if tot_nbqbits else nbqbits)
        self.c_nbq = nbqbits
        self.start_2 = tot_nbqbits - nbqbits if tot_nbqbits else None

    def alldiam(self, idx=None):
        if idx is None:
            self.qc.rx(pi/2, self.qr[:self.c_nbq])
            if self.start_2:
                self.qc.rx(pi/2, self.qr[self.start_2:])
        else:
            if isinstance(idx, int):
                idx = [idx]
            for i in idx:
                self.__verify_index__(i)

                self.qc.rx(pi/2, self.qr[i])
                if self.start_2:
                    self.qc.rx(pi/2, self.qr[self.start_2 + i])

        return self

    def input(self, idx, theta):
        if isinstance(idx, list):
            if len(theta.shape) > 1:
                t1, t2 = theta.T
            else:
                t1, t2 = theta, theta
            for p, i in enumerate(idx):
                self.__verify_index__(i)

                self.qc.rx(pi/2, self.qr[i])
                self.qc.rz(t1[p], self.qr[i])
                self.qc.rx(pi/2, self.qr[i])

                if self.start_2:
                    self.qc.rx(pi/2, self.qr[self.start_2 + i])
                    self.qc.rz(t2[p], self.qr[self.start_2 + i])
                    self.qc.rx(pi/2, self.qr[self.start_2 + i])
        else:
            self.__verify_index__(idx)

            try:
                t1, t2 = theta
            except (TypeError, ValueError):
                t1, t2 = theta, theta

            self.qc.rx(pi/2, self.qr[idx])
            self.qc.rz(t1, self.qr[idx])
            self.qc.rx(pi/2, self.qr[idx])

            if self.start_2:
                self.qc.rx(pi/2, self.qr[self.start_2 + idx])
                self.qc.rz(t2, self.qr[self.start_2 + idx])
                self.qc.rx(pi/2, self.qr[self.start_2 + idx])

        return self

    def allin(self, theta):
        if len(theta.shape) > 1 and theta.shape[1] > 1:
            t1, t2 = theta.T
        else:
            t1, t2 = theta, theta

        qr1 = self.qr[:self.c_nbq]
        self.qc.rx(pi/2, qr1)
        for i, qb in enumerate(qr1):
            self.qc.rz(t1[i], qb)
        self.qc.rx(pi/2, qr1)

        if self.start_2:
            qr2 = self.qr[self.start_2:]
            self.qc.rx(pi/2, qr2)
            for i, qb in enumerate(qr2):
                self.qc.rz(t2[i], qb)
            self.qc.rx(pi/2, qr2)

        return self

    def cz(self, a, b):
        self.__verify_index__(a)
        self.__verify_index__(b)

        self.qc.cz(self.qr[a], self.qr[b])

        if self.start_2:
            self.qc.cz(
                self.qr[self.start_2 + a], self.qr[self.start_2 + b]
            )

        return self
