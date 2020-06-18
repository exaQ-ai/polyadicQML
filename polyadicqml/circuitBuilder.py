"""Circuit builders"""

class circuitBuilder():
    """Builder to create parametrized circuit with repetitive structures, by defining general operations without directly writing gates.
    """
    def __init__(self, nbqbits, *args, **kwargs):
        """Create builder

        Parameters
        ----------
        nbqbits : int
            Number of qubits.
        """
        super().__init__()
        self.nbqbits = nbqbits

    def circuit(self):
        """Return the built circuit.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    ############################################# 
    # GATES

    def measure_all(self):
        """Add measurement.
        """
        raise NotImplementedError

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
