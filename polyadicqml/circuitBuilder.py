"""Circuit builders"""


class circuitBuilder():
    """Builder to create parametrized circuit with repetitive structures, by
    defining general operations without directly writing gates.

    Parameters
    ----------
    nbqbits : int
        Number of qubits.
    """
    def __init__(self, nbqbits, *args, **kwargs):
        super().__init__()
        self.nbqbits = nbqbits

    def __verify_index__(self, idx):
        """Test if the index is valid for this circuit.

        Parameters
        ----------
        idx : int
            Index

        Raises
        ------
        ValueError
            If the index is out of bounds
        TypeError
            If the index is not int
        """
        if not isinstance(idx, int):
            raise TypeError("Index must be integer")
        if self.nbqbits < idx or idx < 0:
            raise ValueError(
                f"Index {idx} is out of range for circuit with {self.nbqbits} qubits"
            )

    def circuit(self):
        """Return the built circuit.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    ##############################
    # GATES

    def measure_all(self):
        """Add measurement.

        Returns
        -------
        circuitBuilder
            self
        """
        raise NotImplementedError

    def alldiam(self, idx=None):
        """Add X-rotation of :math:`\\pi/2`.

        Parameters
        ----------
        idx : iterable, optional
            Indices on which to apply the rotation, by default ``None``.
            If ``None``, apply to all qubits.

        Returns
        -------
        circuitBuilder
            self
        
        Raises
        ------
        ValueError
            If idx is out of range
        TypeError
            If the indices are not int
        """
        raise NotImplementedError

    def input(self, idx, theta):
        """Add input gate.

        It correspond to a rotation of :math:`RX(\\pi/2) RZ(theta) RX(\\pi/2)`.

        Parameters
        ----------
        idx : Union[iterable, int]
            Index[-ices] of qubits on which to input theta.
        theta : Union[list-like, float]
            Parameter[s] to input. Has to have the same length as ``idx``.

        Returns
        -------
        circuitBuilder
            self
        
        Raises
        ------
        ValueError
            If idx is out of range
        TypeError
            If the indices are not int
        """
        raise NotImplementedError

    def allin(self, theta):
        """Add input gate to all qubits and input theta.

        Parameters
        ----------
        theta : list-like
            Parameters to input.

        Returns
        -------
        circuitBuilder
            self
        """
        raise NotImplementedError

    def cz(self, a, b):
        """Add CZ gate between qubits `a` and `b`

        Parameters
        ----------
        a : int
            Control qubit
        b : int
            Target qubit.

        Returns
        -------
        circuitBuilder
            self
        
        Raises
        ------
        ValueError
            If a or b are out of range
        TypeError
            If the indices are not int
        """
        raise NotImplementedError
