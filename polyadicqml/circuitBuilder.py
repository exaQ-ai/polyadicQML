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