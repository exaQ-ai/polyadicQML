"""Implementation of circuit for ML
"""
from numpy import pi, random

class circuitML():
    def __init__(self, make_circuit, nbqbits, nbparams, cbuilder):
        self.nbqbits = nbqbits
        self.nbparams = nbparams

        self.circuitBuilder = cbuilder
        self.make_circuit = make_circuit

    def run(self, X, params, shots=None, job_size=None):
        """Run the circuit with input `X` and parameters `params`.
        
        Parameters
        ----------
        X : array-like
            Input matrix of shape (nb_samples, nb_features).
        params : vector-like
            Parameter vector.
        shots : int, optional
            Number of shots for the circuit run, by default None. If None, uses the backend default.
        job_size : int, optional
            Maximum job size, to split the circuit runs, by default None. If None, put all nb_samples in the same job. 

        Returns
        -------
        array
            Bitstring counts as an array of shape (nb_samples, 2**nbqbits)
        """
        raise NotImplementedError

    def random_params(self, seed=None):
        """Generate a valid vector of random parameters.

        Parameters
        ----------
        seed : int, optional
            random seed, by default None

        Returns
        -------
        vector
        """
        if seed: random.seed(seed)
        return random.randn(self.nbparams)

    def make_circuit(self, x, params, shots=None):
        """Generate the circuit corresponding to input `x` and `params`.
        NOTE: This function is provided by the user.

        Parameters
        ----------
        self : circuitML
            A circuit ML
        x : vector-like
            Input sample
        params : vector-like
            Parameter vector.
        shots : int, optional
            Number of shots, by default None

        Returns
        -------
        quantum circuit
        """
        raise NotImplementedError

    def __repr__(self):
        return "<circuitML>"

    def __str__(self):
        return self.__repr__()
    