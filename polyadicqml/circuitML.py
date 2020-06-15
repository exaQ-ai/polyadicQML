"""Implementation of circuit for ML using the manyQ simulator
"""

class circuitML():
    def __init__(self, circuitBuilder, nbqbits):
        self.nbqbits = nbqbits
        self.circuitBuilder = circuitBuilder

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
        raise NotImplementedError