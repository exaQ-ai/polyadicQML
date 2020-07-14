"""Implementation of circuit for ML
"""
from numpy import pi, random, zeros_like, zeros, log2


class circuitML():
    """Abstract Quantum ML circuit interface.
    Provides a unified interface to run multiple parametric circuits with
    different input and model parameters, agnostic of the backend, implemented
    in the subclasses.

    Parameters
    ----------
    make_circuit : callable of signature self.make_circuit
        Function to generate the circuit corresponding to input `x` and
        `params`.
    nbqbits : int
        Number of qubits.
    nbparams : int
        Number of parameters.
    cbuilder : circuitBuilder
        Circuit builder class to be used. It must correspond to the subclass
        implementation.

    Attributes
    ----------
    nbqbits : int
        Number of qubits.
    nbparams : int
        Number of parameters.
    """
    def __init__(self, make_circuit, nbqbits, nbparams, cbuilder):
        self.nbqbits = nbqbits
        self.nbparams = nbparams

        self.__set_builder__(cbuilder)
        self.make_circuit = make_circuit

    def __set_builder__(self, cbuilder):
        self.__verify_builder__(cbuilder)
        self._circuitBuilder = cbuilder

    def __verify_builder__(self, cbuilder):
        raise NotImplementedError

    def run(self, X, params, nbshots=None, job_size=None):
        """Run the circuit with input `X` and parameters `params`.

        Parameters
        ----------
        X : array-like
            Input matrix of shape *(nb_samples, nb_features)*.
        params : vector-like
            Parameter vector.
        nbshots : int, optional
            Number of shots for the circuit run, by default ``None``. If
            ``None``, uses the backend default.
        job_size : int, optional
            Maximum job size, to split the circuit runs, by default ``None``.
            If ``None``, put all *nb_samples* in the same job.

        Returns
        -------
        array
            Bitstring counts as an array of shape *(nb_samples, 2**nbqbits)*
        """
        raise NotImplementedError

    def random_params(self, seed=None):
        """Generate a valid vector of random parameters.

        Parameters
        ----------
        seed : int, optional
            random seed, by default ``None``

        Returns
        -------
        vector
            Vector of random parameters.
        """
        if seed: random.seed(seed)
        return random.randn(self.nbparams)

    def make_circuit(self, bdr, x, params):
        """Generate the circuit corresponding to input `x` and `params`.
        NOTE: This function is to be provided by the user, with the present
        signature.

        Parameters
        ----------
        bdr : circuitBuilder
            A circuit builder.
        x : vector-like
            Input sample
        params : vector-like
            Parameter vector.

        Returns
        -------
        circuitBuilder
            Instructed builder
        """
        raise NotImplementedError

    def __eq__(self, other):
        return self.make_circuit is other.make_circuit

    def __repr__(self):
        return "<circuitML>"

    def __str__(self):
        return self.__repr__()

    def grad(self, X, params, v=None, eps=None, nbshots=None, job_size=None):
        """Compute the gradient of the circuit w.r.t. parameters *params* on
        input *X*.

        Uses finite differences of the circuit runs.

        Parameters
        ----------
        X : array-like
            Input matrix of shape *(nb_samples, nb_features)*.
        params : vector-like
            Parameter vector of length *nb_params*.
        v : array-like
            Vector or matrix to right multiply the Jacobian with.
        eps : float, optional
            Epsilon for finite differences. By default uses ``1e-8`` if
            `nbshots` is not provided, else uses :math:`\\pi /
            \\sqrt{\\text{nbshots}}`
        nbshots : int, optional
            Number of shots for the circuit run, by default ``None``. If
            ``None``, uses the backend default.
        job_size : int, optional
            Maximum job size, to split the circuit runs, by default ``None``.
            If ``None``, put all *nb_samples* in the same job.

        Returns
        -------
        array
            Jacobian matix as an array of shape *(nb_params, 2**nbqbits)* if
            `v` is None, else Jacobian-vector product: ``J(circuit) @ v``
        """
        dim_out = 2**self.nbqbits
        if v is not None:
            if len(v.shape) > 1:
                dim_out = v.shape[0]
            else:
                dim_out = 1

        if eps is None:
            if nbshots is None:
                eps = 1e-8
            else:
                max(log2(self.nbqbits)*2*pi/3 * min(.5, 1/nbshots**.25), 1e-8)

        num = eps if nbshots is None else eps * nbshots

        out = zeros((self.nbparams, dim_out))
        run_out = self.run(X, params, nbshots, job_size) / num
        for i in range(len(params)):
            d = zeros_like(params)
            d[i] = eps
            pd = self.run(X, params + d, nbshots, job_size) / num - run_out

            out[i] = pd if v is None else pd @ v

        return out
