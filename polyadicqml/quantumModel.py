"""Abstract quantum model with sklearn API
"""

from .circuitML import circuitML


class quantumModel():
    def __init__(self, circuit, **kwargs):
        # Retrieve keyword arguments
        params = kwargs.get('params')
        nbshots = kwargs.get('nbshots')
        nbshots_increment = kwargs.get('nbshots_increment')
        nbshots_incr_delay = kwargs.get('nbshots_incr_delay')
        job_size = kwargs.get('job_size')

        # Testing circuit and setting it
        self.set_circuit(circuit)

        # Setting parameters
        if params is None:
            self.set_params(circuit.random_params())
        else:
            self.set_params(params)

        # Testing for nbshots type
        if not (isinstance(nbshots, int) or (nbshots is None)):
            raise TypeError("Invalid `nbshots` type")
        if nbshots is not None and nbshots < 1:
            nbshots = None
        self.nbshots = nbshots

        # Testing for nbshots_incr_delay
        if not (
            isinstance(nbshots_incr_delay, int) or (nbshots_incr_delay is None)
        ):
            raise TypeError("Invalid `nbshots_incr_delay` type")
        self.nbshots_incr_delay = 20
        if nbshots_incr_delay is not None:
            self.nbshots_incr_delay = nbshots_incr_delay

        self.__set_nbshots_increment__(nbshots_increment)

        self.job_size = job_size
        self.nfev = 0

    def __verify_circuit__(self, circuit):
        """Test wheter a circuit is valid and raise TypeError if it is not.

        Parameters
        ----------
        circuit : circuitML
            QML circuit

        Raises
        ------
        TypeError
            If the circuit is not a circuitML
        ValueError
            If self has a circuit and the new circuit does not uses the same
            make_circuit fuction
        """
        if not isinstance(circuit, circuitML):
            raise TypeError(
                f"Circuit was type {type(circuit)} while circuitML was \
                expected."
            )
        if hasattr(self, 'circuit'):
            if self.circuit != circuit:
                raise ValueError(
                    "Given circuit is different from previous circuit"
                )

    def set_circuit(self, circuit):
        """Set the circuit after testing for validity.

        For a circuit to be valid, it has to be an instance of circuitML and,
        in case self already has a circuit, to use the same make_circuit
        function.

        Parameters
        ----------
        circuit : circuitML
            QML circuit

        Raises
        ------
        Union[TypeError, ValueError]
            If the circuit is invalid.
        """
        self.__verify_circuit__(circuit)
        self.circuit = circuit

    def set_params(self, params):
        """Parameters setter

        Parameters
        ----------
        params : vector
            Parameters vector
        """
        self.params = params

    def __set_nbshots_increment__(self, nbshots_increment):
        __incr__ = nbshots_increment
        if nbshots_increment is None:
            def __incr__(nbshots, n_iter, loss_value):
                return nbshots
        elif isinstance(nbshots_increment, float):
            def __incr__(nbshots, n_iter, loss_value):
                if n_iter % self.nbshots_incr_delay == 0:
                    return int(nbshots_increment * nbshots)
                else:
                    return nbshots
        elif isinstance(nbshots_increment, int):
            def __incr__(nbshots, n_iter, loss_value):
                if n_iter % self.nbshots_incr_delay == 0:
                    return nbshots + nbshots_increment
                else:
                    return nbshots

        self.nbshots_increment = __incr__

    def run_circuit(self, X, params=None):
        """Run the circuit with input `X` and parameters `params`.

        Parameters
        ----------
        X : array-like
            Input matrix of shape (nb_samples, nb_features).
        params : vector-like, optional
            Parameter vector, by default uses the model
            :attr:`~polyadicqml.Classifier.params`

        Returns
        -------
        array
            Bitstring counts as an array of shape (nb_samples, 2**nbqbits)
        """
        if params is None:
            params = self.params

        self.nfev += 1

        return self.circuit.run(
            X, params, self.nbshots, job_size=self.job_size
        )

    def predict_proba(self, X, params=None):
        """Compute the bitstring probabilities associated to each input point
        of the design matrix.

        Parameters
        ----------
        X : array
            Design matrix of n samples
        params : vector, optional
            Circuit parameters, by default None. If not given, model
            parameters are used.
        """
        raise NotImplementedError

    def predict(self, X):
        """Compute the predicted class for each input point of the design
        matrix.

        Parameters
        ----------
        X : array
            Design matrix of n samples
        """
        raise NotImplementedError

    def __call__(self, X):
        """Compute the predicted class for each input point of the design
        matrix.
        Equivalent to :meth:`~polyadicqml.quantumClassifier.predict`

        Parameters
        ----------
        X : array
            Design matrix of n samples
        params : vector, optional
            Circuit parameters, by default None. If not given, model
            parameters are used.

        Returns
        -------
        vector
            Labels vector
        """
        return self.predict(X)

    def fit(self, input_train, target_train, batch_size=None,
            **kwargs):
        """Fit the model according to the given training data.

        Parameters
        ----------
        input_train : array
            Training design matrix.
        target_train : vector
            Labels corresponding to `input_train`.
        batch_size : int, optional
            Minibatches size, by default None. If none uses the full dataset
            with rndom shuffle at each iteration.
        """
        raise NotImplementedError
