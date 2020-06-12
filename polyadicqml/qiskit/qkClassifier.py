from ..quantumClassifier import Classifier

from .qiskitML import qiskitML

class qkClassifier(Classifier):
    """Quantum classifier using qiskitML circuits.
    """
    def __init__(self, circuit, bitstr, nbshots=None,
                 nbshots_increment=None,
                 nbshots_incr_delay=None,
                 job_size=None, budget=200, name=None,
                 save_jobs_info=False,
                 save_path=None,  params=None):
        """ Create Classifier.

        Parameters
        ----------
        circuit : qiskitML
            Quantum circuit wrapper.
        bitstr : list of int or list of str
            Which bitstrings should correspond to each class. The number of classes for the classification is defined by the number of elements.
        nbshots : int, optional
            Number of shots for the quantum circuit. If 0, negative or None, then exact proabilities are computed, by default None
        nbshots_increment : float, int or callable, optional
            How to increase the number of shots as optimization progress. If float or int, the increment arise every `nbshots_incr_delay` iterations: if float, then the increment is multiplicative; if int, then it is added. If callable, the new nbshots is computed by calling `nbshots_increment(nbshots, n_iter)`.
        nbshots_incr_delay : int, optional
            After how many iteration nb_shots has to increse. By default 20, if nbshots_increment is given
        job_size : int, optional
            Number of runs for each qiskit job, by default the number of observations.
        budget : int, optional
            Maximum number of optimization steps, by default 200
        name : srt, optional
            Name to identify this classifier.
        save_jobs_info : bool, optional
            Where to save intermediate training results, by deafult None. If None, intermediate results are not saved.
        """
        super().__init__(circuit, bitstr, nbshots=nbshots,
                         nbshots_increment=nbshots_increment,
                         nbshots_incr_delay=nbshots_incr_delay,
                         budget= budget, 
                         name=name, save_path=save_path)

        if params is None:
            self.set_params(circuit.random_params())
        else:
            self.set_params(params)

        self.job_size = job_size

        self.__save_jobs__ = save_jobs_info

        self.__info__['job_size'] = job_size if job_size else "FULL"
    
    def __verify_circuit__(self, circuit):
        """Test wheter a circuit is valid and raise TypeError if it is not.

        Parameters
        ----------
        circuit : qiskitML
            circuit implementation for qiskit

        Raises
        ------
        TypeError
        """
        if not isinstance(circuit, qiskitML):
            raise TypeError(f"Circuit was type {type(circuit)} while qiskitML was expected.")
    
    def __set_circuit__(self, circuit):
        """Set the circuit after testing for validity.

        Parameters
        ----------
        circuit : qiskitML
            circuit implementation for qiskit

        Raises
        ------
        TypeError
            If the circuit is invalid.
        """
        self.__verify_circuit__(circuit)
        self.circuit = circuit

    def run_circuit(self, X, params=None):
        if params is None:
            params = self.params

        return self.circuit.run(X, params, self.nbshots,
                               job_size=self.job_size, save_jobs_info=self.__save_jobs__)