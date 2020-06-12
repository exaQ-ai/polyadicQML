from qiskit import execute
from .qiskitML import qiskitML, np, sleep

class parallelML(qiskitML):
    """Quantum ML circuit interface for running two qiskitML circuits on parallel on the same QPU. 
    """
    def __init__(self, backend, circuitBuilder, nbqbits, noise_model=None, noise_backend=None,
                 save_path="jobs.json"):
        """Create a parallelML circuit.

        Parameters
        ----------
        backend : Union[Backends, list, qiskit.providers]
            Backend on which to run the circuits
        circuitBuilder : circuitBuilder
            Circuit builder.
        nbqbits : int
            Number of qubits.
        noise_model : Union[list, qiskit.providers.aer.noise.NoiseModel], optional
            Noise model to be provided to the backend, by default None. Cannot be used with `noise_backend`.
        noise_backend : Union[Backends, list, qiskit.IBMQBackend], optional
            IBMQ backend from which the noise model should be generated, by default None.
        save_path : str, optional
            Where to save the jobs outputs, by default "jobs.json". Jobs are saved only if so is specified when calling `qiskitML.run`.

        Raises
        ------
        ValueError
            If both `noise_model` and `noise_backend` are provided.
        """
        super().__init__(backend, circuitBuilder, nbqbits=nbqbits,
                         noise_model=noise_model, noise_backend=noise_backend,
                         save_path=save_path)

        self._len_out = 0
    
    def make_circuit_list(self, X, params, shots=None):
        self._len_out = len(X)
        return [self.make_circuit(X[i:i+2], params, shots)
                        for i in range(0, self._len_out, 2)]

    def result(self, job, qc_list, shots=None, save_info=False):
        wait = 1
        while not job.done():
            sleep(wait)
            if wait < 120 : wait *= 5

        results = job.result()
        if not shots:
            raise NotImplementedError
        else:
            out = np.zeros((2*len(qc_list), 2**(self.nbqbits//2)))
            for n, qc in enumerate(qc_list):
                for key, count in results.get_counts(qc).items():
                    # print(f"{key} : {count}")
                    #! ATTENTION: the order of the qubits and bitstring is reversed
                    key1 = key[:(self.nbqbits//2)]
                    key0 = key[-(self.nbqbits//2):]
                    out[2*n, int(key0, 2)] += count
                    out[2*n + 1, int(key1, 2)] += count

        if save_info: self.save_job(job)
        return out[:self._len_out]
