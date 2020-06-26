from qiskit import execute
from .qkCircuitML import qkCircuitML, np, sleep, qiskitBuilder

class parallelML(qkCircuitML):
    """Quantum ML circuit interface for running two qkCircuitML circuits on parallel on the same QPU. 

    Parameters
    ----------
    backend : Union[Backends, list, qiskit.providers]
        Backend on which to run the circuits
    make_circuit : callable of signature self.make_circuit
        Function to generate the circuit corresponding to input `x` and `params`.
    nbqbits : int
        Number of qubits.
    nbparams : int
        Number of parameters.
    cbuilder : circuitBuilder, optional
        Circuit builder, by default qiskitBuilder
    noise_model : Union[list, qiskit.providers.aer.noise.NoiseModel], optional
        Noise model to be provided to the backend, by default None. Cannot be used with `noise_backend`.
    noise_backend : Union[Backends, list, qiskit.IBMQBackend], optional
        IBMQ backend from which the noise model should be generated, by default None.
    save_path : str, optional
        Where to save the jobs outputs, by default None. Jobs are saved only if a path is specified

    Raises
    ------
    ValueError
        If both `noise_model` and `noise_backend` are provided.
    """
    def __init__(self, backend,  nbqbits, nbparams, cbuilder=qiskitBuilder, noise_model=None, noise_backend=None,
                 save_path=None):
        super().__init__(backend, cbuilder, nbqbits=nbqbits,
                         noise_model=noise_model, noise_backend=noise_backend,
                         save_path=save_path)

        self._len_out = 0
    
    def make_circuit_list(self, X, params, nbshots=None):
        self._len_out = len(X)
        return [self.make_circuit(X[i:i+2], params, nbshots)
                        for i in range(0, self._len_out, 2)]

    def result(self, job, qc_list, nbshots=None):
        wait = 1
        while not job.done():
            sleep(wait)

        results = job.result()
        if not nbshots:
            raise NotImplementedError
        else:
            out = np.zeros((2*len(qc_list), 2**(self.nbqbits//2)))
            for n, qc in enumerate(qc_list):
                for key, count in results.get_counts(qc).items():
                    # print(f"{key} : {count}")
                    #! ATTENTION: the order of the qubits and bitstring is reversed
                    key = key[::-1]
                    key0 = key[:(self.nbqbits//2)]
                    key1 = key[-(self.nbqbits//2):]
                    out[2*n, int(key0, 2)] += count
                    out[2*n + 1, int(key1, 2)] += count

        if self.save_path: self.save_job(job)
        return out[:self._len_out]
