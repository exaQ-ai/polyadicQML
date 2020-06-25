from qiskit import execute
from .qkCircuitML import qkCircuitML, np, sleep, qiskitBuilder

class parallelML(qkCircuitML):
    """Quantum ML circuit interface for running two qkCircuitML circuits on parallel on the same QPU. 
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
