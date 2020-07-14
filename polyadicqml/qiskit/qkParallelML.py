from .qkCircuitML import qkCircuitML, np, sleep
from .qkBuilder import qkParallelBuilder


class qkParallelML(qkCircuitML):
    """Quantum ML circuit interface for running two qkCircuitML circuits on
    parallel on the same QPU.

    Parameters
    ----------
    make_circuit : callable of signature self.make_circuit
        Function to generate the circuit corresponding to input `x` and
        `params`.
    make_circuit : callable of signature self.make_circuit
        Function to generate the circuit corresponding to input `x` and
        `params`.
    nbqbits : int
        Number of qubits of the circuit.
    nbparams : int
        Number of parameters.
    backend : Union[Backends, list, qiskit.providers.BaseBackend]
        Backend on which to run the circuits
    tot_nbqbits : int
        Number of total qubits in the QPU. Has to be at least twice `nbqbits`.
    cbuilder : circuitBuilder, optional
        Circuit builder, by default :class:`qkParallelBuilder`
    noise_model : Union[list, qiskit.providers.aer.noise.NoiseModel], optional
        Noise model to be provided to the backend, by default ``None``. Cannot
        be used with `noise_backend`.
    noise_backend : Union[list, qiskit.providers.ibmq.IBMQBackend], optional
        IBMQ backend from which the noise model should be generated, by
        default ``None``.
    save_path : str, optional
        Where to save the jobs outputs, by default ``None``. Jobs are saved
        only if a path is specified

    Attributes
    ----------
    nbqbits : int
        Number of qubits.
    tot_nbqbits : int
        Number of tatal qubits in the QPU
    nbparams : int
        Number of parameters.

    Raises
    ------
    ValueError
        If both `noise_model` and `noise_backend` are provided.
        If QPU size cannot contain two circuits.
    """
    def __init__(
        self, make_circuit, nbqbits, nbparams, backend,
        tot_nbqbits, cbuilder=qkParallelBuilder,
        noise_model=None, noise_backend=None,
        save_path=None
    ):
        super().__init__(
            make_circuit, nbqbits, nbparams, backend, cbuilder=cbuilder,
            noise_model=noise_model, noise_backend=noise_backend,
            save_path=save_path
        )

        if tot_nbqbits < 2 * nbqbits:
            raise ValueError(
                f"QPU too small to run two circuits (tot_nbqbits {tot_nbqbits} < 2 * {nbqbits})"
            )

        if self.job_limit:
            self.job_limit *= 2
        self.tot_nbqbits = tot_nbqbits
        self._len_out = 0

    def make_circuit_list(self, X, params, nbshots=None):
        def post(bdr):
            if nbshots:
                return bdr.measure_all().circuit()
            return bdr.circuit()

        self._len_out = len(X)
        return [
            post(
                self.make_circuit(
                    self._circuitBuilder(
                        self.nbqbits,
                        self.tot_nbqbits if i+2 < self._len_out else None
                    ),
                    X[i:i+2].T, params
                )
            ) for i in range(0, self._len_out, 2)
        ]

    def result(self, job, qc_list, nbshots=None):
        wait = 1
        while not job.done():
            sleep(wait)

        results = job.result()
        if not nbshots:
            raise NotImplementedError
        else:
            out = np.zeros((2*len(qc_list), 2**self.tot_nbqbits))
            for n, qc in enumerate(qc_list):
                for key, count in results.get_counts(qc).items():
                    # print(f"{key} : {count}")
                    # !ATTENTION: the order of the qubits and bitstring is
                    # reversed
                    key = key[::-1]
                    key0 = key[:self.nbqbits]
                    key1 = key[-self.nbqbits:]
                    out[2*n, int(key0, 2)] += count
                    out[2*n + 1, int(key1, 2)] += count

        if self.save_path:
            self.save_job(job)

        return out[:self._len_out]

    def run(self, X, params, nbshots=None, job_size=None):
        return super().run(
            X, params, nbshots=nbshots,
            job_size= 2 * job_size if job_size else job_size
        )
