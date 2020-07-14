"""Implementeation of quantum circuit for ML using qiskit API.
"""
import qiskit as qk
from qiskit.providers.aer.noise import NoiseModel
from qiskit.exceptions import QiskitError
from qiskit.providers import JobStatus

from sys import exc_info
from os.path import isfile
import numpy as np

from time import asctime, sleep
from itertools import cycle
import json

from .utility.backends import Backends
from ..circuitML import circuitML
from .qkBuilder import __qiskitGeneralBuilder__, qkBuilder


class qkCircuitML(circuitML):
    """Quantum ML circuit interface for qiskit and IBMQ.
    Provides a unified interface to run multiple parametric circuits with
    different input and model parameters.

    Parameters
    ----------
    make_circuit : callable of signature self.make_circuit
        Function to generate the circuit corresponding to input `x` and
        `params`.
    nbqbits : int
        Number of qubits.
    nbparams : int
        Number of parameters.
    backend : Union[Backends, list, qiskit.providers.BaseBackend]
        Backend(s) on which to run the circuits
    cbuilder : circuitBuilder, optional
        Circuit builder, by default :class:`qkBuilder`
    noise_model : Union[list, qiskit.providers.aer.noise.NoiseModel], optional
        Noise model to be provided to the backend, by default ``None``. Cannot
        be used with `noise_backend`.
    coupling_map : list, optional
        Coupling map to be provided to the backend, by default ``None``.
        Cannot be used with `noise_backend`.
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
    nbparams : int
        Number of parameters.

    Raises
    ------
    ValueError
        If both `noise_model` and `noise_backend` are provided.
    """
    def __init__(self, make_circuit, nbqbits, nbparams, backend,
                 cbuilder=qkBuilder,
                 noise_model=None, coupling_map=None,
                 noise_backend=None,
                 save_path=None):
        super().__init__(make_circuit, nbqbits, nbparams, cbuilder)

        self.save_path = save_path

        if isinstance(backend, Backends):
            self.__backend__ = backend
            self.backend = self.__backend__.backends
            self.noise_model = self.__backend__.noise_models
            self.coupling_map = self.__backend__.coupling_maps
            self.job_limit = backend.job_limit

        else:
            backend = backend if isinstance(backend, list) else [backend]
            try:
                self.job_limit = min(map(lambda x: x.job_limit(), backend))
            except AttributeError:
                self.job_limit = None
            self.backend = cycle(backend)

            if noise_model is not None and noise_backend is not None:
                raise ValueError(
                    "Only one between 'noise_model' and 'noise_backend' can \
be passed to the constructor"
                )

            if isinstance(noise_model, list):
                self.noise_model = cycle(noise_model)
            else:
                self.noise_model = cycle([noise_model])

            if isinstance(coupling_map, list):
                self.coupling_map = cycle(coupling_map)
            else:
                self.coupling_map = cycle([coupling_map])

            if noise_backend is not None:
                _noise_back = noise_backend
                if not isinstance(noise_backend, list):
                    _noise_back = [noise_backend]

                self.noise_model = cycle(
                    [NoiseModel.from_backend(_backend)
                        for _backend in _noise_back]
                )
                self.coupling_map = cycle(
                    [_backend.configuration().coupling_map
                        for _backend in _noise_back]
                )

    def __verify_builder__(self, cbuilder):
        bdr = cbuilder(1)
        if isinstance(bdr, __qiskitGeneralBuilder__): return
        raise TypeError(
            f"The circuit builder class is not comaptible: provided \
{cbuilder} expected {__qiskitGeneralBuilder__}"
        )

    def run(self, X, params, nbshots=None, job_size=None):
        if not job_size:
            if self.job_limit is not None and len(X) > self.job_limit:
                job_size = self.job_limit
        elif self.job_limit is not None and job_size > self.job_limit:
            raise ValueError(
                f"Job size {job_size} greater that job limit {self.job_limit}"
            )
        try:
            if not job_size:
                job, qc_list = self.request(X, params, nbshots)
                try:
                    return self.result(job, qc_list, nbshots)
                except:
                    status = job.status()
                    if job.done() or status == JobStatus.DONE:
                        print(f"Completed job {job.job_id()} on {job.backend().name()}")
                    elif status in (JobStatus.CANCELLED, JobStatus.ERROR):
                        print(f"{status} ({job.job_id()}) on {job.backend().name()}")
                    else:
                        print(f"Cancelling job {job.job_id()} on {job.backend().name()}")
                        job.cancel()
                    raise
            else:
                if not isinstance(job_size, int):
                    raise TypeError("'job_size' has to be int")

                n_jobs = len(X) // job_size
                requests = [
                    self.request(X[job_size * n : job_size * (n+1)], params, nbshots)
                    for n in range(n_jobs)
                ]
                if job_size * n_jobs < len(X):
                    requests.append(
                        self.request(X[job_size * n_jobs :], params, nbshots)
                    )
                try:
                    return np.vstack([
                        self.result(job, qc_list, nbshots)
                        for job, qc_list in requests
                    ])
                except:
                    for job, qc_list in requests:
                        status = job.status()
                        if job.done() or status == JobStatus.DONE:
                            print(f"Completed job {job.job_id()} on {job.backend().name()}")
                        elif status in (JobStatus.CANCELLED, JobStatus.ERROR):
                            print(f"{status} ({job.job_id()}) on {job.backend().name()}")
                        else:
                            print(f"Cancelling job {job.job_id()} on {job.backend().name()}")
                            job.cancel()
                    raise
        except KeyboardInterrupt:
            cin = input("[r] to reload backends, [ctrl-c] to confirm interrupt :\n")
            if cin == 'r':
                self.__backend__.load_beckends()
                self.backend = self.__backend__.backends
                self.noise_model = self.__backend__.noise_models
                self.coupling_map = self.__backend__.coupling_maps

            return self.run(X, params, nbshots, job_size)
        except QiskitError as descr:
            error_str = f"{asctime()} - Error in qkCircuitML.run :{exc_info()[0]}\n\t{descr}\n"
            print(error_str)
            with open("error.log", "w") as f:
                f.write(error_str)
            sleep(5)
            return self.run(X, params, nbshots, job_size)

    def make_circuit_list(self, X, params, nbshots=None):
        """Generate a circuit for each sample in `X` rows, with parameters
        `params`.

        Parameters
        ----------
        X : array-like
            Input matrix, of shape *(nb_samples, nb_features)* or
            *(nb_features,)*. In the latter case, *nb_samples* is 1.
        params : vector-like
            Parameter vector.
        nbshots : int, optional
            Number of nbshots, by default ``None``

        Returns
        -------
        list[qiskit.QuantumCircuit]
            List of *nb_samples* circuits.
        """
        def post(bdr):
            if nbshots:
                return bdr.measure_all().circuit()
            return bdr.circuit()

        if len(X.shape) < 2:
            return [post(
                self.make_circuit(
                    self._circuitBuilder(self.nbqbits), X, params
                )
            )]
        else:
            return [
                post(
                    self.make_circuit(
                        self._circuitBuilder(self.nbqbits), x, params
                    )
                )
                for x in X]

    def request(self, X, params, nbshots=None):
        """Create circuits corresponding to samples in `X` and parameters
        `params` and send jobs to the backend for execution.

        Parameters
        ----------
        X : array-like
            Input matrix, of shape *(nb_samples, nb_features)* or
            *(nb_features,)*. In the latter case, *nb_samples* is 1.
        params : vector-like
            Parameter vector.
        nbshots : int, optional
            Number of nbshots, by default ``None``

        Returns
        -------
        (qiskit.providers.BaseJob, list[qiskit.QuantumCircuit])
            Job instance derived from BaseJob and list of corresponding
            circuits.
        """
        qc_list = self.make_circuit_list(X, params, nbshots)

        # Optional arguments for execute are defined here, if they have been
        # given at construction.
        execute_kwargs = {}
        if nbshots:
            execute_kwargs['shots'] = nbshots

        _noise_model = next(self.noise_model)
        if _noise_model is not None:
            execute_kwargs['basis_gates'] = _noise_model.basis_gates
            execute_kwargs['noise_model'] = _noise_model
        _coupling_map = next(self.coupling_map)
        if _coupling_map is not None:
            execute_kwargs['coupling_map'] = _coupling_map

        return qk.execute(
            qc_list, next(self.backend),
            **execute_kwargs,
        ), qc_list

    def result(self, job, qc_list, nbshots=None):
        """Retrieve job results and returns bitstring counts.

        Parameters
        ----------
        job : qiskit.providers.BaseJob
            Job instance.
        qc_list : list[qiskit.circuit.QuantumCircuit]
            List of quantum circuits executed in `job`, of length *nb_samples*.
        nbshots : int, optional
            Number of shots, by default ``None``. If ``None``, raw counts are
            returned.

        Returns
        -------
        array
            Bitstring counts as an array of shape *(nb_samples, 2**nbqbits)*,
            in the same order as `qc_list`.

        Raises
        ------
        QiskitError
            If job status is cancelled or had an error.
        """
        wait = 1
        while not job.done():
            if job.status() in (JobStatus.CANCELLED, JobStatus.ERROR):
                raise QiskitError
            sleep(wait)

        results = job.result()
        if not nbshots:
            out = [results.get_statevector(qc) for qc in qc_list]
            out = np.abs(out)**2
            order = [
                int(f"{key:0>{self.nbqbits}b}"[::-1], 2)
                for key in range(out.shape[1])
            ]
            return out[:, order]
        else:
            out = np.zeros((len(qc_list), 2**self.nbqbits))
            for n, qc in enumerate(qc_list):
                for key, count in results.get_counts(qc).items():
                    # print(f"{key} : {count}")
                    out[n, int(key[::-1], 2)] = count

        if self.save_path:
            self.save_job(job)
        return out

    def save_job(self, job, save_path=None):
        """Save job output to json file.

        Parameters
        ----------
        job : qiskit.providers.BaseJob
            Job instance.
        save_path : path, optional
            Where to save the output, by default ``None``. If None, uses
            :attr:`qkCircuitML.save_path`.
        """
        save_path = self.save_path if save_path is None else save_path

        if isfile(save_path):
            try:
                with open(save_path) as f:
                    out = json.load(f)
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                print(f"ATTENTION: file {save_path} is broken, confirm overwriting!")
                input("Keybord interrupt ([ctrl-c]) to abort")
                out = {}
        else:
            out = {}

        with open(save_path, 'w') as f:
            job_id = job.job_id()
            try:
                times = job.time_per_step()
                info = {key: str(times[key]) for key in times}
            except AttributeError:
                info = {}
            info['results'] = job.result().to_dict()

            out[job_id] = info
            json.dump(out, f)
