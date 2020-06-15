from ..circuitML import circuitML, np
from ..parallelCircuitML import parallelML

############################################################
# MODELS

class irisCircuit(circuitML):
    """Two qubit circuit for four-dimensional input, with 8 parameters.
    """
    def __init__(self, backend, circuitBuilder, noise_model=None, noise_backend=None,
                 save_path="jobs.json"):
        super().__init__(backend, circuitBuilder, nbqbits=2,
                         noise_model=noise_model, noise_backend=noise_backend, save_path=save_path)
    
    def make_circuit(self, x, params, shots=None):
        bdr = self.circuitBuilder(self.nbqbits)
        bdr.alldiam()

        bdr.allin(x[[0,1]])
        bdr.cc(0, 1)

        bdr.allin(params[[0,1]])
        bdr.cc(0, 1)

        bdr.allin(x[[2,3]])
        bdr.cc(0, 1)

        bdr.allin(params[[2,3]])
        bdr.cc(0, 1)

        bdr.allin(x[[0,1]])
        bdr.cc(0, 1)

        bdr.allin(params[[4,5]])
        bdr.cc(0, 1)

        bdr.allin(x[[2,3]])
        bdr.cc(0, 1)

        bdr.allin(params[[6,7]])

        if shots: bdr.measure_all()
        # print(qc.draw('text'))
        return bdr.circuit()

    def random_params(self, seed=None):
        if seed: np.random.seed(seed)
        return np.random.randn(8)

    def __repr__(self):
        return "alldiam() IN(0,x_0)IN(1,x_1) CC(0,1) IN(0,t_0)IN(1,t_1) CC(0,1) IN(0,x_2)IN(1,x_3) CC(0,1) IN(0,t_2)IN(1,t_3) CC(0,1) IN(0,x_0)IN(1,x_1) CC(0,1) IN(0,t_4)IN(1,t_5) CC(0,1) IN(0,x_2)IN(1,x_3) CC(0,1) IN(0,t_6)IN(1,t_7) allmeter() "


class irisCircuit16(circuitML):
    """Two qubit circuit for four-dimensional input, with 16 parameters.
    """
    def __init__(self, backend, circuitBuilder, noise_model=None, noise_backend=None,
                 save_path="jobs.json"):
        super().__init__(backend, circuitBuilder, nbqbits=2,
                         noise_model=noise_model, noise_backend=noise_backend, save_path=save_path)
    
    def make_circuit(self, x, params, shots=None):
        bdr = self.circuitBuilder(self.nbqbits)
        bdr.alldiam()

        # ---------- ALLDIN(x[:2], t[:2])
        bdr.allin(x[:2])
        bdr.allin(params[[0,1]])

        bdr.cc(0, 1)

        # ---------- ALLDIN(x[2,3], t[2,3])
        bdr.allin(x[[2,3]])
        bdr.allin(params[[2,3]])

        bdr.cc(0, 1)

        bdr.allin(params[[4,5]])

        bdr.cc(0, 1)

        # ---------- ALLDIN(x[:2], t[6,7])
        bdr.allin(x[[0,1]])
        bdr.allin(params[[6,7]])

        bdr.cc(0, 1)

        # ---------- ALLDIN(x[2,3], t[8,9])
        bdr.allin(x[[2,3]])
        bdr.allin(params[[8,9]])

        bdr.cc(0, 1)

        bdr.allin(params[[10,11]])

        bdr.cc(0, 1)

        # ---------- ALLDIN(x[:2], t[12,13])
        bdr.allin(x[[0,1]])
        bdr.allin(params[[12,13]])

        bdr.cc(0, 1)

        # ---------- ALLDIN(x[2,3], t[14,15])
        bdr.allin(x[[2,3]])
        bdr.allin(params[[14,15]])

        if shots: bdr.measure_all()
        # print(qc.draw('text'))
        return bdr.circuit()

    def random_params(self, seed=None):
        if seed: np.random.seed(seed)
        return np.random.randn(16)

    def __repr__(self):
        return " alldiam() DIN(0,x_0,t_0)DIN(1,x_1,t_1) CC(0,1) DIN(0,x_2,t_2)DIN(1,x_3,t_3) CC(0,1) IN(0,t_4)IN(1,t_5) CC(0,1) DIN(0,x_0,t_6)DIN(1,x_1,t_7) CC(0,1) DIN(0,x_2,t_8)DIN(1,x_3,t_9) CC(0,1) IN(0,t_10)IN(1,t_11) CC(0,1) DIN(0,x_0,t_12)DIN(1,x_1,t_13) CC(0,1) DIN(0,x_2,t_14)DIN(1,x_3,t_15) allmeter() "


class irisCircuit6(circuitML):
    """Two qubit circuit for four-dimensional input, with six parameters.
    """
    def __init__(self, backend, circuitBuilder, noise_model=None, noise_backend=None,
                 save_path="jobs.json"):
        super().__init__(backend, circuitBuilder, nbqbits=2,
                         noise_model=noise_model, noise_backend=noise_backend, save_path=save_path)

    def make_circuit(self, x, params, shots=None):
        bdr = self.circuitBuilder(self.nbqbits)

        bdr.alldiam()
        # ---------- ALLDIN(x[:2], t[:2])
        bdr.allin(x[:2])
        bdr.allin(params[[0,1]])

        bdr.cc(0, 1)

        # ---------- ALLDIN(x[2,3], t[2,3])
        bdr.allin(x[[2,3]])
        bdr.allin(params[[2,3]])

        bdr.cc(0, 1)

        bdr.allin(params[[4,5]])

        if shots: bdr.measure_all()
        # print(qc.draw('text'))
        return bdr.circuit()
    
    def random_params(self, seed=None):
        if seed: np.random.seed(seed)
        return np.random.randn(6)

    def __repr__(self):
        return "alldiam()DIN(0,x_0,t_0)DIN(1,x_1,t_1)CC(0,1)DIN(0,x_2,t_2)DIN(1,x_3,t_3)CC(0,1)IN(0,t_4)IN(1,t_5)allmeter() "

############################################################
# PARALLEL MODELS

class irisParallel(parallelML):
    """Two qubit circuit for four-dimensional input, with 8 parameters; parallelized for 5-qubits hardware.
    """
    def __init__(self, backend, circuitBuilder, noise_model=None, noise_backend=None,
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
            Where to save the jobs outputs, by default "jobs.json". Jobs are saved only if so is specified when calling `circuitML.run`.

        Raises
        ------
        ValueError
            If both `noise_model` and `noise_backend` are provided.
        """
        super().__init__(backend, circuitBuilder, nbqbits=5,
                         noise_model=noise_model, noise_backend=noise_backend,
                         save_path=save_path)
    
    def make_circuit(self, x, params, shots=None):
        _x = np.zeros((2, x.shape[1]))
        _x[:len(x)] = x

        qc0 = [0,1]
        qc1 = [3,4]

        bdr = self.circuitBuilder(self.nbqbits)
        bdr.alldiam(qc0+qc1)

        bdr.input(qc0, _x[0, :2])
        bdr.input(qc1, _x[1, :2])
        bdr.cc(*qc0)
        bdr.cc(*qc1)

        bdr.input(qc0, params[[0,1]])
        bdr.input(qc1, params[[0,1]])
        bdr.cc(*qc0)
        bdr.cc(*qc1)

        bdr.input(qc0, _x[0, 2:])
        bdr.input(qc1, _x[1, 2:])
        bdr.cc(*qc0)
        bdr.cc(*qc1)

        bdr.input(qc0, params[[2,3]])
        bdr.input(qc1, params[[2,3]])
        bdr.cc(*qc0)
        bdr.cc(*qc1)

        bdr.input(qc0, _x[0, :2])
        bdr.input(qc1, _x[1, :2])
        bdr.cc(*qc0)
        bdr.cc(*qc1)

        bdr.input(qc0, params[[4,5]])
        bdr.input(qc1, params[[4,5]])
        bdr.cc(*qc0)
        bdr.cc(*qc1)

        bdr.input(qc0, _x[0, 2:])
        bdr.input(qc1, _x[1, 2:])
        bdr.cc(*qc0)
        bdr.cc(*qc1)

        bdr.input(qc0, params[[6,7]])
        bdr.input(qc1, params[[6,7]])

        if shots: bdr.measure_all()
        # print(qc.draw('text'))
        return bdr.circuit()

    def random_params(self, seed=None):
        if seed: np.random.seed(seed)
        return np.random.randn(8) 

    def __repr__(self):
        return "alldiam() IN(0,x_0)IN(1,x_1) CC(0,1) IN(0,t_0)IN(1,t_1) CC(0,1) IN(0,x_2)IN(1,x_3) CC(0,1) IN(0,t_2)IN(1,t_3) CC(0,1) IN(0,x_0)IN(1,x_1) CC(0,1) IN(0,t_4)IN(1,t_5) CC(0,1) IN(0,x_2)IN(1,x_3) CC(0,1) IN(0,t_6)IN(1,t_7) allmeter() "