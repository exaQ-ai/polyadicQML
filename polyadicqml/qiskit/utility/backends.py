from qiskit import Aer, IBMQ, QiskitError
from qiskit.providers.aer.noise import NoiseModel

from time import asctime, sleep
from sys import exc_info

AER_SIMULATORS = {"qasm_simulator", "statevector_simulator", "unitary_simulator"}

class Backends():
    """Utility class to load quiskit backends and iterate through them.
    """
    def __init__(self, backend_name, noise_name=None,
                 hub="ibm-q", group="open", project="main",
                 repeat=1):
        """Create instance.

        Parameters
        ----------
        backend_name : Union[str, list[str]]
            Name of the desired backend[s]
        noise_name : Union[str, list[str]], optional
            Name of the backend[s] to use as noise model, by default None. Only used if simulator is True
        simulator : bool, optional
            Whether the backend is a simulator, by default False
        hub : str, optional
            Name of the IBMQ hub, by default None
        group : str, optional
            Name of the IBMQ group, by default None
        project : str, optional
            Name of the IBMQ project, by default None
        repeat : int, optional
            How many time to repeat each backend during iteration, by default 1
        """
        super().__init__()
        self.__names__ = backend_name if isinstance(backend_name, list) else [backend_name]
        self.__noise_names__ = noise_name if isinstance(noise_name, list) else [noise_name] if noise_name is not None else None
        self.__repeat__ = repeat
        self.__simulator__ = False
        if set(self.__names__) & AER_SIMULATORS:
            if set(self.__names__) & AER_SIMULATORS:
                self.__simulator__ = True
            else:
                raise ValueError("Request for both Aer and IBMQ backends")

        self.__hub__ =  hub
        self.__group__ =  group
        self.__project__ =  project

        self.__logged__ = False

        self.load_beckends()

    def load_beckends(self):
        """Load the desired backends and, if not yet logged, log in.
        """
        backends = []

        noise_models = [] #cycle(noise_model) if isinstance(noise_model, list) else cycle([noise_model])
        coupling_maps = [] #cycle([None])

        if self.__simulator__:
            for name in self.__names__:
                backends.append(Aer.get_backend(name))
        
        if not self.__simulator__ or self.__noise_names__ is not None:
            provider = None
            while(not self.__logged__):
                try:
                    IBMQ.load_account()
                    provider = IBMQ.get_provider(hub=self.__hub__,
                                                group=self.__group__,
                                                project=self.__project__)
                    self.__logged__ = True
                except QiskitError:
                    with open("error.log", "w") as f:
                        f.write(f"{asctime()} - Error logging : {exc_info()[0]}\n")
                    sleep(10)
                    continue
            
            if not self.__simulator__:
                for name in self.__names__:
                    back = provider.get_backend(name)
                    for _ in range(self.__repeat__):
                        backends.append(back)
            else:
                for name in self.__noise_names__:
                    back = provider.get_backend(name)
                    for _ in range(self.__repeat__):
                        noise_models.append(NoiseModel.from_backend(back))
                        coupling_maps.append(back.configuration().coupling_map)
        
        self.backends = cycler(backends)
        self.noise_models = cycler(noise_models)
        self.coupling_maps = cycler(coupling_maps)

class cycler():
    """Utility class to cycle over a list.
    """
    def __init__(self, l):
        """Create cycler

        Parameters
        ----------
        l : list
            List to cycle on.
        """
        super().__init__()
        self.__list__ = l
        self.__i__ = 0
    
    def __next__(self):
        if len(self.__list__) == 0 : return None

        i = self.__i__
        self.__i__ = (i + 1) % len(self.__list__)
        return self.__list__[i]
