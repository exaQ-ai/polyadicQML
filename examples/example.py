import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()

from polyadicqml.qiskit.qkCircuitML import qkCircuitML
from polyadicqml.quantumClassifier import Classifier
from polyadicqml.qiskit.utility.backends import Backends
from polyadicqml.circuits.qiskitBdr import ibmqNativeBuilder

##############################
# We create a dataset of 200 points corresponding to the XOR problem
#       1 --|-- 0
#       |   |   |
#      -----|------
#       |   |   |
#       0 --|-- 1

n_pc = 50 # Number of points per cluster

# Create a matrix of vertices of the centered square
X = np.asarray(n_pc * [[.5 ,.5]] +    # First quadrant
               n_pc * [[-.5 ,-.5]] +  # Third quadrant
               n_pc * [[.5 ,-.5]] +   # Second quadrant
               n_pc * [[-.5 ,.5]]     # Fourth quadrant
)
# Add gaussian noise
X += .1 * np.random.randn(*X.shape)

# Create target vecor
y = np.concatenate((np.zeros(2*n_pc), np.ones(2*n_pc)))

# sns.scatterplot(X[:,0], X[:,1], hue=y)
# plt.show()

##############################
# Now we define a circuit as a subclass of circuitML

class myCircuit(qkCircuitML):
    def __init__(self, backend, circuitBuilder):
        super().__init__(backend, circuitBuilder, nbqbits=2)
        # NOTE that we fixed the number of qubits
    
    def make_circuit(self, x, params, shots=None):
        bdr = self.circuitBuilder(self.nbqbits)
        bdr.alldiam()

        bdr.allin(x[[0,1]])
        bdr.cc(0, 1)

        bdr.allin(params[[0,1]])
        bdr.cc(0, 1)

        if shots: bdr.measure_all()
        return bdr.circuit()

    def random_params(self, seed=None):
        if seed: np.random.seed(seed)
        return np.random.randn(2)

##############################
# Now we instanciate a backend and the circuit

backend = Backends("qasm_simulator", simulator=True)
qc = myCircuit(backend, ibmqNativeBuilder)
# print(qc.make_circuit(X[0], qc.random_params()).draw('text'))

bitstr = ['00', '11']
nbshots = 300

params = qc.random_params()

model = Classifier(qc, bitstr, nbshots=nbshots, budget=200)

model.fit(X, y, method="COBYLA")

t = np.linspace(-1,1, num = 10)
X_test = np.array([[t1, t2] for t1 in t for t2 in t])

y_pred = model.predict_label(X_test)

print(y_pred.reshape((10, -1)))