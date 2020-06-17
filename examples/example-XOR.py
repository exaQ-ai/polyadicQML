import numpy as np
import matplotlib.pyplot as plt

from polyadicqml.quantumClassifier import Classifier
from polyadicqml.qiskit.utility.backends import Backends

from polyadicqml.qiskit.qkCircuitML import qkCircuitML

from polyadicqml.manyq.mqCircuitML import mqCircuitML

##############################
# We create a dataset of 200 points corresponding to the XOR problem
#       1 --|-- 0
#       |   |   |
#      -----|------
#       |   |   |
#       0 --|-- 1

n_pc = 50 # Number of points per cluster

# Create a matrix of vertices of the centered square
X = np.asarray(n_pc * [[1. ,1.]] +    # First quadrant
               n_pc * [[-1. ,-1.]] +  # Third quadrant
               n_pc * [[1. ,-1.]] +   # Second quadrant
               n_pc * [[-1. ,1.]]     # Fourth quadrant
)
X *= 1.2
# Add gaussian noise
X += .5 * np.random.randn(*X.shape)
# Rotate of pi/4
X = X @ [[np.cos(np.pi/4), np.sin(np.pi/4)],
         [-np.sin(np.pi/4), np.cos(np.pi/4)]]

# Create target vecor
y = np.concatenate((np.zeros(2*n_pc), np.ones(2*n_pc)))

# sns.scatterplot(X[:,0], X[:,1], hue=y)
# plt.show()

##############################
# Now we define the make_circuit function using the builder interface

def make_circuit(circuitml, x, params, shots=None):
    job_size = 1 if len(x.shape) < 2 else x.shape[1]
    bdr = circuitml.circuitBuilder(circuitml.nbqbits, job_size=job_size)
    bdr.alldiam()
    
    bdr.allin(x[[0,1]])
    bdr.cc(0, 1)

    bdr.allin(params[[0,1]])
    bdr.cc(0, 1)

    bdr.allin(params[[2,3]])
    bdr.cc(0, 1)

    if shots: bdr.measure_all()
    return bdr.circuit()

def random_params(nbparams, seed=None):
    if seed: np.random.seed(seed)
    # return np.pi * (np.random.rand(self.n_params) - .5)
    return 1 / np.sqrt() * np.random.randn(nbparams)

##############################
# Now we instanciate a backend and the circuit

nbqbits = 2
nbparams = 4

backend = Backends("qasm_simulator", simulator=True)
qc = qkCircuitML(backend, 
                 make_circuit=make_circuit,
                 nbqbits=nbqbits, nbparams=nbparams)

qc = mqCircuitML(make_circuit=make_circuit,
                 nbqbits=nbqbits, nbparams=nbparams)
# qc = myCircuit(circuitBuilder=ibmqNativeBuilder, backend=backend)
# print(qc.make_circuit(X[0], qc.random_params()).draw('text'))

bitstr = ['00', '01']
nbshots = None

params = qc.random_params()

model = Classifier(qc, bitstr, nbshots=nbshots, budget=100)

model.fit(X, y, method="BFGS")

##############################
# Then we test the model

t = np.linspace(-np.pi,np.pi, num = 50)
X_test = np.array([[t1, t2] for t1 in t for t2 in t])

y_pred = model.predict_label(X_test)

if True:
    fig, ax = plt.subplots(figsize=(8,8))
    idx = y_pred == 1
    ax.plot(X_test[idx,0], X_test[idx,1], ls="", marker="s", color="coral", alpha=.3)
    ax.plot(X_test[~ idx,0], X_test[~ idx,1], ls="", marker="s", color="deepskyblue", alpha=.3)

    idx = y == 1
    ax.plot(X[idx,0], X[idx,1], ls="", marker="o", color="tab:red",)
    ax.plot(X[~ idx,0], X[~ idx,1], ls="", marker="o", color="tab:blue",)
    ax.plot([-np.pi, np.pi], [-np.pi, np.pi], color="black")
    ax.plot([-np.pi, np.pi], [np.pi, -np.pi], color="black")

    plt.savefig("figure.png", bbox_inches="tight")