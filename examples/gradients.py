from os.path import isdir, mkdir

if not isdir("figure-ex"):
    mkdir("figure-ex")
#############################################

from polyadicqml.manyq import mqCircuitML
from polyadicqml.qiskit import qkCircuitML
from polyadicqml.qiskit.utility import Backends
import numpy as np

import pickle

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# np.random.seed(13)

def simple_circuit(bdr, x, params):
    bdr.allin(x).cz(0,1).allin(params[:2])
    bdr.cz(0,1).allin(params[2:4])

    return bdr

back = Backends("qasm_simulator")

nbparams=4
circuit = qkCircuitML(simple_circuit, 2, nbparams, back)
circuit2 = mqCircuitML(simple_circuit, 2, nbparams)

x = np.random.rand(2)

params = np.random.rand(nbparams)
p2 = params.copy()

nbshots= int(1e3)

eps = np.pi/nbshots**.3
p2[0] += eps

print("Out : \n")
print("Exact :")
R = circuit2.run(x, params)
L = circuit2.run(x, p2)
print(R, L, sep='\n')

print("Shots :")
R = circuit2.run(x, params, nbshots)/nbshots
L = circuit2.run(x, p2, nbshots=nbshots)/nbshots
print(R, L, sep='\n')

G = circuit2.grad(x, params)

if True:
    nbshotss = np.logspace(1,5,15).astype(int)

    b_diffs = []
    b_alphas = []

    for nbshots in tqdm(nbshotss):
        diffs = []
        alphas = np.linspace(0, 1, 20)

        for alpha in tqdm(alphas, leave=False):
            G_fd = circuit.grad(x, params, eps=np.pi/nbshots**eps, nbshots=nbshots)

            diffs.append(np.linalg.norm(G - G_fd))
        
        b_idx = np.argmin(diffs)
        b_alphas.append(alphas[b_idx])
        b_diffs.append(diffs[b_idx])

    fig, ax = plt.subplots(figsize=(8,5))

    l1, *_ = ax.plot(nbshotss, b_alphas, color="steelblue", marker="D")
    ax.set(xscale="log", ylabel="alpha")

    ax2 = ax.twinx()
    l2, *_ = ax2.plot(nbshotss, b_diffs, color="seagreen")
    ax2.grid(False)
    ax2.set(ylabel="grad error")

    ax.legend((l1, l2), ("Best alpha", "Best diff"), loc="best")

    plt.savefig("figures-ex/gradients.png")
    plt.close()

    d = dict(
        alphas = b_alphas,
        diffs = b_diffs
    )

    with open("exp/data/grads.pkl", "wb") as f:
        pickle.dump(d, f)

