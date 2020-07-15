from os.path import isdir, mkdir

if not isdir("figures-ex"):
    mkdir("figures-ex")
#############################################

from polyadicqml.manyq import mqCircuitML
import numpy as np

# `make_circuit` can have any name
def simple_circuit(bdr, x, params):
    return bdr.allin(x).cz(0,1).allin(params)

# We instantiate the circuitML
from polyadicqml.manyq import mqCircuitML

circuit = mqCircuitML(simple_circuit, nbqbits=2, nbparams=2)

# We create an input matrix and a param vector
import numpy as np

X = np.array([[-np.pi/4, np.pi/4],
            [np.pi/4, -np.pi/4]])
params = np.array([np.pi/2, -np.pi/2])

probs = circuit.run(X, params)
counts = circuit.run(X, params, nbshots=100)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def figure(P, name):
    fig, ax = plt.subplots(1,2, figsize=(6,3))

    bitstr = ['00', '10', '01', '11']
    x = [0, 1, 2, 3]
    names= ["$x_1 = [-\pi/4, \pi/4]$", "$x_2 = [\pi/4, -\pi/4]$"]
    colors = ["tab:orange", "tab:green", "tab:blue", "tab:red"]

    for sample in range(2):
        ax[sample].bar(x, height=P[sample], alpha=1, color=colors)

        ax[sample].set_title(f"{names[sample]}") 
        ax[sample].set(xticks=x, xticklabels=bitstr, ylim=(0,sum(P[sample])), xlim=(-.5,3.5))

        for i, v in enumerate(P[sample]):
            ax[sample].text(
                x[i], v + 0.05 * sum(P[sample]),
                f"{int(v)}" if v >=1 or v==0 else f"{v:.2f}",
                color=colors[i], fontweight='bold',
                ha='center', va="bottom"
            )
    ax[1].tick_params(labelleft=False, labelright=True)
    
    plt.savefig(f"figures-ex/circuit-run-{name}.svg", bbox_inches="tight")
    plt.close(fig)

figure(probs, "probs")
figure(counts, "counts")
