import numpy as np
import matplotlib.pyplot as plt

from polyadicqml import Classifier

from polyadicqml.manyq import mqCircuitML

FIGURES = True

SEED = 294
np.random.seed(SEED)

##############################
# We create a dataset of 40 points from two gaussian clouds.

# We take a random polar point
a = 2 * np.pi * np.random.rand()

n_pc = 50 # Number of points per cluster

scale = 1.25
px, py = scale * np.cos(a), scale * np.sin(a)

# Create a matrix on the point and its symmetry and spread apart
X = np.asarray(n_pc * [[px, py]] +    # Polar point
               n_pc * [[-px, -py]]    # Symmetric point
)
# Add gaussian noise
X += 0.7 *  np.random.randn(*X.shape)

# Create target vecor
y = np.concatenate((np.zeros(n_pc), np.ones(n_pc)))

COLORS = ["tab:blue", "tab:red"]
if FIGURES:
    import seaborn as sns
    sns.set()
    fig, ax = plt.subplots(figsize=(5,5))
    idx = y == 1
    ax.plot(X[~ idx,0], X[~ idx,1], ls="", marker="o", color=COLORS[0], label="Class 0")
    ax.plot(X[idx,0], X[idx,1], ls="", marker="o", color=COLORS[1], label="Class 1",)

    graph_args = dict(ls="", marker = "D", ms=10, mec="black", mew=2)

    ax.plot([px], [py], color=COLORS[0], **graph_args)
    ax.plot([-px], [-py], color=COLORS[1], **graph_args)

    ax.set(xlim=[-np.pi,np.pi], ylim=[-np.pi,np.pi])
    ax.legend(loc="upper left")

    plt.savefig("figures/binary-points.svg", bbox_inches="tight")
    plt.close()

##############################
# Now we define the make_circuit function using the builder interface

def simple_circuit(bdr, x, params):
    bdr.allin(x).cz(0,1).allin(params[:2])
    bdr.cz(0,1).allin(params[2:4])
    return bdr

##############################
# Now we instanciate the circuit

nbqbits = 2
nbparams = 4

qc = mqCircuitML(make_circuit=simple_circuit,
                nbqbits=nbqbits, nbparams=nbparams)

bitstr = ['00', '11']

# We can use exact probabilities
model = Classifier(qc, bitstr)
model.fit(X, y, save_loss_progress=True)

# Or pass though shots-based estimates
model2 = Classifier(qc, bitstr, nbshots=300)
model2.fit(X, y, method='COBYLA', save_loss_progress=True)

if FIGURES:
    fig, ax = plt.subplots(figsize=(5,3))
    fig.set_tight_layout(True)

    p0 = model2.__loss_progress__[0]
    l2, = ax.plot(model2.__loss_progress__, c='tab:green')
    l1, = ax.plot([p0] + model.__loss_progress__, c="tab:blue")

    ax.set(ylabel="loss value", xlabel="iteration",)

    ax.legend((l1, l2), ('BFGS - simulated QPU', 'COBYLA - shots'),
            loc='upper right')

    plt.savefig("figures/binary-loss-progress.svg", bbox_inches='tight')
    plt.close()

##############################
# Then we test the model

t = np.linspace(-np.pi,np.pi, num = 50)
X_test = np.array([[t1, t2] for t1 in t for t2 in t])

y_pred = model.predict(X_test)

if FIGURES:
    fig, ax = plt.subplots(figsize=(5,5))
    idx = y_pred == 1
    ax.plot(X_test[idx,0], X_test[idx,1], ls="", marker="s", color="coral", alpha=.3)
    ax.plot(X_test[~ idx,0], X_test[~ idx,1], ls="", marker="s", color="deepskyblue", alpha=.3)

    idx = y == 1
    ax.plot(X[~ idx,0], X[~ idx,1], ls="", marker="o", color=COLORS[0], label="Class 0")
    ax.plot(X[idx,0], X[idx,1], ls="", marker="o", color=COLORS[1], label="Class 1")

    graph_args = dict(ls="", marker = "D", ms=10, mec="black", mew=2)

    ax.plot([px], [py], color=COLORS[0], **graph_args)
    ax.plot([-px], [-py], color=COLORS[1], **graph_args)

    ax.plot([5*py, -5*py], [-5*px, 5*px], color="tab:grey")
    ax.set(xlim=[-np.pi,np.pi], ylim=[-np.pi,np.pi],)
    ax.legend(loc="upper left")

    plt.savefig("figures/binary-predictions.svg", bbox_inches="tight")
    plt.close()

#############################################
# We compute the full circuit output on the train set.

model.nbshots = 300
full_output = model.run_circuit(X)

if FIGURES:
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    fig.set_tight_layout(True)

    bitstr = ['00', '10', '01', '11']
    x = [0, 1, 2, 3]
    names= ["Class 0", "Class 1"]
    colors = ["tab:orange", "tab:green", "tab:blue", "tab:red"]

    for label in range(2):
        count = full_output[y==label]

        for c in count:
            ax[label].bar(x, height=c, alpha=.07, color=colors)

        ax[label].boxplot(
            count, positions=x, sym="_",
            medianprops=dict(ls="-", lw=3, color="black"),
        )

        ax[label].set_title(f"{names[label]}", fontdict={'color': colors[-label]})
        ax[label].set(xticks=x, xticklabels=bitstr, ylim=(0,300), xlim=(-.5,3.5))
    ax[1].tick_params(labelleft=False, labelright=True)

    plt.savefig("figures/binary-counts.svg", bbox_inches="tight")
    plt.close()