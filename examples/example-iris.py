from os.path import isdir

if not isdir("figure-ex"):
    from os import mkdir
    mkdir("figure-ex")
#############################################

import numpy as np

from examples.irisPreparation import makeDatasets

from polyadicqml import Classifier

from polyadicqml.qiskit.utility import Backends
from polyadicqml.qiskit import qkCircuitML

from polyadicqml.manyq import mqCircuitML

##############################
# We load the datatset

input_train, target_train, input_test, target_test = makeDatasets(.6, .4, seed=15201889)

##############################
# We define a circuit

def block(bdr, x, p):
    bdr.allin(x[[0,1]])
    bdr.cz(0,1).allin(p[[0,1]])

    bdr.cz(0,1).allin(x[[2,3]])
    bdr.cz(0,1).allin(p[[2,3]])

def irisCircuit(bdr, x, params):
    # The fist block uses all `x`, but
    # only the first 4 elements of `params`
    block(bdr, x, params[:4])

    # Add one entanglement not to have two adjacent input
    bdr.cz(0,1)
    
    # The block repeats with the other parameters
    block(bdr, x, params[4:])

    return bdr

##############################
# We instanciate and train the classifier

nbqbits = 2
nbparams = 8

qc = mqCircuitML(make_circuit=irisCircuit,
                 nbqbits=nbqbits, nbparams=nbparams)

bitstr = ['00', '01', '10']

model = Classifier(qc, bitstr).fit(input_train, target_train, method="BFGS")

##############################
# We test it using qiskit

backend = Backends("ibmq_ourense", hub="ibm-q")

qc = qkCircuitML(
    make_circuit=irisCircuit,
    nbqbits=nbqbits, nbparams=nbparams,
    backend=backend,
)

model.set_circuit(qc)
model.nbshots = 300
model.job_size = 30

pred_train = model(input_train)
pred_test = model(input_test)

##############################
# We print the results

from sklearn.metrics import confusion_matrix, accuracy_score

def print_results(target, pred, name="target"):
    print('\n' + 30*'#',
        "Confusion matrix on {}:".format(name), confusion_matrix(target, pred),
        "Accuracy : " + str(accuracy_score(target, pred)),
        sep='\n')

print_results(target_train, pred_train, name="train")
print_results(target_test, pred_test, name="test")