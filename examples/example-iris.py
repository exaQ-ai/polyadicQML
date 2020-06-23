import numpy as np

from examples.irisPreparation import makeDatasets

from polyadicqml.quantumClassifier import Classifier

from polyadicqml.qiskit.utility.backends import Backends
from polyadicqml.qiskit.qkCircuitML import qkCircuitML

from polyadicqml.manyq.mqCircuitML import mqCircuitML

##############################
# We load the datatset

input_train, target_train, input_test, target_test = makeDatasets(.6, .4, seed=15201889)

##############################
# We define a circuit

def irisCircuit(bdr, x, params, shots=None):

    bdr.allin(x[[0,1]])
    bdr.cz(0, 1)

    bdr.allin(params[[0,1]])
    bdr.cz(0, 1)

    bdr.allin(x[[2,3]])
    bdr.cz(0, 1)

    bdr.allin(params[[2,3]])
    bdr.cz(0, 1)

    bdr.allin(x[[0,1]])
    bdr.cz(0, 1)

    bdr.allin(params[[4,5]])
    bdr.cz(0, 1)

    bdr.allin(x[[2,3]])
    bdr.cz(0, 1)

    bdr.allin(params[[6,7]])

    if shots: bdr.measure_all()

    return bdr.circuit()

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

backend = Backends("ibmq_burlington")

qc = qkCircuitML(backend=backend,
                 make_circuit=irisCircuit,
                 nbqbits=nbqbits, nbparams=nbparams)

model.set_circuit(qc)
model.nbshots = 300
model.job_size = 30

pred_train = model.predict_label(input_train)
pred_test = model.predict_label(input_test)

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