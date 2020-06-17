from numpy import zeros

############################################################
# MODELS

def irisCircuit(circuitml, x, params, shots=None):
    bdr = circuitml.circuitBuilder(circuitml.nbqbits)
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

def irisCircuit16(circuitml, x, params, shots=None):
    bdr = circuitml.circuitBuilder(circuitml.nbqbits)
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

def irisCircuit6(circuitml, x, params, shots=None):
    bdr = circuitml.circuitBuilder(circuitml.nbqbits)

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

############################################################
# PARALLEL MODELS

def irisParallel(circuitml, x, params, shots=None):
    _x = zeros((2, x.shape[1]))
    _x[:len(x)] = x

    qc0 = [0,1]
    qc1 = [3,4]

    bdr = circuitml.circuitBuilder(circuitml.nbqbits)
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
