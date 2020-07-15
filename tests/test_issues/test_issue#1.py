if False:
    import numpy as np
    from polyadicqml.manyq import mqCircuitML
    from polyadicqml.qiskit import qkBuilder
    from polyadicqml import Classifier

    nbqbits = 2
    nbparams = 22
    bitstr = ['00', '01', '10']

    def block(bdr, x, p):
        
        bdr.allin(x[:2])
        bdr.cz(0,1).allin(p[:2])
        
        bdr.cz(0,1).allin(x[[2,3]])
        bdr.cz(0,1).allin(p[[2,3]])
        
        bdr.cz(0,1).allin(x[[4,5]])
        bdr.cz(0,1).allin(p[[4,5]])
        
        bdr.cz(0,1).allin(x[[6,7]])
        bdr.cz(0,1).allin(p[[6,7]])
        
        bdr.cz(0,1).allin(x[[8,9]])
        bdr.cz(0,1).allin(p[[8,9]])
        
        bdr.cz(0,1).input(0, x[10])
        bdr.cz(0,1).input(0, p[10])
        
    def wineCircuit(bdr, x, params):

        block(bdr, x, params[:11])
        bdr.cz(0,1)
        block(bdr, x, params[11:])
        
        return bdr

    input_train = np.random.rand(2940,11)
    target_train = (3*np.random.rand((2940))).astype(int)

    qc = mqCircuitML(make_circuit=wineCircuit,
                    nbqbits=nbqbits, nbparams=nbparams)

    model = Classifier(qc, bitstr)

    model.fit(input_train, target_train)
