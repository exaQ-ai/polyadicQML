.. include:: ../substs.rst

.. _sec-manyq:

The *manyq* simulator
=====================

We discuss the |manyq| module.
This module wraps the *manyq* simulator, which is conceived as a quantum
simulator for machine learning.
In fact, as its name suggets, it parallelizes computations, based on the
SIMD principle -- i.e. Single Instruction Multiple Data.

The idea is that, given an architecture, all parametric circuits are very
similar, besides some gates in which we change the parameters.
Therefore, manyq relies on tensor contraction to manipulate multiple
circuits as tensors and, so doing, provides a great speed for machine
learning tasks.

The |manyq| module provides a specific circuitML, namely |mqCircuitML|,
as well as a specific circuitBuilder, |mqBuilder|.

GPU support
+++++++++++

.. _CuPy: https://cupy.chainer.org/

The only difference from the basic |circuitML| interface is the support
for gpu, which is provided through CuPy_.
It can be required at instanciation specifying the ``gpu`` keyword
argument:

.. code-block::
    :emphasize-lines: 3

    circuit = mqCircuitML(
        make_circuit, nbqbits, nbparams,
        gpu = True
    )

Alternatively, one can change the backend of a circuit at runtime, using
the |cpu| and |gpu| methods::

    circuit.cpu()   # Switch to NumPy

    circuit.gpu()   # Switch to CuPy

.. note::

    If the circuit is running on cpu, the type of the inputs of |run|,
    and of its return value as well, is :class:`cupy.ndarray`, while it
    is :class:`numpy.ndarray` in the cpu setting.

    For this reason, as of version |version|, the |Classifier| does not
    support gpu.
