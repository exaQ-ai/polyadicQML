.. include:: ../substs.rst

.. _sec-qiskit:

The Qiskit/IBMQ interface
=========================

.. _Qiskit: https://qiskit.org/
.. _`IBMQ`: https://www.ibm.com/quantum-computing/

In this tutorial, we discuss the |qk| module, which provides support for
Qiskit_ simulators as well as `IBMQ`_ backends. 
It implements a specific circuitML, namely |qkCircuitML|, as
well as the corresponding circuitBuilder.

Furthermore, the :mod:`polyadicqml.qiskit.utility` module provides the
|back| class, which wraps the Qiskit/IBMQ backend interface and add a
few functionalities.

.. note::

    To access IBM Quantum systems, you need to configure your IBM Quantum account.
    Detailed instructions are provided on the `Qiskit installation guide`_.
    You can verify your setup if the following runs without producing errors::

        >>> from qiskit import IBMQ
        >>> IBMQ.load_account()

    If you do not have an IBM Quantum account, you can still use |qk_aer|_.

Using |qkCircuitML|
+++++++++++++++++++

.. _`Aer API`: https://qiskit.org/aer

Qiskit provides various backends on which to run circuits.
Some are locally-run simulators, provided by the `Aer API`_, while others
are cloud-based IBM Quantum systems, or simulators.

To run a |qkCircuitML|, we need to provide such a backend to its constructor.
This can be done in two ways: by manually loading a backend from
:mod:`qiskit`; or by using a |back| instance.
In this section we describe the first method and we explain the second in
:numref:`ssec-back`.

Basic simulation
----------------

Suppose we want to simulate a quantum circuit, using a shots-compatible simulator.
We can retrieve the ``qasm_simulator`` with the following code::

    from qiskit import Aer, IBMQ
    backend = Aer.get_backend('qasm_simulator')

At this point, we can instantiate our |qkCircuitML|, by providing the
backend as the fourth positional argument (the first three being common
to all circuitML):

.. code-block::
    :emphasize-lines: 12

    from polyadicqml.qiskit import qkCircuitML

    def make_circuit(bdr, x, params):
        # We define the circuit operations
        ...

    # We instantiate the circuit
    circuit = qkCircuitML(
        make_circuit,
        nbqbits,
        nbparams,
        backend
    )

Simulating Noise
----------------

We may want to use a noise model for the quantum operations, when running our circuit.
To do so, we can provide a noise model to |qkCircuitML| using the
``noise_model`` keyword argument of the constructor.

For instance, let's retrieve the noise model and the coupling map from a
5-qubit real device, namely ``ibmq_ourense``, using the usual Qiskit syntax.
These two properties can be provided to |qkCircuitML| by the
corresponding keyword arguments:

.. code-block::
    :emphasize-lines: 15,16,17,18,19

    from qiskit import IBMQ
    from qiskit.providers.aer.noise import NoiseModel

    # Choose a real device to simulate
    IBMQ.load_account()
    provider = IBMQ.get_provider(group='open')
    device = provider.get_backend('ibmq_ourense')

    # Retrieve the device coupling map
    coupling_map = device.configuration().coupling_map
    # Generate an Aer noise model for device
    noise_model = NoiseModel.from_backend(device)


    circuit = qkCircuitML(
        ...
        noise_model = noise_model,
        coupling_map = coupling_map
    )

While this is the only way to use custom noise models and coupling maps,
when simulating a real device it is more convenient to just provide the
``noise_backend`` to the circuit constructor, which infers by itself the necessary properties.
The previous code thus becomes:

.. code-block::
    :emphasize-lines: 8,9,10,11

    from qiskit import IBMQ

    # Choose a real device to simulate
    IBMQ.load_account()
    provider = IBMQ.get_provider(group='open')
    device = provider.get_backend('ibmq_ourense')

    circuit = qkCircuitML(
        ...
        noise_backend = device
    )

Distributing across backends
----------------------------

One of the functionalities provided by |qkCircuitML| consists of
dispatching jobs over multiple backends at the same time.
This behavior can be obtained by providing to the class builder a list of
backends, instead of just one.
The |run| method will then handle the backends, by iterating over them
each time it dispatch a new job.

To leverage parallelism one should divide the input points across
different jobs, so that in a single |run| call they are processed by
multiple devices. 
This is easily done using the ``job_size`` keyword argument.

For instance, let's run a circuit on two real devices:

.. code-block::
    :emphasize-lines: 6,7,16,23

    from qiskit import IBMQ

    # Choose TWO real devices
    IBMQ.load_account()
    provider = IBMQ.get_provider(group='open')
    device1 = provider.get_backend('ibmq_ourense')
    device2 = provider.get_backend('ibmq_vigo')

    # Define the circuit operations
    def make_circuit(bdr, x, params):
        ...
    
    # instantiate the circuit
    circuit = qkCircuitML(
        make_circuit, nbqbits, nbparams
        backend = [device1, device2],
    )

    # Run it by precising a `job_size`
    circuit.run(
        X, params,
        nbshots = 300,    # Needed as the backend is a physical device
        job_size = 30     # We split X in jobs of 30 points/circuits
    )

In the same way, one can provide multiple noise models and coupling maps,
by passing lists to the corresponding keyword arguments.

As for the backend retrieval, multiple dispatching is supported, with a
shorter syntax, by the |back| class, which we introduce next.

.. _ssec-back:

The |back| class
++++++++++++++++

The |back| class provides a quick interface to load backends, which is
well integrated with |qkCircuitML|.

Base use
--------

To load an IBMQ device, or a *Qiskit Aer* simulator, we create a |back| instance
providing the desired name::

    from polyadicqml.qiskit.utility import Backends

    # Load IBMQ device
    backend = Backends("ibmq_ourense")

    # Simulators are automatically recognised
    backend_sim = Backends("statevector_simulator")

Then, we can directly provide it to a |qkCircuitML| though the
``backend`` argument::

    from polyadicqml.qiskit import qkCircuitML

    circuit = qkCircuitML(
        ...
        backend = backend
    )

Simulate noise
--------------

.. |qasm_s| replace:: ``qasm_simulator``
.. _qasm_s: https://qiskit.org/documentation/stubs/qiskit.providers.aer.QasmSimulator.html#qasmsimulator


To use the |qasm_s|_ and add noise from a real device, it is
enough to provide the corresponding name(s) as the second argument::

    # Simulate "ibmq_ourense" device
    backend = Backends("qasm_simulator", "ibmq_ourense")

Distributing across backends
----------------------------

As we already mentioned, |qkCircuitML| can dispatch jobs across multiple
backends.
This can be handled through a |back| instance, providing the list of
device we want to iterate on.

::

    # We precise multiple IBMQ devices
    backend = Backends(["ibmq_ourense", "ibmq_vigo"])

    # Or we can simulate them
    backend = Backends(
        "qasm_simulator",
        ["ibmq_ourense", "ibmq_vigo"]
    )

Specifying IBMQ provider
------------------------

One can precise the hub/group/project combination for the IBMQ provider using the corresponding keyword arguments::

    backend = Backends(
        ...
        hub="ibm-q",
        group="open",
        project="main"
    )

Parallel computation on a single device
+++++++++++++++++++++++++++++++++++++++

PolyadicQML provides the tools to run two circuits with the same
architecture, but different inputs, on parallel on a single device,
without changing our syntax.

For this purpose, we use the |qkParallelML| class, which comes with its
circuit builder |qkParallelBuilder|.
The syntax is almost the same as before, we describe a parametric circuit
in |make_c|, then we instantiate the |qkParallelML| and we can run it on
given design matrix and parameters.
The only difference is the ``tot_nbqbits`` argument in the constructor.
This specifies the total number of qubits in the device; not to be
confused with ``nbqbits``, the number of qubits used by a single circuit.

For instance, we can define a two-qubit circuit, and parallelize its
execution on one of the 5-qubit devices from IBMQ::

    import numpy as np

    from polyadicqml.qiskit import qkParallelML
    from polyadicqml.qiskit.utility import Backends

    # Define the circuit structure
    def make_circuit(bdr, x, params):
        bdr.allin(x[[0,1]])

        bdr.cz(0, 1)
        bdr.allin(params[[0,1]])

        bdr.cz(0, 1)
        bdr.allin(params[[2,3]])

        return bdr

    # Load a backend
    backend = Backends("ibmq_ourense", hub="ibm-q")

    # instantiate the circuit
    qc = qkParallelML(
        make_circuit=make_circuit,
        nbqbits=1, nbparams=4,
        backend=backend,
        tot_nbqbits=5, # We specify that the backend has 5 qubits
    )

    # Define the design matrix and parameters
    X = np.array([[1,2], [3,4]]
    params = np.array([.1,.2,.3,.4])

    # Run the ciruit, the two datapoints will be processed
    # at the same time
    qc.run(X, params, nbshots=300)

We can verify that it is in fact the builder which parallelize the execution on the QPU, by creating a single circuit made of two independetent parts, each of which corresponds to a different datapoint.
For instance, by running the following with the previous ``make_circuit``, we obtain::

    >>> from polyadicqml.qiskit import qkParallelBuilder

    >>> # Note that using the circuit builder we tranpose X
    >>> # as we want the first index to be that of the features
    >>> print(make_circuit(qkParallelBuilder(2, 5), X.T, params).measure_all().circuit())

          ┌──────────┐┌───────┐┌──────────┐   ┌──────────┐┌─────────┐┌──────────┐   ┌──────────┐┌─────────┐┌──────────┐ ░ ┌─┐
    qr_0: ┤ RX(pi/2) ├┤ RZ(1) ├┤ RX(pi/2) ├─■─┤ RX(pi/2) ├┤ RZ(0.1) ├┤ RX(pi/2) ├─■─┤ RX(pi/2) ├┤ RZ(0.3) ├┤ RX(pi/2) ├─░─┤M├────────────
          ├──────────┤├───────┤├──────────┤ │ ├──────────┤├─────────┤├──────────┤ │ ├──────────┤├─────────┤├──────────┤ ░ └╥┘┌─┐
    qr_1: ┤ RX(pi/2) ├┤ RZ(2) ├┤ RX(pi/2) ├─■─┤ RX(pi/2) ├┤ RZ(0.2) ├┤ RX(pi/2) ├─■─┤ RX(pi/2) ├┤ RZ(0.4) ├┤ RX(pi/2) ├─░──╫─┤M├─────────
          └──────────┘└───────┘└──────────┘   └──────────┘└─────────┘└──────────┘   └──────────┘└─────────┘└──────────┘ ░  ║ └╥┘┌─┐
    qr_2: ──────────────────────────────────────────────────────────────────────────────────────────────────────────────░──╫──╫─┤M├──────
          ┌──────────┐┌───────┐┌──────────┐   ┌──────────┐┌─────────┐┌──────────┐   ┌──────────┐┌─────────┐┌──────────┐ ░  ║  ║ └╥┘┌─┐
    qr_3: ┤ RX(pi/2) ├┤ RZ(3) ├┤ RX(pi/2) ├─■─┤ RX(pi/2) ├┤ RZ(0.1) ├┤ RX(pi/2) ├─■─┤ RX(pi/2) ├┤ RZ(0.3) ├┤ RX(pi/2) ├─░──╫──╫──╫─┤M├───
          ├──────────┤├───────┤├──────────┤ │ ├──────────┤├─────────┤├──────────┤ │ ├──────────┤├─────────┤├──────────┤ ░  ║  ║  ║ └╥┘┌─┐
    qr_4: ┤ RX(pi/2) ├┤ RZ(4) ├┤ RX(pi/2) ├─■─┤ RX(pi/2) ├┤ RZ(0.2) ├┤ RX(pi/2) ├─■─┤ RX(pi/2) ├┤ RZ(0.4) ├┤ RX(pi/2) ├─░──╫──╫──╫──╫─┤M├
          └──────────┘└───────┘└──────────┘   └──────────┘└─────────┘└──────────┘   └──────────┘└─────────┘└──────────┘ ░  ║  ║  ║  ║ └╥┘
    meas: ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════╩══╩══╩══╩══╩═
                                                                                   
