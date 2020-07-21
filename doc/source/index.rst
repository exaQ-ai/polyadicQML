########################################
Welcome to Polyadic QML's documentation!
########################################

.. _Qiskit: https://qiskit.org/
.. _`video presentation`: https://youtu.be/QZ8ynyG-O9U
.. _polyadicQML: https://polyadicqml.entropicalabs.io/

This package provides an high level API to define, train and deploy
**Polyadic Quantum Machine Learning** models.

It implements a general interface which can be used with any quantum provider.
As for now, it supports a fast simulator, *manyq*, and
Qiskit_.
More are coming.

With polyadicQML_, Training a model on a simulator and testing it on a real quantum computer can
be done in a few lines:

.. code-block:: python

   # Define the circuit structure
   make_circuit(bdr, x, params):
      ...
   
   # Prepare a circuit simulator:

   qc = mqCircuitML(make_circuit=make_circuit,
                    nbqbits=nbqbits, nbparams=nbparams)

   # Instanciate and train the model

   model = Classifier(qc, bitstr).fit(input_train, target_train)

   # Prepare to run the circuit on an IBMq machine:

   backend = Backends("ibmq_ourense", hub="ibm-q")

   qc2 = qkCircuitML(
      make_circuit=make_circuit,
      nbqbits=nbqbits, nbparams=nbparams,
      backend=backend
   )

   # Change the model backend and run it
   model.set_circuit(qc2)
   model.nbshots = 300
   model.job_size = 30

   pred_test = model(input_test)

You can find out more in the :ref:`sec-quickstart`  and in the :ref:`sec-tutorial`.
As an introduction to the algorithm you can check out this `video
presentation`_ from the **IBM Singapore Supercomputing Virtual Forum**. 
This code has been used to fully train a Quantum Machine Learning model
on a real quantum computer to classify the Iris flower dataset.

Installation
############

From PyPI, at the command line (not yet supported)::

   pip install polyadicqml

Installing latest stable from github::

   git clone https://github.com/entropicalabs/polyadicQML.git polyadicqml
   cd polyadicqml
   pip install -U .

.. _sec-quickstart:

Quickstart
##########

With polyadicQML_, training a quantum-machine-learning model and using it to
predict on new points can be done in a few lines.

The following links provide a quick overview of the API through some
motivating examples.
To gain a deeper understanding of the interface, and of quantum machine
learning, have a look at our :ref:`sec-tutorial`.

.. toctree::
   :caption: Learn though examples : 
   :glob:
   :numbered:
   :titlesonly:
   
   quickstart/*

.. _sec-tutorial:

User's Guide
############

In the following tutorial, we will discover, step by step, the :mod:`polyadicqml` package.

At first, we learn how to define a **parametric quantum circuit**.
Then, we understand how to train and test a **Quantum Classifier**.
Finally, we see the different backends supported by this package.

At the end of the tutorial, you should be able to use a Quantum Classifier for any problem, provided you have enough computing power.


.. toctree::
   :caption: Tutorial : 
   :maxdepth: 2
   :glob:
   :numbered:
   
   tutorial/*

.. toctree::
   :caption: Modules : 
   :name: modules
   :maxdepth: 3

   polyadicqml
   manyq
   qiskit

Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
