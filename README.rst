#################################
Polyadic Quantum Machine Learning
#################################

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

You can find out more in the `official documentation`_, where you will find tutorials and examples.
A quickstart through examples can be found in the `examples folder`_, as well as on the website.
As an introduction to the algorithm you can check out this `video
presentation`_ from the **IBM Singapore Supercomputing Virtual Forum**. 
This code has been used to fully train a Quantum Machine Learning model
on a real quantum computer to classify the Iris flower dataset.

Documentation
#############

You can find a `quickstart guide`_, the tutorial_ and the module references_ at polyadicqml.entropicalabs.io_.

Installation
############

From PyPI, at the command line (not yet supported)::

   pip install polyadicqml

Installing latest stable from github::

   git clone https://github.com/entropicalabs/polyadicQML.git polyadicqml
   cd polyadicqml
   pip install -U .
 

.. _`video presentation`: https://youtu.be/QZ8ynyG-O9U
.. _polyadicQML: https://polyadicqml.entropicalabs.io/
.. _Qiskit: https://qiskit.org/
.. _polyadicqml.entropicalabs.io: https://polyadicqml.entropicalabs.io
.. _`official documentation`: https://polyadicqml.entropicalabs.io
.. _`examples folder`: https://github.com/entropicalabs/polyadicQML/tree/master/examples
.. _`quickstart guide`: https://polyadicqml.entropicalabs.io/#quickstart
.. _tutorial: https://polyadicqml.entropicalabs.io/#user-s-guide
.. _references: https://polyadicqml.entropicalabs.io/#modules