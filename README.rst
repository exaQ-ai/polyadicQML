#################################
Polyadic Quantum Machine Learning
#################################

This package provides a library to define, train and deploy
**Quantum Machine Learning** models.

This library has been used to train a qmodel with the Iris flower dataset on IBM quantum computers: iris.entropicalabs.io_

The quantum circuits can run on top of any quantum computer provider.
As for now, it implements interfaces for a fast simulator, *manyq*, and 
Qiskit_.

Installation
############

From PyPI, at the command line::

   pip install polyadicqml

Installing latest stable from github::

   git clone https://github.com/entropicalabs/polyadicQML.git polyadicqml
   cd polyadicqml
   pip install -U .


Documentation
#############

You can find a `quickstart guide`_, the tutorial_ and the module references_ in the docs_.


Sample code
###########

Training a model on a simulator and testing it on a real quantum computer can
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

You can find out more in the `documentation`_, where you will find tutorials and examples.
A quickstart through examples can be found in the `examples folder`_, as well as on the website.
As an introduction to the algorithm you can check out this `video
presentation`_. 

.. _iris.entropicalabs.io: https://iris.entropicalabs.io/

.. _`video presentation`: https://youtu.be/QZ8ynyG-O9U
.. _polyadicQML: https://polyadicqml.entropicalabs.io/
.. _Qiskit: https://qiskit.org/
.. _polyadicqml.entropicalabs.io: https://polyadicqml.entropicalabs.io
.. _docs: https://polyadicqml.entropicalabs.io
.. _`documentation`: https://polyadicqml.entropicalabs.io
.. _`examples folder`: https://github.com/entropicalabs/polyadicQML/tree/master/examples
.. _`quickstart guide`: https://polyadicqml.entropicalabs.io/#quickstart
.. _tutorial: https://polyadicqml.entropicalabs.io/#user-s-guide
.. _references: https://polyadicqml.entropicalabs.io/#modules
