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

Quickstart
##########

With polyadicQML_, training a quantum-machine-learning model and using it to
predict on new points can be done in a few lines.
Suppose we have the following ``quickstart.py`` script:

.. literalinclude:: ../../examples/quickstart.py
   :emphasize-lines: 31-34, 39-42, 47-52, 65-66

We can run it and train our first QML model:

.. code-block:: console

   $ python3 quickstart.py

   ##########################
   Confusion matrix on train:
   [[66  3]
   [ 4 67]]
   Accuracy : 0.95

   ##########################
   Confusion matrix on test:
   [[30  1]
   [ 2 27]]
   Accuracy : 0.95

To understand what we where doing in the script, have a look at our :ref:`sec-tutorial`.

User's Guide
############

.. toctree::
   :maxdepth: 3

   tutorial
   use-cases

Modules
#######

.. toctree::
   :maxdepth: 3

   polyadicqml
   manyq
   qiskit

Indices and tables
##################

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
