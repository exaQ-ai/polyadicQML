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

The following links provide a quick overview of the API through some
motivating examples.
To gain a deeper understanding of the interface, and of quantum machine
learning, have a look at our :ref:`sec-tutorial`.

.. toctree::
   :caption: Learn though examples : 
   :glob:
   :numbered:
   
   quickstart/*

User's Guide
############

.. toctree::
   :maxdepth: 3

   tutorial

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
