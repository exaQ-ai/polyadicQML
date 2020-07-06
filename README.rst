#################################
Polyadic Quantum Machine Learning
#################################

Source code for the polyadicQML_ package.

This package provides an high level API to define, train and deploy
**Polyadic Quantum Machine Learning** models.

It implements a general interface which can be used with any quantum provider.
As for now, it supports a fast simulator, *manyq*, and
qiskit_.
More are coming.

As an introduction to the algorithm you can check out this `video
presentation`_ from the **IBM Singapore Supercomputing Virtual Forum**. 
This code has been used to fully train a Quantum Machine Learning model
on a real quantum computer to classify the Iris flower dataset.

Installation
############

From PyPI, at the command line (not yet supported)::

   pip install polyadicqml

.. Installing latest stable from github::

..    git clone https://github.com/entropica/example-repo.git polyadicqml
..    cd polyadicqml
..    pip install -U .

Documentation
#############

You can find a tutorial_ and the module references_ at polyadicqml.entropicalabs.io_

.. _`video presentation`: https://youtu.be/QZ8ynyG-O9U
.. _polyadicQML: https://polyadicqml.entropicalabs.io/
.. _qiskit: https://qiskit.org/
.. _polyadicqml.entropicalabs.io: https://polyadicqml.entropicalabs.io
.. _tutorial: https://polyadicqml.entropicalabs.io/tutorial
.. _references: https://polyadicqml.entropicalabs.io/#modules