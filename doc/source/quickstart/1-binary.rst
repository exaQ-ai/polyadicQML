.. include:: ../substs.rst

A binary classifier
===================

In this tutorial we train and test a binary-quantum-classifier.

Dataset generation
------------------

We generate a dataset and its labels using :mod:`numpy` and we randomly split
it in a train and test with :func:`~sklearn.model_selection.train_test_split`
from :mod:`sklearn`.

.. literalinclude:: ../../../examples/quickstart.py
   :lines: 7-26

Circuit definition
------------------

Now, we need to define the structure of the circuit, we do this using a
|make_c| function.
This function can have any name, but must take exactly three arguments, in the
following order: ``(bdr, x, params)``.
These correspond to a |circuitBuilder|, to an input vector, and to a
parameter vector.

.. literalinclude:: ../../../examples/quickstart.py
   :lines: 31-34

This corresponds to the circuit in the following figure, where we use `input`
gates and `CZ entanglements`.
The precise syntax of |make_c| and the meaning of the `gates` are explained in the ":ref:`sec-circuit`" tutorial .

.. image:: ../figures/circuit-2qb-binary.png
   :scale: 25 %
   :alt: Parametric quantum circuit on two qubits
   :align: center

Now, we need to translate our description in a runnable circuit.
This is obtained using a |circuitML| class, which interacts with a backend;
in this case we use :ref:`sec-manyq`.

.. literalinclude:: ../../../examples/quickstart.py
   :lines: 39-42

Model training 
---------------

At this point, we are ready to create and train our first quantum |Classifier|.
We only need to choose which bitsrings will be used to predict the classes.

.. literalinclude:: ../../../examples/quickstart.py
   :lines: 47-52

Predict on new data
-------------------

Once the model is trained, we can easily predict the class of any new sample.

.. literalinclude:: ../../../examples/quickstart.py
   :lines: 57-58

And we can assert the performance of the model by confronting the predictions
and the true labels.

::

    >>> from polyadicqml.utility import print_results

    >>> print_results(target_train, pred_train, name="train")

   ##########################
   Confusion matrix on train:
   [[66  3]
    [ 4 67]]
   Accuracy : 0.95

    >>> print_results(target_test, pred_test, name="test")

   ##########################
   Confusion matrix on test:
   [[30  1]
    [ 2 27]]
   Accuracy : 0.95

Source code
-----------

This example script can be found in the `GitHub example page`_ as
``quickstart.py``.
