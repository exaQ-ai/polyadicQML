# Examples

## Example 1 : The XOR problem

Our first example is the XOR problem.
We dispose four points over the cartesian axes so to create a centered square; the two points on $x$-axis are labeled as 1, while those on $y$-axis as 0.

### Dataset generation

We use numpy to generate a dataset of 200 points by sampling 50 points from 4 gaussian distibutions centered at the said points.
The label of each sample is given by the center of its distribution.
```python
import numpy as np

n_pc = 50 # Number of points per cluster

# Create a matrix of vertices of the centered square
X = np.asarray(n_pc * [[1.5, 0.]] +
               n_pc * [[-1.5, 0.]] + 
               n_pc * [[0., -1.5]] + 
               n_pc * [[0., 1.5]]
)
# Add gaussian noise
X += .5 * np.random.randn(*X.shape)

# Create target vecor
y = np.concatenate((np.zeros(2*n_pc), np.ones(2*n_pc)))
```

This generates the following dataset, where the circles represent the samples and the squares the distribution centers.

![XOR scatterplot](../figures/XOR-points.png "XOR scatterplot")

### Circuit definition

Now, we define the circuit structure using a `circuitBulder`.
This function has to respect a precise signature: `make_circuit(bdr, x, params, shots=None)`. 
```python

def make_circuit(bdr, x, params, shots=None):
    bdr.allin(x[[0,1]])

    bdr.cz(0, 1)
    bdr.allin(params[[0,1]])

    bdr.cz(0, 1)
    bdr.allin(params[[2,3]])

    if shots: bdr.measure_all()
    return bdr.circuit()
```
Second last line add a measure gates only if a number of shots is provided. This prevents errors in case we want to use the satevector amplitudes for computations, as measuring the quantum state collapses the probabilities.

### Model training 

Finally, we can create and train the classifier. 
We instanciate the `circuitML` subclass that we prefer, in this case the one using the fast *manyq* simualtor, specifying the number of qubits and of parameters.

```python
from polyadicqml.manyq import mqCircuitML

nbqbits = 2
nbparams = 6

qc = mqCircuitML(make_circuit=make_circuit,
                nbqbits=nbqbits, nbparams=nbparams)
```

Then, we create and train the quantum classifer, specifying on which bitstrings we want to read the predicted classes.

```python
from polyadicqml import Classifier

bitstr = ['00', '01']

model = Classifier(qc, bitstr)
model.fit(X, y)
```

### Predict on new data

We can use a model to predict on some new datapoints `X_test` that it never saw before.
To obtain the bitstring probabilities, we can just call the model, by 

```python
pred_prob = model(X_test)
```

Then, we can retrieve the label of each point as the argmax of the corresponding probabilities.
Otherwise, we can combine the two operation by using the shorthand 

```python
y_pred = model.predict_label(X_test)
```

For instance, going back to our XOR problem, we can predict the label of each point on a grid that covers [-pi,pi]x[-pi,pi], to assess the model accuracy.
Using some list comprhension, it would look like this:

```python
t = np.linspace(-np.pi,np.pi, num = 50)
X_test = np.array([[t1, t2] for t1 in t for t2 in t])

y_pred = model.predict_label(X_test)
```

We can now plot the predictions and see that the model is very close to the bayesian prediction (whose decision boundaries are shown as grey lines), which is the best possible.

![XOR predictions](../figures/XOR-predictions.png "XOR predictions")

## Example 2: The iris dataset