####################
# QUICKSTART EXAMPLE
####################

#########################
# Load dataset and labels
import numpy as np
np.random.seed(42)

n_pc = 100  # Points per component

# Create a matrix on the point and its symmetry
px, py = 0.75, 0.75
X = np.asarray(n_pc * [[px, py]] +
               n_pc * [[-px, -py]]
)
# Add gaussian noise
X += 0.7 *  np.random.randn(*X.shape)
# Create target vecor
y = np.concatenate((np.zeros(n_pc), np.ones(n_pc)))

# Split in train and test
from sklearn.model_selection import train_test_split

input_train, input_test, target_train, target_test =\
    train_test_split(X, y, test_size=.3)

#############################
# Define the ciruit structure

def simple_circuit(bdr, x, params):
    bdr.allin(x).cz(0,1).allin(params[:2])
    bdr.cz(0,1).allin(params[2:4])
    return bdr

##############################
# Prepare a circuit simulator:

from polyadicqml.manyq import mqCircuitML

qc = mqCircuitML(make_circuit=simple_circuit,
                 nbqbits=2, nbparams=4)

#################################
# Instanciate and train the model

from polyadicqml import Classifier 

# Choose two bitstrings
bitstr = ["01", "10"]

model = Classifier(qc, bitstr).fit(input_train, target_train)

###############################
# Predict and print the results

pred_train = model(input_train)
pred_test = model(input_test)

from polyadicqml.utility import print_results

print_results(target_train, pred_train, name="train")
print_results(target_test, pred_test, name="test")
