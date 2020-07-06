import numpy as np
from sklearn.metrics import log_loss

def stable_softmax(X, axis=None):
    exps = np.exp(X - np.max(X, axis=axis).reshape(-1,1))
    return exps / np.sum(exps, axis=axis).reshape(-1,1)

def CE_loss(y_true, y_pred, labels=None):
    """Cross entropy loss.

    Parameters
    ----------
    y_true : vector
        Ground truth (correct) labels for n_samples samples
    y_pred : vector
        Predicted probabilities, as returned by a classifier’s ``predict_proba`` method.
    labels : vector, optional
        If not provided, labels will be inferred from `y_true`.

    Returns
    -------
    float
        Loss value.
    """
    # TODO: verify advantage of penalizing missing probabilities
    # Add remaining probabilities as unrepresented class
    # if labels is None:
    #     labels = np.unique(y_pred)
    # labels=np.concatenate((labels, [len(labels)]))
    # _y = np.hstack((y_pred, (1 - y_pred.sum(axis=1)).reshape(-1,1)))

    return log_loss(y_true, stable_softmax(y_pred, axis=1), labels=labels)

def CE_grad(y_true, y_pred):
    """Cross entropy loss gradient, w.r.t. y_pred columns.

    Parameters
    ----------
    y_true : vector
        Ground truth (correct) labels for n_samples samples
    y_pred : vector
        Predicted probabilities, as returned by a classifier’s ``predict_proba`` method.

    Returns
    -------
    float
        Loss value.
    """

    n = len(y_pred)

    sigma = stable_softmax(y_pred, axis=1)
    sigma[range(n), y_true] -= 1
    return sigma.sum(axis=0) / n
