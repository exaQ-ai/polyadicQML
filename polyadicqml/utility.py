import numpy as np
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score


def stable_softmax(X, axis=None):
    exps = np.exp(X - np.max(X, axis=axis).reshape(-1, 1))
    return exps / np.sum(exps, axis=axis).reshape(-1, 1)


def CE_loss(y_true, y_pred, labels=None):
    """Cross entropy loss.

    Parameters
    ----------
    y_true : vector
        Ground truth (correct) labels for n_samples samples
    y_pred : vector
        Predicted probabilities, as returned by a classifier’s
        ``predict_proba`` method.
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
        Predicted probabilities, as returned by a classifier’s
        ``predict_proba`` method.

    Returns
    -------
    float
        Loss value.
    """

    n = len(y_pred)

    sigma = stable_softmax(y_pred, axis=1)
    sigma[range(n), y_true] -= 1
    return sigma.sum(axis=0) / n


def print_results(target, pred, name=None, output=None):
    """Print confusion matrix and accuracy of predicted labels vs actual
    target.

    Parameters
    ----------
    target : vector
        Actual classes
    pred : vector
        Predicted labels
    name : str, optional
        Which name to print, required if output is not None. By default None.
    output : str, optional
        Output type, if None then results are printed to std.Out. By default
        ``None``
        - `dict` : return the scores in a nested dictionary
        ( `{name : {'confusion_matrix' : ..., 'accuracy' : ...})

    Returns
    -------
    [output]
        If `output` is not ``None``, the results are returned in desired
        structure.
    """
    if output is None:
        if name is None:
            name = "input"
        print(
            '\n' + 30*'#',
            "Confusion matrix on {}:".format(name),
            confusion_matrix(target, pred),
            "Accuracy : " + str(accuracy_score(target, pred)),
            sep='\n'
        )
        return None
    elif output == 'dict':
        if name is None:
            raise ValueError("Cannot output dict if `name` is not given")
        out = {name: {"confusion_matrix":
                      confusion_matrix(target, pred).tolist(),
                      "accuracy": accuracy_score(target, pred).item()
                      }
               }
        return out
