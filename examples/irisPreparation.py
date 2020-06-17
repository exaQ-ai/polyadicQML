import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split

def gauss_to_angle(X):
    """ Rescale to [-.95 * pi, .95 * pi] a standardized matrix, only keeping the points in the 99% confidence interval (in [-3, 3]).

    Parameters
    ----------
    X : array
        Standardized matrix.

    Returns
    -------
    array, list
        Rescaled matrix (w/o outliers), outliers list.
    """
    # We drop points outside the 99% Gaussian Confidence Interval (< -3 or > 3)
    X = X.copy()
    X[X > 3] = np.nan
    X[X < -3] = np.nan

    idx_to_drop = X.index[np.where(X.isna())[0]]
    X.dropna(inplace=True)

    # We project the [-3, 3] interval to [-0.95*\pi, 0.95*\pi]
    X = X / 3 * 0.95 * np.pi
    return X, idx_to_drop

COLUMNS = ['sep_len', 'sep_wid', 'pet_len', 'pet_wid']
COL_ORDER = [0, 1, 2, 3]

def makeDatasets(train_size, test_size, col_order=None, seed=None, **kwargs):
    """Create quantum compatible iris dataset and split it in train and test.
    
    Parameters
    ----------
    train_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples.
    test_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
    col_order : list, optional
        The desired order of the columns of the input matrix. It should be an integer indexing of the list "['sep_len', 'sep_wid', 'pet_len', 'pet_wid']".
        Example [2,0,3,1] -> ['pet_len', 'sep_len', 'pet_wid', 'sep_wid']
        If None, then uses natural order. By default is None.
    seed : int, optional
        Random seed for the split, by default None
    
    Returns
    -------
    dataframe, series, dataframe, series
        input_train, target_train, input_test, target_test
    """
    if seed is not None: np.random.seed(seed)
    if col_order is None: col_order = COL_ORDER

    iris = datasets.load_iris()
    columns = [COLUMNS[i] for i in col_order]
    data = pd.DataFrame(iris.data[:, col_order],
                        columns=columns)
    data["label"] = iris.target

    # Train-test split
    data_train, data_test = train_test_split(
        data, test_size=test_size, train_size=train_size, random_state=seed, stratify=data.label)

    # Splitting design matrix and response
    # NOTE: We copy target and input as indexing returns a view, to free memory we deelete the datased afterwards
    target_train = data_train.label.copy()
    input_train = data_train.drop(columns="label").copy()
    # del data_train

    target_test = data_test.label.copy()
    input_test = data_test.drop(columns="label").copy()
    # del data_test

    # NORMALIZATION
    mean = input_train.mean(axis=0)
    std = input_train.std(axis=0)

    input_train = (input_train - mean) / std
    input_test = (input_test - mean) / std

    input_train, tr_to_drop = gauss_to_angle(input_train)
    target_train.drop(index=tr_to_drop, inplace=True)

    input_test, te_to_drop = gauss_to_angle(input_test)
    target_test.drop(index=te_to_drop, inplace=True)

    return input_train.values, target_train.values, input_test.values, target_test.values
