import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

COLUMNS = ['sep_len', 'sep_wid', 'pet_len', 'pet_wid']
COL_ORDER = [0, 1, 2, 3]

def makeDatasets(train_size, test_size, seed=None, **kwargs):
    """Create quantum compatible iris dataset and split it in train and test.
    
    Parameters
    ----------
    train_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split. If int, represents the absolute number of train samples.
    test_size : float or int
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.
    seed : int, optional
        Random seed for the split, by default None
    
    Returns
    -------
    dataframe, series, dataframe, series
        input_train, target_train, input_test, target_test
    """
    if seed is not None: np.random.seed(seed)

    iris = datasets.load_iris()
    data = iris.data
    target = iris.target

    # Train-test split
    input_train, input_test, target_train, target_test = train_test_split(
        data, target, test_size=test_size, train_size=train_size, random_state=seed, stratify=target)

    # del data_test

    # NORMALIZATION
    mean = input_train.mean(axis=0)
    std = input_train.std(axis=0)

    input_train = (input_train - mean) / std / 3 * 0.95 * np.pi
    input_test = (input_test - mean) / std / 3 * 0.95 * np.pi

    return input_train, target_train, input_test, target_test
