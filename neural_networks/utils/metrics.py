import numpy as np

from neural_networks.utils.loss_functions import mse


def r2_score(y_true: np.array, y_pred: np.array) -> float:
    """
    R^2 score for regression tasks a.k.a. Coefficient of determination
    :param y_true: np.array of true values
    :param y_pred: np.array of predicted values
    :return: r2_score, float
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return 1 - mse(y_true, y_pred) / np.mean((y_true - np.mean(y_true)) ** 2)


def accuracy_score(y_true: np.array, y_pred: np.array) -> float:
    """
    Accuracy score for classification tasks
    :param y_true: np.array of true classes
    :param y_pred: np.array of predicted classes
    :return: accuracy_score, float
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return np.mean(y_true == y_pred)
