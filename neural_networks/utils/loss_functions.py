import numpy as np


def mse(y_true: np.array, y_pred: np.array) -> float:
    """
    Mean Squared Error (MSE) for regression tasks
    :param y_true: true values
    :param y_pred: predicted values
    :return: MSE
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_derivative(y_true: np.array, y_pred: np.array) -> np.array:
    """
    MSE derivative for regression tasks
    :param y_true: true values
    :param y_pred: predicted values
    :return: np.array with derivatives for each weight
    """
    return -2 * (y_true - y_pred) / len(y_true)


def log_loss(y_true: np.array, a_pred: np.array) -> float:
    """
    LogLoss for binary classification tasks
    :param y_true: true values (
    :param a_pred: predicted probabilities of each class [0, 1] (after sigmoid function or others)
    :return: LogLoss
    """
    return np.mean(-y_true * np.log(a_pred) - (1 - y_true) * np.log(1 - a_pred))


def log_loss_derivative(y_true: np.array, a_pred: np.array) -> np.array:
    """
    LogLoss derivative for binary classification tasks
    :param y_true: true values (true labels, i.e. [0, 1, 1, 0...])
    :param a_pred: predicted probabilities of each class [0, 1] (after sigmoid function or others)
    :return: np.array with derivatives
    """
    return (-y_true / a_pred + (1 - y_true) / (1 - a_pred)) / len(y_true)


def cross_entropy_loss(y_true: np.array, a_pred: np.array) -> float:
    """
    CrossEntropyLoss for multi-classification tasks
    :param y_true: 2D vector with classes, i.e. [[0], [3], [4], [1], [2]]
    :param a_pred: scores for each class before softmax function with shape [n_samples, n_classes]
    :return: CrossEntropyLoss
    """
    lenght_y = list(range(len(y_true)))
    arg = -a_pred[lenght_y, y_true.ravel()]
    sum_exp = np.sum(np.exp(a_pred), axis=1)
    loss = np.sum(arg + np.log(sum_exp))
    return loss / len(y_true)


def cross_entropy_loss_derivative(y_true: np.array, a_pred: np.array) -> np.array:
    """
    CrossEntropyLoss derivative for multi-classification tasks
    :param y_true: 2D vector with classes, i.e. [[0], [3], [4], [1], [2]]
    :param a_pred: scores for each class before softmax function with shape [n_samples, n_classes]
    :return: np.array with shape [n_samples, n_classes] with CrossEntropyLoss derivatives for each weight
    """
    lenght_y = list(range(len(y_true)))
    sum_exp = np.sum(np.exp(a_pred), axis=1).reshape(-1, 1)
    loss = np.exp(a_pred.copy()) / sum_exp
    loss[lenght_y, y_true.ravel()] -= 1

    return loss / len(y_true)


def cross_entropy_loss_naive(y_true: np.array, a_pred: np.array) -> float:
    """
    CrossEntropyLoss for multi-classification tasks, but naive, works pretty slow
    :param y_true: 2D vector with classes, i.e. [[0], [3], [4], [1], [2]]
    :param a_pred: scores for each class before softmax function with shape [n_samples, n_classes]
    :return: CrossEntropyLoss
    """
    loss = 0
    for i, j in enumerate(y_true):
        sum_exp = np.sum(np.exp(a_pred[i]))
        loss += -a_pred[i][j[0]] + np.log(sum_exp)
    return loss / len(y_true)


def cross_entropy_loss_derivative_naive(y_true: np.array, a_pred: np.array) -> np.array:
    """
    CrossEntropyLoss derivative for multi-classification tasks, but naive, works pretty slow
    :param y_true: 2D vector with classes, i.e. [[0], [3], [4], [1], [2]]
    :param a_pred: scores for each class before softmax function with shape [n_samples, n_classes]
    :return: np.array with shape [n_samples, n_classes] with CrossEntropyLoss derivatives for each weight
    """
    loss = np.zeros((y_true.shape[0], a_pred.shape[1]))
    for sample_idx, target in enumerate(y_true):
        sum_exp = np.sum(np.exp(a_pred[sample_idx]))
        deriv = -1 + np.exp(a_pred[sample_idx][target[0]]) / sum_exp
        loss[sample_idx][target[0]] = deriv / len(y_true)
        no_target_values = np.where(np.array(range(loss.shape[1])) != target[0])[0]

        for no_target in no_target_values:
            deriv = np.exp(a_pred[sample_idx][no_target]) / sum_exp
            loss[sample_idx][no_target] = deriv / len(y_true)
    return loss
