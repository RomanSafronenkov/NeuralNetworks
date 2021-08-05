from abc import ABC, abstractmethod

import numpy as np


class BasePreprocessor(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(self, x: np.array) -> np.array:
        pass

    @abstractmethod
    def transform(self, x: np.array) -> np.array:
        pass

    def fit_transform(self, x: np.array) -> np.array:
        """
        Performs fit() and transform() methods
        :param x: np.array with shape (n_rows, n_cols)
        :return: transformed np.array
        """
        self.fit(x)
        return self.transform(x)


class StandardScaler(BasePreprocessor):
    """
    StandardScaler class
    """
    def __init__(self, ddof: int = 0) -> None:
        """
        :param ddof: degrees of freedom when finding std
        """
        self.mean = None
        self.std = None
        self.ddof = ddof

    def fit(self, x: np.array):
        """
        :param x: np.array with shape (n_rows, n_cols)
        :return: self
        """
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0, ddof=self.ddof)
        return self

    def transform(self, x: np.array) -> np.array:
        """
        :param x: np.array with shape (n_rows, n_cols)
        :return: transformed np.array
        """
        eps = 1e-10
        x = x.copy()
        return (x - self.mean) / (self.std + eps)


class MinMaxScaler(BasePreprocessor):
    """
    MinMaxScaler class
    """
    def __init__(self) -> None:
        self.max = None
        self.min = None

    def fit(self, x: np.array):
        """
        :param x: np.array with shape (n_rows, n_cols)
        :return: self
        """
        self.max = np.max(x, axis=0)
        self.min = np.min(x, axis=0)
        return self

    def transform(self, x: np.array) -> np.array:
        """
        :param x: np.array with shape (n_rows, n_cols)
        :return: transformed np.array
        """
        eps = 1e-10
        x = x.copy()
        return (x - self.min) / (self.max - self.min + eps)
