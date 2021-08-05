import numpy as np
from typing import Tuple


def train_test_split(
        x: np.array,
        y: np.array,
        test_size: float,
        random_state: int = None
) -> Tuple[np.array, np.array, np.array, np.array]:
    assert 0 < test_size < 1, 'test_size should be in [0, 1)'
    if random_state:
        np.random.seed(random_state)
    n_obs = len(x)
    size = int(n_obs * (1 - test_size))

    indexes = np.arange(n_obs)
    train_idx = np.random.choice(indexes, size=size, replace=False)
    test_idx = np.delete(indexes, train_idx)

    x_train, y_train = x[train_idx], y[train_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    return x_train, x_test, y_train, y_test
