from __future__ import annotations

from typing import Generator, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold


def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    return x_train, y_train, x_test, y_test


def kfold_splits(
    x: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Generator[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]], None, None]:
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    for train_idx, val_idx in kf.split(x):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        yield (x_train, y_train), (x_val, y_val)
