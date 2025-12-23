from __future__ import annotations

from typing import Generator, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold


Array = np.ndarray
SplitGen = Generator[
    Tuple[Tuple[Array, Array], Tuple[Array, Array]], None, None
]


def load_cifar10() -> Tuple[Array, Array, Array, Array]:
    """
    Загружает CIFAR-10 из tf.keras.datasets.

    Возвращает
    ----------
    x_train : (50000, 32, 32, 3) uint8
    y_train : (50000,) int64
    x_test  : (10000, 32, 32, 3) uint8
    y_test  : (10000,) int64
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze().astype(np.int64)
    y_test = y_test.squeeze().astype(np.int64)
    return x_train, y_train, x_test, y_test


def kfold_splits(
    x: Array,
    y: Array,
    n_splits: int = 5,
    random_state: int = 42,
) -> SplitGen:
    """
    Генератор разбиений K-fold по индексам объектов.

    Параметры
    ---------
    x : np.ndarray
        Обучающие объекты.
    y : np.ndarray
        Метки.
    n_splits : int
        Число фолдов.
    random_state : int
        Seed для перемешивания.

    Возвращает
    ----------
    Генератор ((x_train, y_train), (x_val, y_val)) для каждого фолда.
    """
    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    for train_idx, val_idx in kf.split(x):
        x_train, x_val = x[train_idx], x[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        yield (x_train, y_train), (x_val, y_val)
