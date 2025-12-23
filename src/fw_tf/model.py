from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


def make_cifar10_cnn(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
) -> tf.keras.Model:
    """
    Простая CNN для CIFAR-10.

    Архитектура:
      - Rescaling(1/255)
      - Conv(32) x2 + MaxPool + Dropout
      - Conv(64) x2 + MaxPool + Dropout
      - Flatten + Dense(256) + Dropout
      - Dense(num_classes, softmax)
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="cifar10_cnn")
    return model


def make_cifar10_backbone_with_head(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
    num_classes: int = 10,
) -> tuple[tf.keras.Model, tf.keras.layers.Layer]:
    """
    Возвращает полную модель и ссылку на последний Dense-слой (head).

    Это удобно для экспериментов вида:
      1) тренируем весь backbone Adam'ом,
      2) замораживаем backbone, дообучаем только head FW-оптимизатором.
    """
    model = make_cifar10_cnn(input_shape=input_shape, num_classes=num_classes)
    head = model.get_layer("classifier")
    return model, head
