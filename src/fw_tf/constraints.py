from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf


class ConstraintSet(ABC):
    """Абстракция множества S: определяет линейный оракул argmin_{s in S} <g, s>."""

    @abstractmethod
    def argmin(self, grad: tf.Tensor, var: tf.Tensor) -> tf.Tensor:
        """
        Возвращает s_k = argmin_{s in S} <grad, s>.

        grad: градиент по var (той же формы, что и var).
        var: текущие веса (может использоваться при определении S, если нужно).
        """
        raise NotImplementedError


class L2BallConstraint(ConstraintSet):
    """
    Ограничение вида ||w||_2 <= radius для ОДНОГО тензора var.

    Линейный оракул:
        s = -radius * g / ||g||, если g != 0,
        s = 0, если g == 0.
    """

    def __init__(self, radius: float = 1.0) -> None:
        self.radius = float(radius)

    def argmin(self, grad: tf.Tensor, var: tf.Tensor) -> tf.Tensor:  # noqa: ARG002
        grad = tf.convert_to_tensor(grad)
        norm = tf.norm(grad)

        def on_non_zero() -> tf.Tensor:
            direction = grad / norm
            return -self.radius * direction

        def on_zero() -> tf.Tensor:
            return tf.zeros_like(grad)

        return tf.cond(norm > 0, on_non_zero, on_zero)


class LInfBallConstraint(ConstraintSet):
    """
    Ограничение вида |w_i| <= radius поэлементно (L∞-шар).

    Линейный оракул:
        s_i = -radius * sign(g_i).
    """

    def __init__(self, radius: float = 1.0) -> None:
        self.radius = float(radius)

    def argmin(self, grad: tf.Tensor, var: tf.Tensor) -> tf.Tensor:  # noqa: ARG002
        grad = tf.convert_to_tensor(grad)
        # sign(0) = 0, это ок
        return -self.radius * tf.sign(grad)
