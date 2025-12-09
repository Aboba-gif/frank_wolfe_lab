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


class L2BallConstraint:
    def __init__(self, radius: float) -> None:
        self.radius = float(radius)

    def argmin(self, grad: tf.Tensor, var: tf.Tensor) -> tf.Tensor:
        # argmin_{||s||_2 <= R} <grad, s> = -R * grad / ||grad||
        grad = tf.convert_to_tensor(grad)
        eps = tf.cast(1e-12, grad.dtype)
        grad_norm = tf.norm(grad)
        safe_norm = tf.maximum(grad_norm, eps)
        direction = -grad / safe_norm
        radius = tf.cast(self.radius, grad.dtype)
        return radius * direction


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
        return -self.radius * tf.sign(grad)