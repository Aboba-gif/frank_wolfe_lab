from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf


class ConstraintSet(ABC):
    """
    Абстракция множества S для метода Франка–Вольфа.

    Определяет линейный оракул:
        argmin_{s in S} <grad, s>
    для заданного градиента по переменной var.
    """

    @abstractmethod
    def argmin(self, grad: tf.Tensor, var: tf.Tensor) -> tf.Tensor:
        """
        Возвращает s_k = argmin_{s in S} <grad, s>.

        Параметры
        ---------
        grad : tf.Tensor
            Градиент по var (той же формы, что и var).
        var : tf.Tensor
            Текущие веса. Может использоваться при определении множества S.

        Возвращает
        ----------
        s_k : tf.Tensor
            Тензор той же формы, что и var.
        """
        raise NotImplementedError


class L2BallConstraint(ConstraintSet):
    """
    Ограничение вида ||w||_2 <= radius (L2-шар).

    Линейный оракул:
        s_k = -radius * grad / ||grad||_2.
    """

    def __init__(self, radius: float = 1.0) -> None:
        self.radius = float(radius)

    def argmin(self, grad: tf.Tensor, var: tf.Tensor) -> tf.Tensor:  # noqa: ARG002
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

    def argmin(self, grad: tf.Tensor, var: tf.Tensor) -> tf.Tensor:
        grad = tf.convert_to_tensor(grad)
        radius = tf.cast(self.radius, grad.dtype)
        return -radius * tf.sign(grad)
