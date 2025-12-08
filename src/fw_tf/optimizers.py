from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import tensorflow as tf

from .constraints import ConstraintSet


class FrankWolfeOptimizer(tf.keras.optimizers.Optimizer):
    """
    Оптимизатор Frank–Wolfe для Keras-моделей.

    Обновление для каждого тензора параметров var:
        1) grad = ∂L/∂var
        2) s_k = argmin_{s in S} <grad, s>
        3) var_{k+1} = (1 - gamma_k) * var_k + gamma_k * s_k,
           где gamma_k = 2 / (k + 2).
    """

    def __init__(
        self,
        constraint: ConstraintSet,
        name: str = "FrankWolfe",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self._constraint = constraint

        # Счётчик итераций (глобальный для всех параметров).
        self._step = self.add_weight(
            name="iteration",
            shape=[],
            dtype=tf.int64,
            initializer="zeros",
            trainable=False,
        )

    def _create_slots(self, var_list: Any) -> None:  # noqa: D401, ANN401
        # Слоты не нужны: FW не использует моменты и т.п.
        return

    def _resource_apply_dense(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        apply_state: Optional[Mapping[str, Any]] = None,
    ) -> tf.Tensor:
        # k <- k + 1
        step = self._step.assign_add(1)
        step_f = tf.cast(step, var.dtype)

        # gamma_k = 2 / (k + 2)
        gamma = 2.0 / (step_f + 2.0)

        # s_k = argmin_{s in S} <grad, s>
        s_k = self._constraint.argmin(grad=grad, var=var)

        new_var = (1.0 - gamma) * var + gamma * s_k
        return var.assign(new_var)

    def _resource_apply_sparse(
        self,
        grad: tf.Tensor,
        var: tf.Variable,
        indices: tf.Tensor,
        apply_state: Optional[Mapping[str, Any]] = None,
    ) -> tf.Tensor:
        # Простейшая (не оптимальная) реализация: разреженный -> плотный
        dense_grad = tf.convert_to_tensor(grad)
        return self._resource_apply_dense(dense_grad, var, apply_state)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        # constraint нельзя просто так сериализовать; сохраняем класс по имени.
        return {
            **base_config,
            "constraint_class": self._constraint.__class__.__name__,
        }
