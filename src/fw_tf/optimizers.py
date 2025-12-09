from __future__ import annotations

from typing import Any

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
        gamma: float = 0.05,
        name: str = "FrankWolfe",
        **kwargs: Any,
    ) -> None:

        super().__init__(name=name, **{"learning_rate": 1.0, **kwargs})
        self._constraint = constraint
        self.gamma = float(gamma)

    def build(self, var_list):
        super().build(var_list)

    def update_step(self, gradient, variable, learning_rate=None):
        """Single FW update: x_{k+1} = (1 - γ) x_k + γ s_k.

        Keras 3 передаёт сюда (gradient, variable, learning_rate);
        learning_rate игнорируем.
        """
        if gradient is None:
            return

        s_k = self._constraint.argmin(grad=gradient, var=variable)

        gamma = tf.cast(self.gamma, variable.dtype)
        new_var = (1.0 - gamma) * variable + gamma * s_k
        variable.assign(new_var)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "constraint": tf.keras.saving.serialize_keras_object(
                    self._constraint
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        constraint_cfg = config.pop("constraint")
        constraint = tf.keras.saving.deserialize_keras_object(constraint_cfg)
        return cls(constraint=constraint, **config)