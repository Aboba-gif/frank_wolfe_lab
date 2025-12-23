from __future__ import annotations

from typing import Any

import tensorflow as tf

from .constraints import ConstraintSet


class FrankWolfeOptimizer(tf.keras.optimizers.Optimizer):
    """
    Оптимизатор Frank–Wolfe для Keras-моделей.

    Обновление для каждого тензора параметров `var`:
      1) grad = ∂L/∂var
      2) s_k = argmin_{s in S} <grad, s>
      3) var_{k+1} = (1 - γ_k) * var_k + γ_k * s_k,

    где:
      - S задаётся объектом `ConstraintSet` (например, L2/L∞-шар),
      - γ_k либо фиксирован (gamma), либо убывает как 2/(k+2).

    Замечание:
      * learning_rate в этом оптимизаторе не используется (он всегда 1.0),
        но присутствует для совместимости с API Keras/TF.
    """

    def __init__(
        self,
        constraint: ConstraintSet,
        gamma: float = 0.05,
        use_diminishing_step: bool = False,
        name: str = "FrankWolfe",
        **kwargs: Any,
    ) -> None:

        super().__init__(name=name, **{"learning_rate": 1.0, **kwargs})
        self._constraint = constraint
        self.gamma = float(gamma)
        self.use_diminishing_step = bool(use_diminishing_step)

        # Счётчик шагов по батчам (не через add_weight, чтобы не зависеть от Keras 3)
        self._step = tf.Variable(
            0,
            dtype=tf.int64,
            trainable=False,
            name=f"{name}_step",
        )

    def build(self, var_list):
        super().build(var_list)

    def _current_gamma(self) -> tf.Tensor:
        """
        Возвращает γ_k в зависимости от настроек:
          - если use_diminishing_step=False: γ_k ≡ gamma;
          - иначе: γ_k = 2 / (k + 2), k = 0,1,2,...
        """
        if not self.use_diminishing_step:
            return tf.cast(self.gamma, tf.float32)

        k = tf.cast(self._step, tf.float32)
        gamma_k = 2.0 / (k + 2.0)
        gamma_k = tf.clip_by_value(gamma_k, 0.0, 1.0)
        return gamma_k

    def update_step(self, gradient, variable, learning_rate=None):
        """
        Один FW-апдейт:
            var <- (1 - γ_k) * var + γ_k * s_k.
        """
        if gradient is None:
            return

        s_k = self._constraint.argmin(grad=gradient, var=variable)

        gamma = tf.cast(self._current_gamma(), variable.dtype)
        new_var = (1.0 - gamma) * variable + gamma * s_k
        variable.assign(new_var)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """
        Обёртка над базовым apply_gradients без явного аргумента `name`,
        чтобы не зависеть от конкретной сигнатуры Keras/TF.
        """
        self._step.assign_add(1)
        return super().apply_gradients(grads_and_vars, *args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "gamma": self.gamma,
                "use_diminishing_step": self.use_diminishing_step,
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


class HybridFrankWolfeOptimizer(tf.keras.optimizers.Optimizer):
    """
    Гибридный оптимизатор: стохастический выбор между FW-обновлением и SGD.

    Для каждого тензора параметров `var` на каждом шаге:
      - с вероятностью fw_prob:
            FW:  var <- (1 - γ_k) * var + γ_k * s_k,
      - с вероятностью (1 - fw_prob):
            SGD: var <- var - η * grad.

    Параметры
    ---------
    constraint : ConstraintSet
        Множество S для FW-части (например, L2BallConstraint).
    fw_prob : float
        Вероятность применить FW-обновление к конкретному тензору
        на данном шаге.
    gamma : float
        Базовое значение γ (если use_diminishing_step=False).
    use_diminishing_step : bool
        Если True, используем γ_k = 2/(k+2) вместо константного.
    learning_rate : float
        Шаг для SGD-обновления (η).
    """

    def __init__(
        self,
        constraint: ConstraintSet,
        fw_prob: float = 0.3,
        gamma: float = 0.05,
        use_diminishing_step: bool = False,
        learning_rate: float = 1e-3,
        name: str = "HybridFrankWolfe",
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, **{"learning_rate": learning_rate, **kwargs})
        self._constraint = constraint
        self.fw_prob = float(fw_prob)
        self.gamma = float(gamma)
        self.use_diminishing_step = bool(use_diminishing_step)

        self._step = tf.Variable(
            0,
            dtype=tf.int64,
            trainable=False,
            name=f"{name}_step",
        )

    def build(self, var_list):
        super().build(var_list)

    def _current_gamma(self) -> tf.Tensor:
        if not self.use_diminishing_step:
            return tf.cast(self.gamma, tf.float32)
        k = tf.cast(self._step, tf.float32)
        gamma_k = 2.0 / (k + 2.0)
        gamma_k = tf.clip_by_value(gamma_k, 0.0, 1.0)
        return gamma_k

    def update_step(self, gradient, variable, learning_rate=None):
        if gradient is None:
            return

        lr = tf.cast(
            learning_rate if learning_rate is not None else self.learning_rate,
            variable.dtype,
        )
        grad = tf.cast(gradient, variable.dtype)

        # бросаем "монетку" per-variable
        u = tf.random.uniform((), dtype=tf.float32)
        fw_threshold = tf.cast(self.fw_prob, tf.float32)

        def fw_update():
            s_k = self._constraint.argmin(grad=grad, var=variable)
            gamma = tf.cast(self._current_gamma(), variable.dtype)
            new_var = (1.0 - gamma) * variable + gamma * s_k
            variable.assign(new_var)

        def sgd_update():
            variable.assign_sub(lr * grad)

        tf.cond(u < fw_threshold, fw_update, sgd_update)

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        self._step.assign_add(1)
        return super().apply_gradients(grads_and_vars, *args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "fw_prob": self.fw_prob,
                "gamma": self.gamma,
                "use_diminishing_step": self.use_diminishing_step,
                "constraint": tf.keras.saving.serialize_keras_object(
                    self._constraint
                ),
                "learning_rate": self.learning_rate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        constraint_cfg = config.pop("constraint")
        constraint = tf.keras.saving.deserialize_keras_object(constraint_cfg)
        return cls(constraint=constraint, **config)
