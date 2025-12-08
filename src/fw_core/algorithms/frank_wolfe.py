from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np

from fw_core.types import LinearMinimizationOracle, ObjectiveFunction, Vector

# Тип для правила шага:
# step_rule(k: int) -> float, где k — номер итерации, начиная с 0.
StepRule = Callable[[int], float]


class FWState(NamedTuple):
    """
    Состояние метода Франка–Вольфа на итерации k.

    Поля:
      - x: текущая точка x_k
      - f_value: значение f(x_k)
      - dual_gap: dual gap g(x_k) = <∇f(x_k), x_k - s_k>

    Dual gap для выпуклого случая даёт верхнюю оценку на субоптимальность:
        f(x_k) - f(x*) <= g(x_k),
    см. Jaggi (2013), §2.1.
    """

    x: Vector
    f_value: float
    dual_gap: float


def constant_step_size(_: int, value: float = 0.1) -> float:
    """
    Постоянный шаг: gamma_k = value.

    Параметры:
      - _: номер итерации (не используется)
      - value: константа в [0, 1]

    Замечание:
      - Для теоретических гарантий сходимости классический FW обычно
        использует убывающий шаг (см. diminishing_step_size),
        но постоянный шаг удобен для экспериментов.

    Документация по numpy:
      - https://numpy.org/doc/stable/
    """
    return value


def diminishing_step_size(k: int) -> float:
    """
    Классическое правило шага для метода Франка–Вольфа:
        gamma_k = 2 / (k + 2).

    Это правило даёт оценку сходимости порядка O(1 / k)
    для выпуклой гладкой функции с Lipschitz-градиентом.

    См.:
      - Jaggi, "Revisiting Frank-Wolfe: Projection-Free Sparse
        Convex Optimization", ICML 2013, §2.
      - Boyd & Vandenberghe, "Convex Optimization", §9.1.3.
    """
    return 2.0 / float(k + 2)


def frank_wolfe(
    f: ObjectiveFunction,
    oracle: LinearMinimizationOracle,
    x0: Vector,
    max_iter: int,
    step_rule: StepRule = diminishing_step_size,
    tol: float | None = None,
) -> list[FWState]:
    """
    Базовый метод Франка–Вольфа (Conditional Gradient Method).

    Задача:
        min_{x in S} f(x),
    где S — непустое компактное выпуклое множество,
    f — гладкая (обычно выпуклая) функция.

    Итерация k (k = 0, 1, 2, ...):
      1) s_k = argmin_{s in S} <∇f(x_k), s>
      2) x_{k+1} = (1 - gamma_k) * x_k + gamma_k * s_k

    Dual gap:
      g(x_k) = <∇f(x_k), x_k - s_k>

    Для выпуклого случая выполняется:
      f(x_k) - f(x*) <= g(x_k),
    см. Jaggi (2013), §2.1.

    Параметры:
      - f: ObjectiveFunction
          Объект с методами:
            value(x): float
            gradient(x): Vector
      - oracle: LinearMinimizationOracle
          Линейный оракул на множестве S.
      - x0: Vector
          Начальная точка x_0 ∈ S.
      - max_iter: int
          Максимальное число итераций.
      - step_rule: StepRule
          Правило выбора размера шага gamma_k = step_rule(k).
          Ожидается, что 0 <= gamma_k <= 1.
      - tol: float | None
          Допуск по dual gap. Если tol не None, то метод
          останавливается, когда g(x_k) <= tol.

    Возвращает:
      - trajectory: list[FWState]
          Список состояний [state_0, state_1, ..., state_T],
          где T <= max_iter (если сработал tol, T < max_iter),
          и state_k соответствует точке x_k.

    Теоретические ссылки:
      - Jaggi, "Revisiting Frank-Wolfe: Projection-Free Sparse Convex
        Optimization", ICML 2013.
      - Boyd & Vandenberghe, "Convex Optimization", §9.1.3.
    """
    trajectory: list[FWState] = []

    # Копируем x0, чтобы не модифицировать внешнюю ссылку.
    x: Vector = np.array(x0, dtype=np.float64)

    for k in range(max_iter):
        grad: Vector = f.gradient(x)
        s: Vector = oracle.argmin(grad)

        # Dual gap g(x_k) = <∇f(x_k), x_k - s_k>
        dual_gap = float(np.dot(grad, x - s))
        f_val = f.value(x)

        trajectory.append(FWState(x=x.copy(), f_value=f_val, dual_gap=dual_gap))

        if tol is not None and dual_gap <= tol:
            break

        gamma: float = step_rule(k)
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"step_rule returned gamma={gamma}, expected in [0, 1]")

        x = (1.0 - gamma) * x + gamma * s

    # Финальное состояние (если цикл завершился по max_iter, логично
    # сохранить ещё одну точку после последнего шага).
    grad = f.gradient(x)
    s = oracle.argmin(grad)
    dual_gap = float(np.dot(grad, x - s))
    f_val = f.value(x)
    trajectory.append(FWState(x=x.copy(), f_value=f_val, dual_gap=dual_gap))

    return trajectory
