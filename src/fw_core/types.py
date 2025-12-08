from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

# Базовый тип для векторов (и в дальнейшем можно будет интерпретировать его
# и как матрицы / тензоры, если потребуется).
Vector: TypeAlias = NDArray[np.float64]
Matrix: TypeAlias = NDArray[np.float64]


class ObjectiveFunction(Protocol):
    """
    Дифференцируемая функция f(x) с методами:
      - value(x): float  — значение f(x)
      - gradient(x): Vector — градиент ∇f(x)

    Предполагается, что:
      - область определения f — подмножество R^n,
      - value(x) и gradient(x) согласованы по размерности x.
    """

    def value(self, x: Vector) -> float:
        ...

    def gradient(self, x: Vector) -> Vector:
        ...


class LinearMinimizationOracle(Protocol):
    """
    Линейный оракул на множестве S.

    По вектору c возвращает точку
        argmin_{s in S} <c, s>,

    где <·,·> — стандартное евклидово скалярное произведение.
    """

    def argmin(self, c: Vector) -> Vector:
        ...
