from __future__ import annotations

"""
TensorFlow/Keras utilities for experimenting with Frank–Wolfe optimizers.

Public API:
- Constraints: L2BallConstraint, LInfBallConstraint.
- Optimizers: FrankWolfeOptimizer, HybridFrankWolfeOptimizer.
- Models: make_cifar10_cnn, make_cifar10_backbone_with_head.
- Data helpers: load_cifar10, kfold_splits.
- Training routines for CIFAR-10: run_experiments, run_experiments_fw_head,
  run_imagenet_feature_experiment.
"""

from .constraints import ConstraintSet, L2BallConstraint, LInfBallConstraint
from .data import load_cifar10, kfold_splits
from .model import make_cifar10_cnn, make_cifar10_backbone_with_head
from .optimizers import FrankWolfeOptimizer, HybridFrankWolfeOptimizer
from .training import (
    TrainConfig,
    run_experiments,
    run_experiments_fw_head,
)
from .imagenet_fw_head import (
    ImagenetFWConfig,
    run_imagenet_feature_experiment,
)

__all__ = [
    "ConstraintSet",
    "L2BallConstraint",
    "LInfBallConstraint",
    "FrankWolfeOptimizer",
    "HybridFrankWolfeOptimizer",
    "make_cifar10_cnn",
    "make_cifar10_backbone_with_head",
    "load_cifar10",
    "kfold_splits",
    "TrainConfig",
    "run_experiments",
    "run_experiments_fw_head",
    "ImagenetFWConfig",
    "run_imagenet_feature_experiment",
]
