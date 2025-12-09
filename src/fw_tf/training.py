from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorflow as tf

from .constraints import L2BallConstraint, LInfBallConstraint
from .data import kfold_splits, load_cifar10
from .model import make_cifar10_cnn, make_cifar10_backbone_with_head
from .optimizers import FrankWolfeOptimizer


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 10
    k_folds: int = 5
    random_state: int = 42
    # для двухшагового обучения
    backbone_epochs: int = 5
    fw_head_epochs: int = 5


def build_optimizer(name: str) -> tf.keras.optimizers.Optimizer:
    name = name.lower()
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=1e-3)
    if name == "fw_l2":
        constraint = L2BallConstraint(radius=100.0)
        return FrankWolfeOptimizer(constraint=constraint, gamma=0.05)
    if name == "fw_linf":
        constraint = LInfBallConstraint(radius=1.0)
        return FrankWolfeOptimizer(constraint=constraint, gamma=0.05)
    raise ValueError(f"Unknown optimizer: {name}")


# ===== старый вариант: оптимизатор на всех слоях (для SGD/Adam и как негативный пример FW) =====

def train_one_fold(
    optimizer_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> Dict[str, float]:
    model = make_cifar10_cnn()
    optimizer = build_optimizer(optimizer_name)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        verbose=2,
    )

    best_val_acc = float(max(history.history["val_accuracy"]))
    return {"best_val_accuracy": best_val_acc}


def run_experiments(
    optimizer_names: List[str],
    cfg: TrainConfig,
) -> Dict[str, List[float]]:
    x_train_full, y_train_full, x_test, y_test = load_cifar10()  # noqa: F841

    results: Dict[str, List[float]] = {name: [] for name in optimizer_names}

    for opt_name in optimizer_names:
        print(f"\n=== Optimizer: {opt_name} ===")
        for fold, ((x_tr, y_tr), (x_val, y_val)) in enumerate(
            kfold_splits(
                x_train_full,
                y_train_full,
                n_splits=cfg.k_folds,
                random_state=cfg.random_state,
            ),
            start=1,
        ):
            print(f"\nFold {fold}/{cfg.k_folds}")
            metrics = train_one_fold(
                optimizer_name=opt_name,
                x_train=x_tr,
                y_train=y_tr,
                x_val=x_val,
                y_val=y_val,
                cfg=cfg,
            )
            results[opt_name].append(metrics["best_val_accuracy"])
            print(f"Best val accuracy: {metrics['best_val_accuracy']:.4f}")

    return results


# ===== новый вариант: FW только на последнем слое (линейный классификатор поверх фиксированных фич) =====

def train_one_fold_fw_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
) -> Dict[str, float]:
    
    # строим модель и получаем ссылку на последний слой
    model, head = make_cifar10_backbone_with_head()
    assert isinstance(head, tf.keras.layers.Dense)

    # обучение backbone (все слои trainable) обычным Adam
    backbone_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=backbone_opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Pretraining backbone for {cfg.backbone_epochs} epochs with Adam")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=cfg.batch_size,
        epochs=cfg.backbone_epochs,
        verbose=2,
    )

    # замораживаем все слои, кроме head
    for layer in model.layers:
        layer.trainable = (layer is head)

    # этап FW: линейный классификатор на фиксированных фичах
    fw_constraint = L2BallConstraint(radius=10.0)
    fw_opt = FrankWolfeOptimizer(constraint=fw_constraint, gamma=0.1)

    model.compile(
        optimizer=fw_opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Training FW head for {cfg.fw_head_epochs} epochs")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        batch_size=cfg.batch_size,
        epochs=cfg.fw_head_epochs,
        verbose=2,
    )

    best_val_acc = float(max(history.history["val_accuracy"]))
    return {"best_val_accuracy": best_val_acc}


def run_experiments_fw_head(cfg: TrainConfig) -> Dict[str, List[float]]:
    """Эксперимент: FW только на последнем слое (backbone фиксированный)."""
    x_train_full, y_train_full, x_test, y_test = load_cifar10()  # noqa: F841

    results: Dict[str, List[float]] = {"fw_head_l2": []}

    print("\n=== Optimizer: FW on last layer (L2 ball) ===")
    for fold, ((x_tr, y_tr), (x_val, y_val)) in enumerate(
        kfold_splits(
            x_train_full,
            y_train_full,
            n_splits=cfg.k_folds,
            random_state=cfg.random_state,
        ),
        start=1,
    ):
        print(f"\nFold {fold}/{cfg.k_folds}")
        metrics = train_one_fold_fw_head(
            x_train=x_tr,
            y_train=y_tr,
            x_val=x_val,
            y_val=y_val,
            cfg=cfg,
        )
        results["fw_head_l2"].append(metrics["best_val_accuracy"])
        print(f"Best val accuracy (FW head): {metrics['best_val_accuracy']:.4f}")

    return results
