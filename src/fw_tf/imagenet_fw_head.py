from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import applications, layers, models

from .constraints import L2BallConstraint
from .data import load_cifar10
from .optimizers import FrankWolfeOptimizer


@dataclass
class ImagenetFWConfig:
    batch_size: int = 128
    epochs: int = 10
    
    # размер выходного вектора фич Backbone
    feature_dim: int = 2048

    # радиус L2-шара и шаг FW
    fw_radius: float = 50.0
    fw_gamma: float = 0.02


def make_imagenet_backbone(
    input_shape: Tuple[int, int, int] = (32, 32, 3),
) -> tf.keras.Model:
    """
    Предобученный на ImageNet backbone (ResNet50) с замороженными весами.
    Мы ресайзим CIFAR-10 до 224x224, как ожидает ResNet.

    Возвращает модель, выдающую вектор признаков размера (feature_dim,).
    """
    resnet_input_shape = (224, 224, 3)
    base_model = applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=resnet_input_shape,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    # upsample CIFAR-10 до 224x224
    x = layers.Resizing(224, 224)(inputs)
    # предобработка под ResNet
    x = applications.resnet.preprocess_input(x)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)  # -> (batch, 2048)
    backbone = models.Model(inputs=inputs, outputs=x, name="resnet50_backbone")
    return backbone


def make_linear_head(
    feature_dim: int,
    num_classes: int = 10,
    name: str = "linear_head",
) -> tf.keras.Model:
    """
    Линейный классификатор: Dense(num_classes, softmax) над вектором фич.
    """
    inputs = layers.Input(shape=(feature_dim,))
    outputs = layers.Dense(num_classes, activation="softmax", name=name)(inputs)
    model = models.Model(inputs=inputs, outputs=outputs, name=f"{name}_model")
    return model


def extract_features(
    backbone: tf.keras.Model,
    x: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """
    Прогоняет все изображения через backbone и возвращает матрицу фич.
    """
    features = backbone.predict(x, batch_size=batch_size, verbose=1)
    return features


def train_linear_head_with_optimizer(
    optimizer: tf.keras.optimizers.Optimizer,
    x_feats_train: np.ndarray,
    y_train: np.ndarray,
    x_feats_val: np.ndarray,
    y_val: np.ndarray,
    cfg: ImagenetFWConfig,
    optimizer_name: str,
) -> Dict[str, float]:
    head = make_linear_head(feature_dim=cfg.feature_dim, num_classes=10)

    head.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\nTraining linear head with {optimizer_name} for {cfg.epochs} epochs")
    history = head.fit(
        x_feats_train,
        y_train,
        validation_data=(x_feats_val, y_val),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        verbose=2,
    )

    best_val_acc = float(max(history.history["val_accuracy"]))
    return {"best_val_accuracy": best_val_acc}


def run_imagenet_feature_experiment(cfg: ImagenetFWConfig) -> Dict[str, float]:
    """
    Эксперимент:
      1) Используем предобученный ResNet50 как фиксированный извлекатель признаков.
      2) Строим матрицы фич для train / test CIFAR-10.
      3) Обучаем чистый линейный классификатор:
         - Adam (baseline),
         - Frank-Wolfe (L2-шар) на тех же фичах.

    Возвращает словарь: имя оптимизатора -> best val accuracy.
    """
    (x_train, y_train, x_test, y_test) = load_cifar10()
    num_classes = 10

    print("Building frozen ResNet50 backbone...")
    backbone = make_imagenet_backbone(input_shape=x_train.shape[1:])

    print("Extracting features for train set...")
    x_feats_train = extract_features(backbone, x_train, batch_size=cfg.batch_size)
    print("Extracting features for test set...")
    x_feats_test = extract_features(backbone, x_test, batch_size=cfg.batch_size)

    # определяем фактическое feature_dim по backbone
    feature_dim = x_feats_train.shape[1]
    if cfg.feature_dim != feature_dim:
        print(
            f"[warn] cfg.feature_dim={cfg.feature_dim}, "
            f"but backbone produced {feature_dim}; using {feature_dim}"
        )
        cfg.feature_dim = feature_dim

    # Baseline: Adam
    adam_opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    adam_metrics = train_linear_head_with_optimizer(
        optimizer=adam_opt,
        x_feats_train=x_feats_train,
        y_train=y_train,
        x_feats_val=x_feats_test,
        y_val=y_test,
        cfg=cfg,
        optimizer_name="Adam",
    )

    # FW: L2 шар
    fw_constraint = L2BallConstraint(radius=cfg.fw_radius)
    fw_opt = FrankWolfeOptimizer(
        constraint=fw_constraint,
        gamma=cfg.fw_gamma,
    )
    fw_metrics = train_linear_head_with_optimizer(
        optimizer=fw_opt,
        x_feats_train=x_feats_train,
        y_train=y_train,
        x_feats_val=x_feats_test,
        y_val=y_test,
        cfg=cfg,
        optimizer_name=f"FrankWolfe(L2, R={cfg.fw_radius}, γ={cfg.fw_gamma})",
    )

    results: Dict[str, float] = {
        "adam": adam_metrics["best_val_accuracy"],
        "fw_l2": fw_metrics["best_val_accuracy"],
    }
    print("\n=== Imagenet feature experiment results ===")
    for name, acc in results.items():
        print(f"{name}: best val acc = {acc:.4f}")

    return results
