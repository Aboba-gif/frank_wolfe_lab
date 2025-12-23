from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def _mean_std_over_folds(
    histories: List[Dict[str, List[float]]],
    key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    histories: список history-словарей (как keras_history.history) по фолдам.
    key: имя метрики, например "loss" или "val_accuracy".

    Возвращает:
      mean: среднее по фолдам для каждой эпохи,
      std: стандартное отклонение по фолдам.
    """
    arrays = []
    for h in histories:
        if key in h:
            arrays.append(np.array(h[key], dtype=float))

    if not arrays:
        raise ValueError(f"No histories contain key={key}")

    min_len = min(a.shape[0] for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    stacked = np.stack(arrays, axis=0)  # (n_folds, n_epochs)
    return stacked.mean(axis=0), stacked.std(axis=0)


# ======================= ALL LAYERS: SGD / Adam / FW / Hybrid =======================


def plot_all_layers_curves() -> None:
    """
    Строит графики для эксперимента --mode all_layers:
      - train loss vs epoch
      - val accuracy vs epoch
      для оптимизаторов: sgd, adam, fw_l2, hybrid_fw_l2.

    Ожидает файл results/all_layers_histories.json.
    """
    json_path = RESULTS_DIR / "all_layers_histories.json"
    if not json_path.exists():
        print(f"[plot_all_layers_curves] {json_path} not found, skipping")
        return

    with json_path.open() as f:
        # dict: opt_name -> list[history_dict_per_fold]
        all_histories: Dict[str, List[Dict[str, List[float]]]] = json.load(f)

    optimizers_to_plot = ["sgd", "adam", "fw_l2", "hybrid_fw_l2"]
    colors = {
        "sgd": "tab:blue",
        "adam": "tab:orange",
        "fw_l2": "tab:red",
        "hybrid_fw_l2": "tab:green",
    }

    plt.figure(figsize=(10, 4))

    # --- Train loss ---
    plt.subplot(1, 2, 1)
    for name in optimizers_to_plot:
        if name not in all_histories:
            continue
        try:
            mean, std = _mean_std_over_folds(all_histories[name], key="loss")
        except ValueError:
            continue
        epochs = np.arange(1, len(mean) + 1)
        plt.plot(epochs, mean, label=name, color=colors.get(name))
        plt.fill_between(
            epochs, mean - std, mean + std, color=colors.get(name), alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Train loss vs epoch (all layers)")
    plt.legend()

    # --- Val accuracy ---
    plt.subplot(1, 2, 2)
    for name in optimizers_to_plot:
        if name not in all_histories:
            continue
        try:
            mean, std = _mean_std_over_folds(all_histories[name], key="val_accuracy")
        except ValueError:
            continue
        epochs = np.arange(1, len(mean) + 1)
        plt.plot(epochs, mean, label=name, color=colors.get(name))
        plt.fill_between(
            epochs, mean - std, mean + std, color=colors.get(name), alpha=0.2
        )
    plt.xlabel("Epoch")
    plt.ylabel("Val accuracy")
    plt.title("Val accuracy vs epoch (all layers)")
    plt.legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / "all_layers_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot_all_layers_curves] Saved {out_path}")


# ======================= FW HEAD ONLY: backbone Adam + FW on head =======================


def plot_fw_head_curves() -> None:
    """
    Строит графики для эксперимента --mode fw_head.

    Использует results/fw_head_histories.json, в котором структура:
      {
        "fw_head_l2": [
            {
              "backbone": {metric -> [..epochs..]},
              "fw_head":  {metric -> [..epochs..]},
            },
            ...
        ]
      }

    Графики:
      - val accuracy vs "глобальная эпоха" (Adam-backbone + FW-head подряд)
      - отдельно train/val для FW-фазы (усреднённые по фолдам).
    """
    json_path = RESULTS_DIR / "fw_head_histories.json"
    if not json_path.exists():
        print(f"[plot_fw_head_curves] {json_path} not found, skipping")
        return

    with json_path.open() as f:
        data = json.load(f)

    folds = data.get("fw_head_l2", [])
    if not folds:
        print("[plot_fw_head_curves] No fw_head_l2 data, skipping")
        return

    backbone_histories = [fold["backbone"] for fold in folds]
    fw_histories = [fold["fw_head"] for fold in folds]

    # Глобальная кривая val_accuracy: сначала Adam, потом FW
    # Для визуализации берем среднее по фолдам
    backbone_val_mean, _ = _mean_std_over_folds(
        backbone_histories, key="val_accuracy"
    )
    fw_val_mean, _ = _mean_std_over_folds(fw_histories, key="val_accuracy")

    n_backbone = len(backbone_val_mean)
    n_fw = len(fw_val_mean)

    global_epochs = np.arange(1, n_backbone + n_fw + 1)
    global_val = np.concatenate([backbone_val_mean, fw_val_mean])

    plt.figure(figsize=(8, 4))
    plt.plot(global_epochs, global_val, label="val_accuracy", color="tab:blue")
    plt.axvline(
        x=n_backbone + 0.5,
        color="k",
        linestyle="--",
        label="switch to FW on head",
    )
    plt.xlabel("Epoch (Adam backbone + FW head)")
    plt.ylabel("Val accuracy")
    plt.title("FW on last layer: backbone (Adam) + head (FW)")
    plt.legend()
    out_path = RESULTS_DIR / "fw_head_global_val_acc.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot_fw_head_curves] Saved {out_path}")

    # Только FW-фаза: train loss и val accuracy, усреднённые по фолдам
    fw_loss_mean, fw_loss_std = _mean_std_over_folds(fw_histories, key="loss")
    fw_val_mean, fw_val_std = _mean_std_over_folds(fw_histories, key="val_accuracy")
    fw_epochs = np.arange(1, len(fw_loss_mean) + 1)

    plt.figure(figsize=(10, 4))
    # train loss
    plt.subplot(1, 2, 1)
    plt.plot(fw_epochs, fw_loss_mean, color="tab:red", label="train loss (FW)")
    plt.fill_between(
        fw_epochs,
        fw_loss_mean - fw_loss_std,
        fw_loss_mean + fw_loss_std,
        color="tab:red",
        alpha=0.2,
    )
    plt.xlabel("Epoch (FW head)")
    plt.ylabel("Train loss")
    plt.title("FW head: train loss")
    plt.legend()

    # val accuracy
    plt.subplot(1, 2, 2)
    plt.plot(
        fw_epochs,
        fw_val_mean,
        color="tab:green",
        label="val accuracy (FW)",
    )
    plt.fill_between(
        fw_epochs,
        fw_val_mean - fw_val_std,
        fw_val_mean + fw_val_std,
        color="tab:green",
        alpha=0.2,
    )
    plt.xlabel("Epoch (FW head)")
    plt.ylabel("Val accuracy")
    plt.title("FW head: val accuracy")
    plt.legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / "fw_head_fw_phase_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot_fw_head_curves] Saved {out_path}")


# ======================= IMAGENET FEATURES: linear head Adam vs FW =======================


def plot_imagenet_curves() -> None:
    """
    Строит графики для эксперимента --mode imagenet.

    Ожидает results/imagenet_histories.json со структурой:
      {
        "adam": {metric -> [..]},
        "fw_l2": {metric -> [..]},
      }

    Рисуем:
      - train loss vs epoch
      - val accuracy vs epoch
    """
    json_path = RESULTS_DIR / "imagenet_histories.json"
    if not json_path.exists():
        print(f"[plot_imagenet_curves] {json_path} not found, skipping")
        return

    with json_path.open() as f:
        histories: Dict[str, Dict[str, List[float]]] = json.load(f)

    methods = ["adam", "fw_l2"]
    colors = {"adam": "tab:blue", "fw_l2": "tab:red"}

    plt.figure(figsize=(10, 4))

    # Train loss
    plt.subplot(1, 2, 1)
    for name in methods:
        if name not in histories:
            continue
        h = histories[name]
        if "loss" not in h:
            continue
        loss = np.array(h["loss"], dtype=float)
        epochs = np.arange(1, len(loss) + 1)
        plt.plot(epochs, loss, label=name, color=colors.get(name))
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Imagenet features: train loss")
    plt.legend()

    # Val accuracy
    plt.subplot(1, 2, 2)
    for name in methods:
        if name not in histories:
            continue
        h = histories[name]
        if "val_accuracy" not in h:
            continue
        val_acc = np.array(h["val_accuracy"], dtype=float)
        epochs = np.arange(1, len(val_acc) + 1)
        plt.plot(epochs, val_acc, label=name, color=colors.get(name))
    plt.xlabel("Epoch")
    plt.ylabel("Val accuracy")
    plt.title("Imagenet features: val accuracy")
    plt.legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / "imagenet_curves.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot_imagenet_curves] Saved {out_path}")


def main() -> None:
    plot_all_layers_curves()
    plot_fw_head_curves()
    plot_imagenet_curves()


if __name__ == "__main__":
    main()
