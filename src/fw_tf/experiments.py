import argparse
from typing import List

from . import (
    TrainConfig,
    run_experiments,
    run_experiments_fw_head,
    ImagenetFWConfig,
    run_imagenet_feature_experiment,
)


def main():
    parser = argparse.ArgumentParser(description="Frank–Wolfe experiments on CIFAR-10")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all_layers", "fw_head", "imagenet"],
        default="all_layers",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs for main training",
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=3,
        help="Number of folds for cross-validation",
    )
    args = parser.parse_args()

    if args.mode == "all_layers":
        cfg = TrainConfig(epochs=args.epochs, k_folds=args.k_folds)
        optimizers: List[str] = ["sgd", "adam", "fw_l2", "hybrid_fw_l2"]
        results = run_experiments(optimizer_names=optimizers, cfg=cfg)
        print("\n=== Results: all layers ===")
        for name, vals in results.items():
            mean_acc = sum(vals) / len(vals)
            print(f"{name}: {vals} -> mean={mean_acc:.4f}")

    elif args.mode == "fw_head":
        cfg = TrainConfig(
            k_folds=args.k_folds,
            backbone_epochs=5,
            fw_head_epochs=5,
        )
        results = run_experiments_fw_head(cfg=cfg)
        print("\n=== Results: FW head only ===")
        for name, vals in results.items():
            mean_acc = sum(vals) / len(vals)
            print(f"{name}: {vals} -> mean={mean_acc:.4f}")

    elif args.mode == "imagenet":
        cfg = ImagenetFWConfig(epochs=args.epochs)
        results = run_imagenet_feature_experiment(cfg=cfg)
        print("\n=== Results: Imagenet features experiment ===")
        for name, acc in results.items():
            print(f"{name}: best val acc = {acc:.4f}")


if __name__ == "__main__":
    main()
