from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from research.config import TrainConfig, ensure_output_dirs, get_path_config, set_seed
from research.dataset_wesad import (
    WESADWindowDataset,
    filter_labels,
    load_wesad_npz,
    prepare_splits,
    remap_labels,
    to_channel_first,
)
from research.models import SimpleWESADCNN

try:
    import seaborn as sns
except ImportError:
    sns = None


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * batch_x.size(0)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch_y.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def main() -> None:
    paths = get_path_config()
    cfg = TrainConfig()
    ensure_output_dirs(paths)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA is required but not available. In Colab, switch Runtime -> Change runtime type -> GPU."
        )
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    x, y = load_wesad_npz(str(paths.npz_path))
    x, y = filter_labels(x, y, cfg.labels_to_use)
    x = to_channel_first(x)
    y, _, index_to_label = remap_labels(y)

    split_data = prepare_splits(
        x=x,
        y=y,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )

    loader = DataLoader(
        WESADWindowDataset(split_data.x_test, split_data.y_test),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SimpleWESADCNN(
        in_channels=split_data.x_test.shape[1],
        num_classes=len(np.unique(split_data.y_test)),
    ).to(device)

    checkpoint = torch.load(paths.model_dir / f"{cfg.model_name}_best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device)

    print("=== Evaluation ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print("Classification report:")
    print(metrics["classification_report"])

    labels_sorted = [str(index_to_label[i]) for i in range(len(index_to_label))]
    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(
            metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels_sorted,
            yticklabels=labels_sorted,
        )
    else:
        plt.imshow(metrics["confusion_matrix"], cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels_sorted)), labels_sorted)
        plt.yticks(range(len(labels_sorted)), labels_sorted)
        for i in range(metrics["confusion_matrix"].shape[0]):
            for j in range(metrics["confusion_matrix"].shape[1]):
                plt.text(j, i, str(metrics["confusion_matrix"][i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("WESAD Test Confusion Matrix")
    plt.tight_layout()
    fig_path = paths.figure_dir / f"{cfg.model_name}_confusion_matrix.png"
    plt.savefig(fig_path, dpi=150)
    plt.show()

    with (paths.log_dir / f"{cfg.model_name}_evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "confusion_matrix": metrics["confusion_matrix"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Saved figure to: {fig_path}")


if __name__ == "__main__":
    main()
