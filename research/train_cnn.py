from __future__ import annotations

import json
import sys
from pathlib import Path

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


def create_dataloaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    use_pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        WESADWindowDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        WESADWindowDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        WESADWindowDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    return train_loader, val_loader, test_loader


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            preds = torch.argmax(logits, dim=1)
            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(batch_y.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return avg_loss, accuracy, macro_f1


@torch.no_grad()
def evaluate_model(
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
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
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
    else:
        print("GPU is not available. Training will run on CPU.")

    x, y = load_wesad_npz(str(paths.npz_path))
    print(f"Loaded raw NPZ: X={x.shape}, y={y.shape}")

    x, y = filter_labels(x, y, cfg.labels_to_use)
    x = to_channel_first(x)
    y, label_to_index, index_to_label = remap_labels(y)

    split_data = prepare_splits(
        x=x,
        y=y,
        val_size=cfg.val_size,
        test_size=cfg.test_size,
        random_state=cfg.seed,
    )

    print(f"Train: {split_data.x_train.shape}, {split_data.y_train.shape}")
    print(f"Val  : {split_data.x_val.shape}, {split_data.y_val.shape}")
    print(f"Test : {split_data.x_test.shape}, {split_data.y_test.shape}")

    train_loader, val_loader, test_loader = create_dataloaders(
        split_data.x_train,
        split_data.y_train,
        split_data.x_val,
        split_data.y_val,
        split_data.x_test,
        split_data.y_test,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    model = SimpleWESADCNN(
        in_channels=split_data.x_train.shape[1],
        num_classes=len(np.unique(split_data.y_train)),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    use_amp = cfg.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_metric = -1.0
    best_model_path = paths.model_dir / f"{cfg.model_name}_best.pt"
    history: list[dict] = []

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc, train_f1 = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            optimizer=optimizer,
        )
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        if val_metrics[cfg.save_metric] > best_metric:
            best_metric = val_metrics[cfg.save_metric]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_to_index": label_to_index,
                    "index_to_label": index_to_label,
                    "config": cfg.__dict__,
                    "input_shape": split_data.x_train.shape[1:],
                },
                best_model_path,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        print(
            f"[Epoch {epoch:02d}/{cfg.num_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    print("\n=== Test Metrics ===")
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Macro F1 : {test_metrics['macro_f1']:.4f}")
    print("Confusion matrix:")
    print(test_metrics["confusion_matrix"])
    print("Classification report:")
    print(test_metrics["classification_report"])

    with (paths.log_dir / f"{cfg.model_name}_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (paths.log_dir / f"{cfg.model_name}_test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": float(test_metrics["accuracy"]),
                "macro_f1": float(test_metrics["macro_f1"]),
                "confusion_matrix": test_metrics["confusion_matrix"].tolist(),
            },
            f,
            indent=2,
        )
    print(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
