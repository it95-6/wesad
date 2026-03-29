from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]


def load_wesad_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return data["X"], data["y"]


def filter_labels(
    x: np.ndarray,
    y: np.ndarray,
    labels_to_use: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, labels_to_use)
    return x[mask], y[mask]


def to_channel_first(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected X to be 3D, got {x.shape}")
    return np.transpose(x, (0, 2, 1)).astype(np.float32, copy=False)


def remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict[int, int], dict[int, int]]:
    unique_labels = sorted(np.unique(y).tolist())
    original_to_new = {label: idx for idx, label in enumerate(unique_labels)}
    new_to_original = {idx: label for label, idx in original_to_new.items()}
    y_new = np.array([original_to_new[int(label)] for label in y], dtype=np.int64)
    return y_new, original_to_new, new_to_original


def prepare_splits(
    x: np.ndarray,
    y: np.ndarray,
    val_size: float,
    test_size: float,
    random_state: int,
) -> SplitData:
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_ratio_in_train_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_ratio_in_train_val,
        random_state=random_state,
        stratify=y_train_val,
    )

    class_names = [str(label) for label in sorted(np.unique(y).tolist())]
    return SplitData(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        class_names=class_names,
    )


class WESADWindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]
