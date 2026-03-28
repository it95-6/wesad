from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path("data_wesad/WESAD")
CHEST_SAMPLE_RATE = 700
WINDOW_SECONDS = 30
OVERLAP_RATIO = 0.5
WINDOW_SIZE = CHEST_SAMPLE_RATE * WINDOW_SECONDS
STEP_SIZE = int(WINDOW_SIZE * (1.0 - OVERLAP_RATIO))


def load_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file, encoding="latin1")


def majority_label(labels: np.ndarray) -> int:
    labels = labels.astype(np.int64, copy=False)
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])


def create_subject_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_pickle(path)
    chest = data["signal"]["chest"]

    ecg = np.asarray(chest["ECG"]).reshape(-1)
    eda = np.asarray(chest["EDA"]).reshape(-1)
    labels = np.asarray(data["label"]).reshape(-1)

    if not (len(ecg) == len(eda) == len(labels)):
        raise ValueError(f"Signal length mismatch in {path}")

    features = np.column_stack((ecg, eda))
    windows: list[np.ndarray] = []
    window_labels: list[int] = []

    for start in range(0, len(features) - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        windows.append(features[start:end])
        window_labels.append(majority_label(labels[start:end]))

    if not windows:
        empty_x = np.empty((0, WINDOW_SIZE, 2), dtype=features.dtype)
        empty_y = np.empty((0,), dtype=np.int64)
        return empty_x, empty_y

    x = np.stack(windows).astype(np.float32, copy=False)
    y = np.asarray(window_labels, dtype=np.int64)
    return x, y


def create_dataset(root_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    subject_paths = sorted(root_dir.glob("S*/S*.pkl"))
    if not subject_paths:
        raise FileNotFoundError(f"No pickle files found under {root_dir}")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for subject_path in subject_paths:
        x_subject, y_subject = create_subject_windows(subject_path)
        print(
            f"{subject_path.stem}: windows={len(y_subject)}, "
            f"x_shape={x_subject.shape}, unique_labels={np.unique(y_subject)}"
        )
        if len(y_subject) == 0:
            continue
        x_parts.append(x_subject)
        y_parts.append(y_subject)

    if not x_parts:
        empty_x = np.empty((0, WINDOW_SIZE, 2), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int64)
        return empty_x, empty_y

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed-length windows from the WESAD dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing subject folders such as S2/S2.pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional .npz output path. If set, saves X and y.",
    )
    args = parser.parse_args()

    x, y = create_dataset(args.root)

    print(f"\nFinal dataset shapes: X={x.shape}, y={y.shape}")
    print(f"Window size: {WINDOW_SIZE} samples ({WINDOW_SECONDS} seconds)")
    print(f"Step size: {STEP_SIZE} samples ({WINDOW_SECONDS * (1.0 - OVERLAP_RATIO):.1f} seconds)")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    if args.output is not None:
        np.savez_compressed(args.output, X=x, y=y)
        print(f"Saved dataset to: {args.output}")


if __name__ == "__main__":
    main()
