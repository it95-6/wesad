from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wesad_windowing import DEFAULT_ROOT, create_dataset


DEFAULT_LABELS = (1, 2, 3, 4)
DEFAULT_INPUT_NPZ = Path("wesad_windows.npz")
MODEL_NAMES = ("logreg", "rf")


def load_windowed_data(root: Path, input_npz: Path | None) -> tuple[np.ndarray, np.ndarray]:
    if input_npz is not None:
        data = np.load(input_npz)
        return data["X"], data["y"]
    return create_dataset(root)


def extract_stat_features(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3 or x.shape[2] != 2:
        raise ValueError(f"Expected X to have shape (num_windows, window, 2), got {x.shape}")

    ecg = x[:, :, 0]
    eda = x[:, :, 1]

    feature_blocks = []
    for signal in (ecg, eda):
        feature_blocks.extend(
            [
                signal.mean(axis=1),
                signal.std(axis=1),
                signal.min(axis=1),
                signal.max(axis=1),
                np.median(signal, axis=1),
                np.percentile(signal, 25, axis=1),
                np.percentile(signal, 75, axis=1),
            ]
        )

    features = np.column_stack(feature_blocks).astype(np.float32, copy=False)
    return features


def filter_labels(x: np.ndarray, y: np.ndarray, labels_to_keep: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(y, labels_to_keep)
    return x[mask], y[mask]


def build_model(model_name: str, random_state: int) -> Pipeline | RandomForestClassifier:
    if model_name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple baseline classifier on WESAD windowed data.")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing WESAD subject folders. Used when --input-npz is not provided.",
    )
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=DEFAULT_INPUT_NPZ,
        help="Precomputed windowed dataset (.npz) containing X and y.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset used for test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--include-all-labels",
        action="store_true",
        help="Use all labels instead of the default experimental labels 1,2,3,4.",
    )
    args = parser.parse_args()

    x, y = load_windowed_data(args.root, args.input_npz)
    print(f"Loaded windowed data: X={x.shape}, y={y.shape}")

    if not args.include_all_labels:
        x, y = filter_labels(x, y, DEFAULT_LABELS)
        print(f"Filtered labels to {DEFAULT_LABELS}: X={x.shape}, y={y.shape}")
    else:
        print(f"Using all labels: {np.unique(y)}")

    if len(y) == 0:
        raise ValueError("No samples remain after label filtering.")

    features = extract_stat_features(x)
    print(f"Stat feature matrix shape: {features.shape}")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"\nTrain shape: X={x_train.shape}, y={y_train.shape}")
    print(f"Test shape: X={x_test.shape}, y={y_test.shape}")

    for model_name in MODEL_NAMES:
        model = build_model(model_name, args.random_state)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))

        print("\n" + "=" * 60)
        print(f"Model: {model_name}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Macro F1 : {f1_macro:.4f}")
        print("Confusion matrix:")
        print(cm)
        print("Classification report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
