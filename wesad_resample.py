from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("data_wesad/WESAD")
DEFAULT_PKL_PATH = Path("data_wesad/WESAD/S2/S2.pkl")
DEFAULT_OUTPUT_DIR = Path("data_wesad/resampled")
CHEST_ECG_RATE = 700.0
WRIST_EDA_RATE = 4.0
VALID_TARGET_RATES = (4.0, 10.0)
VALID_OUTPUT_FORMATS = ("csv", "npz", "both")


def load_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file, encoding="latin1")


def build_signal_frame(values: np.ndarray, sample_rate_hz: float, name: str) -> pd.DataFrame:
    signal = np.asarray(values).reshape(-1).astype(np.float64, copy=False)
    time_seconds = np.arange(signal.shape[0], dtype=np.float64) / sample_rate_hz
    return pd.DataFrame({"time_s": time_seconds, name: signal})


def interpolate_to_common_grid(
    frame: pd.DataFrame,
    value_name: str,
    target_index: pd.Index,
    *,
    interpolation: str,
) -> pd.Series:
    source = frame.drop_duplicates(subset="time_s").set_index("time_s")[value_name]
    aligned = source.reindex(source.index.union(target_index)).sort_index()
    interpolated = aligned.interpolate(method=interpolation).reindex(target_index)
    return interpolated


def resample_subject(path: Path, target_rate_hz: float) -> pd.DataFrame:
    data = load_pickle(path)
    chest = data["signal"]["chest"]
    wrist = data["signal"]["wrist"]

    ecg = np.asarray(chest["ECG"]).reshape(-1)
    eda = np.asarray(wrist["EDA"]).reshape(-1)
    labels = np.asarray(data["label"]).reshape(-1)

    ecg_frame = build_signal_frame(ecg, CHEST_ECG_RATE, "ECG")
    eda_frame = build_signal_frame(eda, WRIST_EDA_RATE, "EDA")
    label_frame = build_signal_frame(labels, CHEST_ECG_RATE, "label")

    start_time = max(
        ecg_frame["time_s"].iloc[0],
        eda_frame["time_s"].iloc[0],
        label_frame["time_s"].iloc[0],
    )
    end_time = min(
        ecg_frame["time_s"].iloc[-1],
        eda_frame["time_s"].iloc[-1],
        label_frame["time_s"].iloc[-1],
    )
    step = 1.0 / target_rate_hz

    common_time = np.arange(start_time, end_time + (step * 0.5), step, dtype=np.float64)
    target_index = pd.Index(common_time, name="time_s")

    ecg_resampled = interpolate_to_common_grid(
        ecg_frame, "ECG", target_index, interpolation="index"
    ).astype(np.float32)
    eda_resampled = interpolate_to_common_grid(
        eda_frame, "EDA", target_index, interpolation="index"
    ).astype(np.float32)

    label_source = label_frame.drop_duplicates(subset="time_s").set_index("time_s")["label"]
    label_resampled = (
        label_source.reindex(target_index, method="nearest")
        .astype(np.int64)
    )

    return pd.DataFrame(
        {
            "time_s": common_time,
            "ECG": ecg_resampled.to_numpy(),
            "EDA": eda_resampled.to_numpy(),
            "label": label_resampled.to_numpy(),
        }
    )


def save_resampled(
    frame: pd.DataFrame,
    subject_id: str,
    output_dir: Path,
    output_format: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    if output_format in {"csv", "both"}:
        csv_path = output_dir / f"{subject_id}_resampled.csv"
        frame.to_csv(csv_path, index=False)
        saved_paths.append(csv_path)

    if output_format in {"npz", "both"}:
        npz_path = output_dir / f"{subject_id}_resampled.npz"
        np.savez_compressed(
            npz_path,
            time_s=frame["time_s"].to_numpy(dtype=np.float64),
            X=frame[["ECG", "EDA"]].to_numpy(dtype=np.float32),
            y=frame["label"].to_numpy(dtype=np.int64),
            ECG=frame["ECG"].to_numpy(dtype=np.float32),
            EDA=frame["EDA"].to_numpy(dtype=np.float32),
            label=frame["label"].to_numpy(dtype=np.int64),
        )
        saved_paths.append(npz_path)

    return saved_paths


def process_subject(path: Path, target_rate_hz: float, output_dir: Path, output_format: str) -> None:
    frame = resample_subject(path, target_rate_hz)
    saved_paths = save_resampled(frame, path.stem, output_dir, output_format)

    print(
        f"{path.stem}: resampled_shape={frame.shape}, "
        f"time_range=({frame['time_s'].iloc[0]:.2f}, {frame['time_s'].iloc[-1]:.2f}), "
        f"labels={np.unique(frame['label'])}"
    )
    for saved_path in saved_paths:
        print(f"  saved: {saved_path}")


def process_all_subjects(root_dir: Path, target_rate_hz: float, output_dir: Path, output_format: str) -> None:
    subject_paths = sorted(root_dir.glob("S*/S*.pkl"))
    if not subject_paths:
        raise FileNotFoundError(f"No pickle files found under {root_dir}")

    print(f"Found {len(subject_paths)} subject files under {root_dir}")
    for subject_path in subject_paths:
        process_subject(subject_path, target_rate_hz, output_dir, output_format)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample WESAD ECG and EDA to a common time base.")
    parser.add_argument(
        "--pkl",
        type=Path,
        default=None,
        help="Optional path to a single WESAD subject pickle file. If omitted, all subjects are processed.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root directory containing subject folders such as S2/S2.pkl.",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=4.0,
        choices=VALID_TARGET_RATES,
        help="Target sampling rate in Hz. Supported values: 4 or 10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where resampled CSV/NPZ files are written.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=VALID_OUTPUT_FORMATS,
        help="Output format: csv, npz, or both.",
    )
    args = parser.parse_args()

    if args.pkl is not None:
        process_subject(args.pkl, args.target_rate, args.output_dir, args.format)
    else:
        process_all_subjects(args.root, args.target_rate, args.output_dir, args.format)


if __name__ == "__main__":
    main()
