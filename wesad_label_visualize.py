from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PKL_PATH = Path("data_wesad/WESAD/S2/S2.pkl")
TARGET_LABELS = (1, 2, 3)
CHEST_SAMPLE_RATE = 700.0


def load_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file, encoding="latin1")


def find_label_segments(labels: np.ndarray, target: int) -> list[tuple[int, int]]:
    mask = labels == target
    segments: list[tuple[int, int]] = []
    start: int | None = None

    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append((start, index - 1))
            start = None

    if start is not None:
        segments.append((start, len(labels) - 1))

    return segments


def choose_segment(segments: list[tuple[int, int]]) -> tuple[int, int]:
    if not segments:
        raise ValueError("No segment found for the requested label.")
    return max(segments, key=lambda segment: segment[1] - segment[0] + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize WESAD ECG/EDA segments for labels 1, 2, and 3.")
    parser.add_argument(
        "--pkl",
        type=Path,
        default=DEFAULT_PKL_PATH,
        help="Path to the WESAD pickle file (default: data_wesad/WESAD/S2/S2.pkl).",
    )
    args = parser.parse_args()

    data = load_pickle(args.pkl)
    labels = np.asarray(data["label"])
    chest = data["signal"]["chest"]
    ecg = np.asarray(chest["ECG"]).squeeze()
    eda = np.asarray(chest["EDA"]).squeeze()

    if len(labels) != len(ecg) or len(labels) != len(eda):
        raise ValueError("label, ECG, and EDA must have the same number of samples.")

    fig, axes = plt.subplots(len(TARGET_LABELS), 1, figsize=(16, 12), constrained_layout=True)
    fig.suptitle(f"WESAD {args.pkl.stem}: ECG and EDA for labels 1, 2, 3", fontsize=16)

    if len(TARGET_LABELS) == 1:
        axes = [axes]

    for axis, label_value in zip(axes, TARGET_LABELS):
        segments = find_label_segments(labels, label_value)
        start, end = choose_segment(segments)
        segment_length = end - start + 1
        time_seconds = np.arange(segment_length) / CHEST_SAMPLE_RATE

        ecg_segment = ecg[start : end + 1]
        eda_segment = eda[start : end + 1]

        print(
            f"label={label_value} segment_count={len(segments)} "
            f"chosen_segment=({start}, {end}) duration_sec={segment_length / CHEST_SAMPLE_RATE:.2f}"
        )

        axis.plot(time_seconds, ecg_segment, color="tab:blue", linewidth=0.8)
        axis.set_title(f"Label {label_value}: samples {start}-{end}")
        axis.set_ylabel("ECG", color="tab:blue")
        axis.tick_params(axis="y", labelcolor="tab:blue")
        axis.grid(True, alpha=0.3)

        eda_axis = axis.twinx()
        eda_axis.plot(time_seconds, eda_segment, color="tab:orange", linewidth=1.0, alpha=0.85)
        eda_axis.set_ylabel("EDA", color="tab:orange")
        eda_axis.tick_params(axis="y", labelcolor="tab:orange")

        axis.set_xlabel("Time from segment start [s]")

    plt.show()


if __name__ == "__main__":
    main()
