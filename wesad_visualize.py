from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PKL_PATH = Path("data_wesad/WESAD/S2/S2.pkl")
SAMPLE_COUNT = 1000


def load_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file, encoding="latin1")


def print_structure(data: dict) -> None:
    print("=== WESAD pickle structure ===")
    print(f"top-level keys: {sorted(data.keys())}")

    for key, value in data.items():
        if isinstance(value, dict):
            print(f"\n[{key}]")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    print(f"  {subkey}: dict")
                    for inner_key, inner_value in subvalue.items():
                        shape = getattr(inner_value, "shape", None)
                        dtype = getattr(inner_value, "dtype", None)
                        print(f"    - {inner_key}: shape={shape}, dtype={dtype}")
                else:
                    shape = getattr(subvalue, "shape", None)
                    dtype = getattr(subvalue, "dtype", None)
                    print(f"  {subkey}: shape={shape}, dtype={dtype}")
        else:
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            print(f"\n[{key}] shape={shape}, dtype={dtype}, type={type(value).__name__}")


def read_empatica_csv(path: Path) -> tuple[np.ndarray, float | None, float | None]:
    with path.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()
        second_line = file.readline().strip()

    start_time = float(first_line.split(",")[0]) if first_line else None
    sample_rate = float(second_line.split(",")[0]) if second_line else None
    values = np.loadtxt(path, delimiter=",", skiprows=2)
    return np.atleast_2d(values) if values.ndim == 2 else np.asarray(values), start_time, sample_rate


def read_ibi_csv(path: Path) -> tuple[np.ndarray, float | None]:
    with path.open("r", encoding="utf-8") as file:
        first_line = file.readline().strip()

    start_time = float(first_line.split(",")[0]) if first_line else None
    values = np.loadtxt(path, delimiter=",", skiprows=1)
    values = np.atleast_2d(values)
    return values, start_time


def squeeze_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal)
    if signal.ndim == 2 and signal.shape[1] == 1:
        return signal[:, 0]
    return signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize WESAD S2 signals from pickle/CSV data.")
    parser.add_argument(
        "--pkl",
        type=Path,
        default=DEFAULT_PKL_PATH,
        help="Path to the WESAD pickle file (default: data_wesad/WESAD/S2/S2.pkl).",
    )
    args = parser.parse_args()

    pkl_path = args.pkl
    subject_dir = pkl_path.parent
    e4_dir = subject_dir / f"{pkl_path.stem}_E4_Data"

    data = load_pickle(pkl_path)
    print_structure(data)

    wrist = data["signal"]["wrist"]
    labels = np.asarray(data["label"])[:SAMPLE_COUNT]

    acc = np.asarray(wrist["ACC"])[:SAMPLE_COUNT]
    bvp = squeeze_signal(wrist["BVP"])[:SAMPLE_COUNT]
    eda = squeeze_signal(wrist["EDA"])[:SAMPLE_COUNT]
    temp = squeeze_signal(wrist["TEMP"])[:SAMPLE_COUNT]

    hr, hr_start, hr_rate = read_empatica_csv(e4_dir / "HR.csv")
    hr = squeeze_signal(hr)[:SAMPLE_COUNT]

    ibi, ibi_start = read_ibi_csv(e4_dir / "IBI.csv")
    ibi = ibi[:SAMPLE_COUNT]

    print("\n=== Signals used for plotting ===")
    print(f"ACC (pickle wrist): {acc.shape}")
    print(f"BVP (pickle wrist): {bvp.shape}")
    print(f"EDA (pickle wrist): {eda.shape}")
    print(f"TEMP (pickle wrist): {temp.shape}")
    print(f"HR (csv): {hr.shape}, start_time={hr_start}, sample_rate={hr_rate}")
    print(f"IBI (csv): {ibi.shape}, start_time={ibi_start}")
    print(f"label (pickle): {labels.shape}, unique={np.unique(labels)}")

    fig, axes = plt.subplots(7, 1, figsize=(16, 20), sharex=False)
    fig.suptitle(f"WESAD {pkl_path.stem}: first {SAMPLE_COUNT} samples", fontsize=16)

    acc_index = np.arange(len(acc))
    axes[0].plot(acc_index, acc[:, 0], label="ACC_X", linewidth=0.9)
    axes[0].plot(acc_index, acc[:, 1], label="ACC_Y", linewidth=0.9)
    axes[0].plot(acc_index, acc[:, 2], label="ACC_Z", linewidth=0.9)
    axes[0].set_title("ACC (wrist, pickle)")
    axes[0].set_ylabel("value")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.arange(len(bvp)), bvp, color="tab:red", linewidth=0.9)
    axes[1].set_title("BVP (wrist, pickle)")
    axes[1].set_ylabel("value")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(np.arange(len(eda)), eda, color="tab:orange", linewidth=0.9)
    axes[2].set_title("EDA (wrist, pickle)")
    axes[2].set_ylabel("value")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(np.arange(len(hr)), hr, color="tab:green", linewidth=0.9)
    axes[3].set_title("HR (E4 csv)")
    axes[3].set_ylabel("bpm")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(np.arange(len(ibi)), ibi[:, 1], color="tab:purple", linewidth=0.9)
    axes[4].set_title("IBI (E4 csv)")
    axes[4].set_ylabel("seconds")
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(np.arange(len(temp)), temp, color="tab:brown", linewidth=0.9)
    axes[5].set_title("TEMP (wrist, pickle)")
    axes[5].set_ylabel("degC")
    axes[5].grid(True, alpha=0.3)

    axes[6].step(np.arange(len(labels)), labels, where="post", color="black", linewidth=1.0)
    axes[6].set_title(f"Label (pickle) unique in window: {np.unique(labels).tolist()}")
    axes[6].set_xlabel("sample index")
    axes[6].set_ylabel("label")
    axes[6].grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()


if __name__ == "__main__":
    main()
