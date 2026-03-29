from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PathConfig:
    project_root: Path
    repo_root: Path
    data_root: Path
    output_root: Path
    npz_path: Path
    model_dir: Path
    log_dir: Path
    figure_dir: Path


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = int(os.environ.get("WESAD_BATCH_SIZE", "128" if torch.cuda.is_available() else "64"))
    num_epochs: int = int(os.environ.get("WESAD_NUM_EPOCHS", "20"))
    learning_rate: float = float(os.environ.get("WESAD_LEARNING_RATE", "1e-3"))
    weight_decay: float = float(os.environ.get("WESAD_WEIGHT_DECAY", "1e-4"))
    num_workers: int = int(os.environ.get("WESAD_NUM_WORKERS", "2" if torch.cuda.is_available() else "0"))
    val_size: float = float(os.environ.get("WESAD_VAL_SIZE", "0.1"))
    test_size: float = float(os.environ.get("WESAD_TEST_SIZE", "0.2"))
    labels_to_use: tuple[int, ...] = (1, 2, 3, 4)
    model_name: str = "cnn1d_ecg_eda"
    save_metric: str = "macro_f1"
    use_amp: bool = os.environ.get("WESAD_USE_AMP", "1") == "1"
    require_cuda: bool = os.environ.get("WESAD_REQUIRE_CUDA", "0") == "1"


def get_path_config() -> PathConfig:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root = Path(os.environ.get("WESAD_REPO_ROOT", repo_root)).expanduser().resolve()
    default_project_root = repo_root.parent
    if (repo_root / "wesad_windows.npz").exists() or (repo_root / "data_wesad").exists():
        default_project_root = repo_root

    project_root = Path(os.environ.get("WESAD_PROJECT_ROOT", default_project_root)).expanduser().resolve()

    default_data_root = project_root / "data"
    if not default_data_root.exists() and (repo_root / "wesad_windows.npz").exists():
        default_data_root = repo_root

    data_root = Path(os.environ.get("WESAD_DATA_ROOT", default_data_root)).expanduser().resolve()
    output_root = Path(os.environ.get("WESAD_OUTPUT_ROOT", project_root / "outputs")).expanduser().resolve()

    default_npz_path = data_root / "wesad_windows.npz"
    if not default_npz_path.exists() and (repo_root / "wesad_windows.npz").exists():
        default_npz_path = repo_root / "wesad_windows.npz"

    return PathConfig(
        project_root=project_root,
        repo_root=repo_root,
        data_root=data_root,
        output_root=output_root,
        npz_path=Path(os.environ.get("WESAD_NPZ_PATH", default_npz_path)).expanduser().resolve(),
        model_dir=output_root / "models",
        log_dir=output_root / "logs",
        figure_dir=output_root / "figures",
    )


def ensure_output_dirs(paths: PathConfig) -> None:
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.figure_dir.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
