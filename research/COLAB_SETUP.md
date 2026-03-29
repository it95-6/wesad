# Colab / VS Code setup

## Colab

Recommended setup for Colab:

- code repository: `/content/wesad_repo`
- data and outputs: Google Drive

This is faster and more reliable than running the repo directly from Drive.

Important:

- If you added `research/` only in your local working tree, Colab will not see it until you either:
  - push the changes to GitHub and clone again in Colab, or
  - upload/copy the updated repository itself to Drive and use that path as `REPO_ROOT`

Before running the code, change the runtime to GPU in Colab:

`Runtime -> Change runtime type -> Hardware accelerator -> GPU`

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

```python
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/content/drive/MyDrive/WESAD_project")
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
REPO_ROOT = Path("/content/wesad_repo")
REPO_URL = "https://github.com/it95-6/wesad.git"

if not REPO_ROOT.exists():
    !git clone {REPO_URL} {REPO_ROOT}

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["WESAD_PROJECT_ROOT"] = str(PROJECT_ROOT)
os.environ["WESAD_REPO_ROOT"] = str(REPO_ROOT)
os.environ["WESAD_DATA_ROOT"] = str(DATA_ROOT)
os.environ["WESAD_OUTPUT_ROOT"] = str(OUTPUT_ROOT)
os.environ["WESAD_NPZ_PATH"] = str(DATA_ROOT / "wesad_windows.npz")
os.environ["WESAD_NUM_WORKERS"] = "2"
os.environ["WESAD_REQUIRE_CUDA"] = "1"
os.environ["WESAD_USE_AMP"] = "1"

print("repo_root   =", REPO_ROOT)
print("data_root   =", DATA_ROOT)
print("output_root =", OUTPUT_ROOT)
print("npz_path    =", Path(os.environ["WESAD_NPZ_PATH"]))
print("npz_exists  =", Path(os.environ["WESAD_NPZ_PATH"]).exists())
```

```python
%cd /content/wesad_repo
!pip install -q torch torchvision torchaudio scikit-learn seaborn matplotlib numpy pandas
!nvidia-smi
!python -m research.train_cnn
!python -m research.evaluate_cnn
```

## VS Code

Run from the repository root.

```bash
export WESAD_PROJECT_ROOT="$(pwd)/.."
export WESAD_REPO_ROOT="$(pwd)"
export WESAD_DATA_ROOT="$(pwd)/../data"
export WESAD_OUTPUT_ROOT="$(pwd)/../outputs"
export WESAD_NUM_WORKERS="0"

python -m research.train_cnn
python -m research.evaluate_cnn
```
