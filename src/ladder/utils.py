from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def get_device() -> str:
    """Best available device: CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def output_dir(experiment: str) -> Path:
    """Per-experiment output folder under outputs/ (git-ignored), created on demand."""
    path = OUTPUTS_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path
