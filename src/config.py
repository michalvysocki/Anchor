import os
from pathlib import Path
from typing import Literal

ObservabilityMode = Literal["insufficient", "sufficient"]

SEED = 42
HIDDEN_SIZE = 64
DP_HIDDEN_SIZE = 256
K_MAX = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def model_path(mode: ObservabilityMode) -> Path:
    return MODELS_DIR / f"predictor_{mode}.pt"


def direct_model_path(mode: ObservabilityMode) -> Path:
    return MODELS_DIR / f"direct_predictor_{mode}.pt"


RESULTS_DIR = PROJECT_ROOT / "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
