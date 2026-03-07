from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
EVAL_DIR = ARTIFACTS_DIR / "evaluations"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2

T_GAL = 0.80
T_STAR = 0.40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

for p in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, EVAL_DIR, FIGURES_DIR]:
    p.mkdir(parents=True, exist_ok=True)