import os
from pathlib import Path

MODELS_DIR = Path(os.getenv("MODELS_DIR", "models/production"))
MODEL_PATH = MODELS_DIR / "hgb_model.joblib"
PREPROC_PATH = MODELS_DIR / "preprocess.joblib"
META_PATH = MODELS_DIR / "metadata.json"

API_TITLE = "Sepsis 2019 Scoring API"
API_VERSION = "1.0.0"

API_PATH = "/api/v1"
