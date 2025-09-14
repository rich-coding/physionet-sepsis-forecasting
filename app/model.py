from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load
from .config import MODEL_PATH, META_PATH
from .preprocessing import Preprocessor

class HGBCModel:
    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
        self.model = load(model_path)
        self.meta = json.loads(Path(meta_path).read_text())
        self.prep = Preprocessor()

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        Xt = self.prep.transform(X_df)
        return self.model.predict_proba(Xt)[:, 1]

    @property
    def threshold(self) -> float:
        return self.prep.threshold
