from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from joblib import load
from .config import PREPROC_PATH, META_PATH

class Preprocessor:
    def __init__(self, preproc_path: Path = PREPROC_PATH, meta_path: Path = META_PATH):
        self.pipe = load(preproc_path)
        self.meta = json.loads(Path(meta_path).read_text())
        self.features_order = self.meta["features_order"]

    def transform(self, X_df: pd.DataFrame):
        X_df = X_df[self.features_order]
        return self.pipe.transform(X_df)

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold_f2"])
