from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

from .config import META_PATH, MODEL_PATH
from .preprocessing import Preprocessor
from .schemas import SepsisBatchRequest

class HGBCModel:
    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
        self.model = load(model_path)
        self.meta = json.loads(Path(meta_path).read_text())
        self.prep = Preprocessor()

    def predict_proba(self, request: SepsisBatchRequest) -> pd.DataFrame:
        patient_list, Xt  = self.prep.transform(request)
        inferences = self.model.predict_proba(Xt)[:, 1]
        return self.get_max_prob_per_patient(inferences, patient_list)

    def get_max_prob_per_patient(self, inferences: pd.DataFrame, patient_list: list):
        df = pd.DataFrame({
            "patient_id": patient_list,
            "prob": np.asarray(inferences, dtype=float)
        })

        agg = (df.groupby("patient_id", as_index=False)["prob"]
                 .max()
                 .rename(columns={"prob": "max_prob"}))

        # Reordenar según primer aparición en patient_list
        order = list(dict.fromkeys(patient_list))
        out = agg.set_index("patient_id").reindex(order).reset_index()

        return out

    @property
    def threshold(self) -> float:
        return self.prep.threshold
