from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
from joblib import load

from .config import META_PATH, PREPROC_PATH
from src.feature_engineering import engineer_features
from .schemas import SepsisBatchRequest

class Preprocessor:
    def __init__(self, preproc_path: Path = PREPROC_PATH, meta_path: Path = META_PATH):
        self.pipe = load(preproc_path)
        self.meta = json.loads(Path(meta_path).read_text())
        self.features_order = self.meta["features_order"]

    def transform(self, request: SepsisBatchRequest):
         # 1) Generar las mismas features que en training
        raw_df = self.convert_object_dataframe(request)
        df_fe = engineer_features(raw_df)

        # 2) Alinear columnas del request con las del training:
        X_df = df_fe.reindex(columns=self.features_order, fill_value=np.nan)

        # 3) Forzar numérico y tipo float64 para evitar problemas de schema
        X_df = X_df.apply(pd.to_numeric, errors="coerce").astype("float64")

        # 4) Transformar con el pipeline guardado
        X_trans = self.pipe.transform(X_df)
        return raw_df["patient_id"].tolist(), X_trans

    def convert_object_dataframe(self, request: SepsisBatchRequest) -> pd.DataFrame: 
        rows = []
        for req in request.batch:  # SepsisRequest
            pid = req.patient_id
            for rec in req.records:  # SepsisRecord
                d = rec.model_dump()
                d["patient_id"] = pid
                rows.append(d)

        return pd.DataFrame(rows)

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold_f2"])
