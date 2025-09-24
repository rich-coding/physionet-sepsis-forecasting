from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import HTTPException

from .config import DATA_INFER_DIR, PARQUET_PATTERN

RECORD_FIELDS: List[str] = [
    "ICULOS",
    "Temp",
    "BaseExcess",
    "DBP",
    "FiO2",
    "Gender",
    "Age",
    "HCO3",
    "HR",
    "HospAdmTime",
    "Magnesium",
    "O2Sat",
    "Resp",
]
REQUIRED_COLUMNS: List[str] = ["patient_id"] + RECORD_FIELDS


class Inference:
    def __init__(
        self, parquet_dir: Path = DATA_INFER_DIR, parquet_pattern: str = PARQUET_PATTERN
    ) -> None:
        self.parquet_dir = Path(parquet_dir)
        self.parquet_pattern = parquet_pattern

    def get_json(self, number: int) -> Dict[str, Any]:
        if number < 1 or number > 5:
            raise HTTPException(status_code=422, detail="number debe estar entre 1 y 5")

        parquet_path = self.parquet_dir / self.parquet_pattern.format(n=number)
        if not parquet_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Parquet no encontrado: {parquet_path}"
            )

        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error leyendo parquet: {e}")

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=422, detail=f"Faltan columnas requeridas: {missing}"
            )

        df = df[REQUIRED_COLUMNS].copy()
        for col in RECORD_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["patient_id"] = df["patient_id"].astype(str)
        df = df.sort_values(["patient_id", "ICULOS"], kind="mergesort")

        batch: List[Dict[str, Any]] = []
        for pid, g in df.groupby("patient_id", sort=False):
            records = [
                {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                for row in g[RECORD_FIELDS].to_dict(orient="records")
            ]
            batch.append({"patient_id": pid, "records": records})

        # Metadata con turno y total de pacientes
        total_patients = int(df["patient_id"].nunique())
        metadata = {
            "turn": int(number),
            "total_patients": total_patients,
        }

        return {"metadata": metadata, "batch": batch}
