from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict


class SepsisRecord(BaseModel):
    patient_id: str
    ICULOS: float
    Temp: float
    BaseExcess: float
    DBP: float
    FiO2: float
    Gender: int
    Age: float
    HCO3: float
    HR: float
    HospAdmTime: float
    Magnesium: float
    O2Sat: float
    Resp: float

    model_config = ConfigDict(extra="forbid")


class SepsisBatchRequest(BaseModel):
    records: List[SepsisRecord]


class SepsisScore(BaseModel):
    patient_id: str
    score: float
    pred: int
    threshold: float
