from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class SepsisRecord(BaseModel):
    ICULOS: float
    Temp: Optional[float] = None
    BaseExcess: Optional[float] = None
    DBP: Optional[float] = None
    FiO2: Optional[float] = None
    Gender: Optional[int] = None
    Age: float
    HCO3: Optional[float] = None
    HR: Optional[float] = None
    HospAdmTime: float
    Magnesium: Optional[float] = None
    O2Sat: Optional[float] = None
    Resp: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


class SepsisRequest(BaseModel):
    patient_id: str
    records: List[SepsisRecord]


class SepsisBatchRequest(BaseModel):
    batch: List[SepsisRequest]


class SepsisScore(BaseModel):
    patient_id: str
    score: float
    pred: int
    threshold: float
