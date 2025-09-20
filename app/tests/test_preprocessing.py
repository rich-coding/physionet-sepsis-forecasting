import numpy as np
import pandas as pd

from app.preprocessing import Preprocessor
from app.schemas import SepsisBatchRequest
from app.schemas import SepsisRecord
from app.schemas import SepsisRequest

cols = [
    "ICULOS", "Temp", "BaseExcess", "DBP", "FiO2",
    "Gender", "Age", "HCO3", "HR", "HospAdmTime",
    "Magnesium", "O2Sat", "Resp"
]

def test_preprocessor_load_and_transform():
    prep = Preprocessor()
    request = get_batch_request()

    # --- Validación del request con dump ---
    dumped = request.model_dump()
    assert "batch" in dumped and len(dumped["batch"]) == 1

    first_req = dumped["batch"][0]
    assert "patient_id" in first_req
    assert "records" in first_req
    record = first_req["records"][0]

    for field in cols:
        assert field in record, f"Falta el campo {field}"
        assert isinstance(record[field], (int, float)), f"El campo {field} no es numérico"
        assert np.isfinite(record[field]), f"El campo {field} tiene un valor inválido"

    # --- Ejecutar transformador ---
    patient_ids, Xt = prep.transform(request)

    # Validar que patient_ids tenga el id correcto
    assert isinstance(patient_ids, list)
    assert len(patient_ids) == 1
    assert patient_ids[0] == first_req["patient_id"]

    # Validar la matriz de features transformada
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == (1, 174)
    assert np.isfinite(Xt).all()

def get_batch_request() -> SepsisBatchRequest:
    record = SepsisRecord(
        ICULOS=40.0,
        Temp=37.8,
        BaseExcess=6.0,
        DBP=57.0,
        FiO2=0.4,
        Gender=1,
        Age=42.9,
        HCO3=26.0,
        HR=106.0,
        HospAdmTime=-0.03,
        Magnesium=1.8,
        O2Sat=95.0,
        Resp=22.0
    )

    request = SepsisRequest(
        patient_id="p000283",
        records=[record]
    )

    return SepsisBatchRequest(
        batch=[request]
    )
