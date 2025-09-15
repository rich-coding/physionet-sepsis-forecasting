from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from .config import API_PATH, API_TITLE, API_VERSION
from .model import HGBCModel
from .schemas import SepsisBatchRequest, SepsisScore

FEATURES = [
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

app = FastAPI(title=API_TITLE, version=API_VERSION)
model = HGBCModel()

api = APIRouter(prefix=API_PATH)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@api.get("/health")
def health():
    return {
        "status": "ok",
        "model": "HGB",
        "version": API_VERSION,
        "threshold": model.threshold,
    }


@api.post("/score", response_model=list[SepsisScore])
def score(req: SepsisBatchRequest):
    # DataFrame de entrada
    raw_df = pd.DataFrame([r.model_dump() for r in req.records])
    raw_df["patient_id"] = raw_df["patient_id"].astype(str)

    # Solo columnas de features (en el orden esperado por el preproc guardado)
    X = raw_df[FEATURES].copy()

    # Coerción robusta a numérico (por si algo viene como string)
    X = X.apply(pd.to_numeric, errors="coerce")

    # Predict
    p = model.predict_proba(X)
    thr = model.threshold
    preds = (p >= thr).astype(int)

    # Respuesta (conservando el patient_id original como string)
    out = [
        SepsisScore(
            patient_id=str(pid), score=float(prob), pred=int(pred), threshold=thr
        )
        for pid, prob, pred in zip(
            raw_df["patient_id"].tolist(), p.tolist(), preds.tolist()
        )
    ]
    return out


app.include_router(api)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)
