from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, FastAPI
from fastapi.responses import RedirectResponse

from .config import API_PATH, API_TITLE, API_VERSION
from .model import HGBCModel
from .schemas import SepsisBatchRequest, SepsisScore

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
    raw_df = pd.DataFrame([r.model_dump() for r in req.records])
    raw_df["patient_id"] = raw_df["patient_id"].astype(str)

    base_cols = [c for c in raw_df.columns if c != "patient_id"]
    raw_df[base_cols] = raw_df[base_cols].apply(pd.to_numeric, errors="coerce")

    p = model.predict_proba(raw_df)
    thr = model.threshold
    preds = (p >= thr).astype(int)

    return [
        SepsisScore(patient_id=str(pid), score=float(prob), pred=int(pred), threshold=thr)
        for pid, prob, pred in zip(raw_df["patient_id"].tolist(), p.tolist(), preds.tolist())
    ]


app.include_router(api)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)
