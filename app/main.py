from __future__ import annotations

from fastapi import APIRouter, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse, RedirectResponse

from .config import API_PATH, API_TITLE, API_VERSION
from .inference import Inference
from .model import HGBCModel
from .schemas import SepsisBatchRequest, SepsisScore

infer = Inference()

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    default_response_class=ORJSONResponse,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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
def score(request: SepsisBatchRequest):
    df_prob = model.predict_proba(request)
    thr = model.threshold

    pids = df_prob["patient_id"].astype(str).tolist()
    probs = df_prob["max_prob"].astype(float).tolist()
    preds = [int(p >= thr) for p in probs]

    return [
        SepsisScore(
            patient_id=str(pid), score=float(prob), pred=int(pred), threshold=thr
        )
        for pid, prob, pred in zip(pids, probs, preds)
    ]


@api.get("/turn")
def get_turn(number: int = Query(..., ge=1, le=5)):
    return infer.get_json(number=number)


app.include_router(api)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)
