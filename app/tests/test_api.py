from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "threshold" in body

def test_score_one_record():
    payload = {
        "records": [{
            "patient_id": 123,
            "ICULOS": 12, "Temp": 36.8, "BaseExcess": 0.0, "DBP": 60, "FiO2": 0.21,
            "Gender": 1, "Age": 65, "HCO3": 22, "HR": 95, "HospAdmTime": -24.0,
            "Magnesium": 2.0, "O2Sat": 97.0, "Resp": 18
        }]
    }
    r = client.post("/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) == 1
    assert "score" in body[0] and "pred" in body[0] and "threshold" in body[0]
