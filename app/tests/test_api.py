from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "threshold" in body


def test_score_one_record():
    payload = {
      "batch": [
        {
          "patient_id": "p000283",
          "records": [
            {
              "ICULOS"     :  17.0,
              "Temp"       :  36.5,
              "BaseExcess" :   5.0,
              "DBP"        :  70.0,
              "FiO2"       :   0.4,
              "Gender"     :   1.0,
              "Age"        :  42.95,
              "HCO3"       :  27.0,
              "HR"         :  93.5,
              "HospAdmTime":  -0.03,
              "Magnesium"  :   1.9,
              "O2Sat"      : 100.0,
              "Resp"       :  18.0
            },
            {
              "ICULOS"     :  40.0,
              "Temp"       :  37.89,
              "BaseExcess" :   6.0,
              "DBP"        :  57.0,
              "FiO2"       :   0.4,
              "Gender"     :   1.0,
              "Age"        :  42.95,
              "HCO3"       :  26.0,
              "HR"         : 106.0,
              "HospAdmTime":  -0.03,
              "Magnesium"  :   1.8,
              "O2Sat"      : 100.0,
              "Resp"       :  17.5
            }
          ]
        },
        {
          "patient_id": "p000009",
          "records": [
            {
              "ICULOS"     :   7.0,
              "Temp"       :  36.0,
              "BaseExcess" :  -1.0,
              "DBP"        :  64.0,
              "FiO2"       :   1.0,
              "Gender"     :   1.0,
              "Age"        :  27.92,
              "HCO3"       :  25.0,
              "HR"         : 120.0,
              "HospAdmTime":  -0.03,
              "Magnesium"  :   1.1,
              "O2Sat"      : 100.0,
              "Resp"       :  30.0
            },
            {
              "ICULOS"     : 20.0,
              "Temp"       : 37.0,
              "BaseExcess" : -2.0,
              "DBP"        : 60.0,
              "FiO2"       :  1.0,
              "Gender"     :  1.0,
              "Age"        : 27.92,
              "HCO3"       : 23.0,
              "HR"         : 98.0,
              "HospAdmTime": -0.03,
              "Magnesium"  :  2.0,
              "O2Sat"      : 93.0,
              "Resp"       : 20.75
            },
            {
              "ICULOS"     : 201.0,
              "Temp"       :  38.67,
              "BaseExcess" :   8.0,
              "DBP"        :  71.0,
              "FiO2"       :   0.4,
              "Gender"     :   1.0,
              "Age"        :  27.92,
              "HCO3"       :  30.0,
              "HR"         : 126.0,
              "HospAdmTime":  -0.03,
              "Magnesium"  :   2.0,
              "O2Sat"      : 100.0,
              "Resp"       :  20.0
            }
          ]
        }
      ]
    }

    r = client.post("api/v1/score", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, list) and len(body) == 2
    assert "score" in body[0] and "pred" in body[0] and "threshold" in body[0]
