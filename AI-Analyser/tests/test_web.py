from __future__ import annotations

import time

from conftest import write_vcf
from fastapi.testclient import TestClient
from test_pipeline import BRCA2_STOP, SCN1A_FS

from wes.web.app import app


def test_full_web_flow(tmp_path):
    vcf = write_vcf(tmp_path / "web.vcf", ["S1"], [
        ("2", 1000, "AC", "A", SCN1A_FS, ["0/1"]),
        ("13", 3000, "C", "T", BRCA2_STOP, ["0/1"]),
    ])
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/cases/new").status_code == 200
        assert client.get("/system").status_code == 200
        assert client.get("/api/hpo", params={"q": "seiz"}).json()[0]["id"] == "HP:0001250"

        with open(vcf, "rb") as fh:
            r = client.post("/cases", data={
                "case_type": "single", "proband_name": "Test Child", "proband_patient_id": "PID1",
                "proband_age": "3y", "proband_sex": "female", "indication": "Epilepsy",
                "clinical_history": "Febrile seizures from 6 months", "hpo": "HP:0001250, HP:0001263",
                "consent_secondary": "true",
            }, files={"proband_vcf": ("web.vcf", fh, "text/plain")}, follow_redirects=False)
        assert r.status_code == 303, r.text
        case_url = r.headers["location"]
        cid = int(case_url.rsplit("/", 1)[1])

        for _ in range(100):
            st = client.get(f"/api/cases/{cid}/status").json()
            if st["status"] in ("review", "failed"):
                break
            time.sleep(0.1)
        assert st["status"] == "review", st

        page = client.get(case_url)
        assert page.status_code == 200 and "Review" in page.text
        data = client.get(f"/api/cases/{cid}/data").json()
        assert data["candidates"][0]["gene"] == "SCN1A"
        review = data["review"]
        review["clinical_interpretation"] = "Reviewed interpretation."
        r = client.post(f"/api/cases/{cid}/review", json={
            "flag": "POSITIVE", "summary_line": "LP SCN1A variant", "findings": review["findings"],
            "clinical_interpretation": review["clinical_interpretation"],
            "recommendations": ["Genetic counselling"], "limitations": []})
        assert r.status_code == 200, r.text
        assert client.post(f"/api/cases/{cid}/review", json={"flag": "CARRIER", "findings": []}).status_code == 400

        assert client.get(f"/cases/{cid}/report").status_code == 200
        r = client.post(f"/cases/{cid}/approve", data={"approver": "Dr Reviewer"}, follow_redirects=False)
        assert r.status_code == 303
        pdf = client.get(f"/cases/{cid}/report.pdf")
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
        html = client.get(f"/cases/{cid}/report").text
        assert "data:image/jpeg" in html and "DRAFT" not in html.split("<body>")[1][:300]
        # approved cases are locked
        assert client.post(f"/api/cases/{cid}/review", json={"flag": "NEGATIVE", "findings": []}).status_code == 409
