"""Regression tests for web-layer bugs: unsafe accessions, accession reuse after a
delete, a stale confirmed flag surviving a re-run, spoofable user identity, the
login open redirect, and identifier scrubbing that mangled clinical text."""
from __future__ import annotations

import time

import pytest
from conftest import write_vcf
from fastapi.testclient import TestClient
from test_pipeline import SCN1A_FS

from wes import ai
from wes.config import settings
from wes.web import app as webapp
from wes.web.app import app


def _vcf(tmp_path):
    return write_vcf(tmp_path / "p.vcf", ["S1"], [("2", 1000, "AC", "A", SCN1A_FS, ["0/1"])])


def _create(client, vcf, **form):
    with open(vcf, "rb") as fh:
        return client.post("/cases", data={"case_type": "single", "proband_sex": "female",
                                           "clinical_history": "Seizures", **form},
                           files={"proband_vcf": ("p.vcf", fh, "text/plain")}, follow_redirects=False)


def _wait(client, cid):
    for _ in range(100):
        st = client.get(f"/api/cases/{cid}/status").json()
        if st["status"] in ("review", "failed"):
            return st
        time.sleep(0.1)
    raise AssertionError("case did not finish")


def _cid(resp) -> int:
    assert resp.status_code == 303, resp.text
    return int(resp.headers["location"].rsplit("/", 1)[1])


@pytest.mark.parametrize("acc", ["..", ".", ".hidden"])
def test_dot_accessions_rejected(tmp_path, acc):
    canary = settings.data_dir / "canary.txt"
    canary.write_text("keep")
    with TestClient(app) as client:
        r = _create(client, _vcf(tmp_path), accession=acc)
    assert r.status_code == 400
    assert canary.exists()


def test_accession_not_reused_after_delete(tmp_path):
    vcf = _vcf(tmp_path)
    with TestClient(app) as client:
        ids = [_cid(_create(client, vcf)) for _ in range(3)]
        for cid in ids:
            _wait(client, cid)
        assert client.post(f"/cases/{ids[1]}/delete", follow_redirects=False).status_code == 303
        _wait(client, _cid(_create(client, vcf)))


def test_rerun_clears_confirmed_flag(tmp_path):
    with TestClient(app) as client:
        cid = _cid(_create(client, _vcf(tmp_path)))
        _wait(client, cid)
        findings = client.get(f"/api/cases/{cid}/data").json()["review"]["findings"]
        assert client.post(f"/api/cases/{cid}/review", json={"flag": "NEGATIVE", "findings": findings}).status_code == 200
        assert client.get(f"/api/cases/{cid}/data").json()["final_flag"] == "NEGATIVE"

        client.post(f"/cases/{cid}/run", follow_redirects=False)
        _wait(client, cid)
        assert client.get(f"/api/cases/{cid}/data").json()["final_flag"] is None
        # the new draft has to be confirmed again before it can be approved
        assert client.post(f"/cases/{cid}/approve", follow_redirects=False).status_code == 409


def test_x_user_header_ignored_unless_trusted(tmp_path):
    from wes.db import Case, SessionLocal
    # create_case records the analyst from current_user, so a forged header would show up there
    with TestClient(app, headers={"X-User": "Dr Forged"}) as client:
        _wait(client, _cid(_create(client, _vcf(tmp_path), accession="XUSER1")))
    db = SessionLocal()
    try:
        assert db.query(Case).filter_by(accession="XUSER1").one().analyst == "lab-user"
    finally:
        db.close()


@pytest.mark.parametrize("nxt,expected", [("/cases/1", "/cases/1"), ("//evil.example", "/"),
                                          ("https://evil.example", "/"), ("/\\evil.example", "/")])
def test_login_redirect_stays_on_site(monkeypatch, nxt, expected):
    monkeypatch.setattr(webapp, "_USERS", {"alice": "pw"})
    with TestClient(app) as client:
        r = client.post("/login", data={"username": "alice", "password": "pw", "next": nxt}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == expected


def test_scrub_whole_words_only():
    text = ("Ram has seizures; parameter tests and a program of therapy. Anu reported manual dexterity "
            "issues. Father is well. Seen at Apollo Children's Hospital on 12/03/2025, call 98765 43210.")
    out = ai.scrub(text, ["Ram Kumar", "Anu", "Father", "Apollo Children's Hospital"])
    assert "parameter" in out and "program" in out and "manual" in out
    assert "Ram " not in out and "Anu " not in out and "Apollo" not in out
    assert "Father is well" in out          # role-name default sample names are not identifiers
    assert "12/03/2025" not in out and "98765 43210" not in out
