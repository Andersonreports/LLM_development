"""FastAPI web application."""
from __future__ import annotations

import base64
import logging
import re
import secrets
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from .. import ai, pipeline, report
from ..annotate import tools_available
from ..config import settings
from ..db import Case, Event, Sample, SessionLocal, init_db, now
from ..resources import kb
from ..vcf import read_header

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("wes")

BASE = Path(__file__).resolve().parent.parent


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    db = SessionLocal()
    # jobs do not survive a restart
    for c in db.scalars(select(Case).where(Case.status.in_(["processing", "queued"]))):
        c.status, c.error = "failed", "Interrupted by server restart - click Run full analysis to restart"
    db.commit()
    db.close()
    yield


app = FastAPI(title="WES AI Analyser", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


def _local(dt, fmt: str = "%d %b %Y, %H:%M") -> str:
    """Render a stored (UTC) datetime in the server's local timezone."""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().strftime(fmt)


templates.env.filters["local"] = _local

# Cache-busting for /static/*: browsers aggressively cache CSS/JS by URL, so without
# this, a server restart alone doesn't guarantee a CSS/JS fix actually reaches a browser
# that already cached the old file - it has to be a new URL. Computed once from the
# newest file mtime under static/ at process startup, so every restart (which is when
# an edit takes effect anyway) naturally busts the cache for anything that changed.
def _static_version() -> str:
    static_dir = BASE / "static"
    try:
        newest = max((f.stat().st_mtime for f in static_dir.rglob("*") if f.is_file()), default=0)
    except OSError:
        newest = 0
    return f"{int(newest)}"


templates.env.globals["static_version"] = _static_version()

ROLES = pipeline.ROLES
FLAGS = ["POSITIVE", "INCONCLUSIVE", "NEGATIVE", "CARRIER"]
SECTIONS = ["primary", "vus", "carrier", "secondary", "not_reported"]
CLASSES = ["Pathogenic", "Likely pathogenic", "Uncertain significance", "Likely benign", "Benign"]


# ------------------------------------------------------------------ auth
# Simple in-memory session store: fine for a small internal tool, and consistent with
# background jobs already not surviving a restart. Set WES_USERS to require login.
_USERS = dict(u.split(":", 1) for u in settings.users.split(",") if ":" in u)
templates.env.globals["show_logout"] = bool(_USERS)
_SESSIONS: dict[str, str] = {}  # token -> username
SESSION_COOKIE = "wes_session"


def current_user(request: Request) -> str:
    if not _USERS:
        return request.headers.get("X-User", "lab-user")
    token = request.cookies.get(SESSION_COOKIE)
    if token and token in _SESSIONS:
        return _SESSIONS[token]
    auth = request.headers.get("Authorization", "")  # fallback for scripts/API callers
    if auth.startswith("Basic "):
        try:
            user, pw = base64.b64decode(auth[6:]).decode().split(":", 1)
        except Exception:  # noqa: BLE001
            user, pw = "", ""
        if user in _USERS and secrets.compare_digest(_USERS[user], pw):
            return user
    if request.url.path.startswith("/api/"):
        raise HTTPException(401, "Session expired - please log in again")
    raise HTTPException(303, headers={"Location": "/login"})


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/"):
    if not _USERS:
        return RedirectResponse("/")
    return templates.TemplateResponse(request, "login.html", {"next": next, "error": None})


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/")):
    if username in _USERS and secrets.compare_digest(_USERS[username], password):
        token = secrets.token_urlsafe(32)
        _SESSIONS[token] = username
        resp = RedirectResponse(next or "/", status_code=303)
        resp.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 14)
        return resp
    return templates.TemplateResponse(request, "login.html", {"next": next, "error": "Incorrect username or password"},
                                      status_code=401)


@app.post("/logout")
def logout(request: Request):
    _SESSIONS.pop(request.cookies.get(SESSION_COOKIE), None)
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(SESSION_COOKIE)
    return resp


# ------------------------------------------------------------------ helpers
def _event(db, case: Case, kind: str, msg: str, user: str) -> None:
    db.add(Event(case_id=case.id, kind=kind, message=msg, user=user))


def _case_or_404(db, case_id: int) -> Case:
    c = db.get(Case, case_id)
    if not c:
        raise HTTPException(404, "Case not found")
    return c


def _new_accession(db) -> str:
    stamp = datetime.now().strftime("%y%m%d")
    n = db.query(Case).filter(Case.accession.like(f"WES{stamp}%")).count() + 1
    return f"WES{stamp}{n:03d}"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:120] or "upload.vcf"


async def _save_upload(up: UploadFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / _safe_name(up.filename or "upload.vcf")
    with open(dest, "wb") as fh:
        while chunk := await up.read(1 << 20):
            fh.write(chunk)
    read_header(dest)  # raises ValueError if not a VCF
    return dest


def _auto_map(case: Case) -> list[str]:
    """Try to assign VCF sample columns to roles. Returns problems (empty = ok)."""
    problems = []
    for s in case.samples:
        cols = read_header(Path(s.vcf_path)).samples
        if s.vcf_sample and s.vcf_sample in cols:
            continue
        if len(cols) == 1:
            s.vcf_sample = cols[0]
            continue
        guess = [c for c in cols if c and (c == s.patient_id or c.lower() == (s.name or "").lower()
                                           or s.role in c.lower())]
        if len(guess) == 1:
            s.vcf_sample = guess[0]
        else:
            s.vcf_sample = ""
            problems.append(f"{s.role}: choose sample column ({', '.join(cols)})")
    return problems


# ------------------------------------------------------------------ pages
@app.get("/", response_class=HTMLResponse)
def index(request: Request, db=Depends(get_db), user=Depends(current_user)):
    cases = db.scalars(select(Case).order_by(Case.id.desc()).limit(500)).all()
    month = datetime.now().strftime("%Y-%m")
    stats = {
        "total": len(cases),
        "running": sum(c.status in ("processing", "queued") for c in cases),
        "review": sum(c.status in ("review", "needs_mapping", "failed") for c in cases),
        "approved": sum(c.status == "approved" and c.approved_at is not None
                        and c.approved_at.strftime("%Y-%m") == month for c in cases),
    }
    step_no = {k: i for i, (k, _, _) in enumerate(pipeline.STEPS)}
    return templates.TemplateResponse(request, "index.html", {
        "cases": cases, "user": user, "nav": "cases", "stats": stats, "step_no": step_no,
        "n_steps": len(pipeline.STEPS), "step_label": pipeline.STEP_LABEL})


@app.get("/system", response_class=HTMLResponse)
def system(request: Request, user=Depends(current_user)):
    k = kb()
    return templates.TemplateResponse(request, "system.html", {
        "tools": tools_available(), "kb_warnings": k.warnings, "genes": len(k.genes),
        "hpo_terms": len(k.ontology.terms) if k.ontology else 0, "ai": ai.available(),
        "settings": settings, "user": user, "nav": "system"})


@app.get("/cases/new", response_class=HTMLResponse)
def new_case(request: Request, user=Depends(current_user)):
    return templates.TemplateResponse(request, "new_case.html", {"user": user, "nav": "new"})


@app.post("/cases")
async def create_case(
    request: Request,
    db=Depends(get_db),
    user=Depends(current_user),
    case_type: str = Form(...),
    accession: str = Form(""),
    referring_clinician: str = Form(""),
    hospital: str = Form(""),
    indication: str = Form(""),
    clinical_history: str = Form(""),
    family_history: str = Form(""),
    consanguinity: str = Form("unknown"),
    hpo: str = Form(""),
    consent_secondary: bool = Form(False),
    consent_carrier: bool = Form(False),
    combined_vcf: UploadFile | None = File(None),
    coverage_file: UploadFile | None = File(None),
    seq_total_gb: str = Form(""),
    seq_q30: str = Form(""),
    start: bool = Form(True),
):
    if case_type not in ROLES:
        raise HTTPException(400, "Invalid case type")
    if case_type != "couple" and not clinical_history.strip():
        raise HTTPException(400, "Clinical history is required - the analysis cannot judge which findings "
                             "are relevant to the patient without it")
    form = await request.form()
    acc = _safe_name(accession.strip()) if accession.strip() else _new_accession(db)
    if db.scalar(select(Case).where(Case.accession == acc)):
        raise HTTPException(400, f"Accession {acc} already exists")
    updir = settings.uploads_dir / acc

    manual_hpo = []
    for tok in re.findall(r"HP:\d{7}", hpo):
        manual_hpo.append({"id": tok, "name": "", "excluded": False})

    case = Case(accession=acc, case_type=case_type, referring_clinician=referring_clinician or None,
                hospital=hospital or None, indication=indication or None,
                clinical_history=clinical_history, family_history=family_history or None,
                consanguinity=consanguinity if consanguinity in ("yes", "no") else None,
                manual_hpo=manual_hpo, consent_secondary=consent_secondary,
                consent_carrier=consent_carrier, analyst=user, status="created",
                seq_total_gb=seq_total_gb.strip() or None, seq_q30=seq_q30.strip() or None)
    db.add(case)
    db.flush()

    combined = None
    if combined_vcf is not None and combined_vcf.filename:
        try:
            combined = await _save_upload(combined_vcf, updir)
        except ValueError as e:
            db.rollback()
            raise HTTPException(400, str(e))

    if coverage_file is not None and coverage_file.filename:
        cov = updir / "coverage" / _safe_name(coverage_file.filename)
        cov.parent.mkdir(parents=True, exist_ok=True)
        cov.write_bytes(await coverage_file.read())
        case.coverage_path = str(cov)

    for role in ROLES[case_type]:
        up = form.get(f"{role}_vcf")
        path = combined
        if up is not None and getattr(up, "filename", None):
            try:
                path = await _save_upload(up, updir / role)
            except ValueError as e:
                db.rollback()
                raise HTTPException(400, str(e))
        if path is None:
            db.rollback()
            raise HTTPException(400, f"No VCF provided for {role} (upload a per-sample VCF or one combined multi-sample VCF)")
        db.add(Sample(
            case_id=case.id, role=role, vcf_path=str(path),
            vcf_sample=str(form.get(f"{role}_vcf_sample", "")).strip(),
            name=str(form.get(f"{role}_name", "")).strip() or role.capitalize(),
            patient_id=str(form.get(f"{role}_patient_id", "")).strip() or None,
            pin=str(form.get(f"{role}_pin", "")).strip() or None,
            sample_number=str(form.get(f"{role}_sample_number", "")).strip() or None,
            received_on=str(form.get(f"{role}_received_on", "")).strip() or None,
            age=str(form.get(f"{role}_age", "")).strip() or None,
            sex=str(form.get(f"{role}_sex", "")).strip() or _default_sex(role),
            affected=role == "proband" or bool(form.get(f"{role}_affected")),
            sample_type=str(form.get(f"{role}_sample_type", "")).strip() or None,
            collected_on=str(form.get(f"{role}_collected_on", "")).strip() or None,
        ))
    db.flush()
    db.refresh(case)
    problems = _auto_map(case)
    _event(db, case, "info", f"Case created ({case_type})", user)
    if problems:
        case.status = "needs_mapping"
        _event(db, case, "warning", "; ".join(problems), user)
    db.commit()
    if not problems and start:
        _queue(db, case, user)
    return RedirectResponse(f"/cases/{case.id}", status_code=303)


def _default_sex(role: str) -> str | None:
    return {"father": "male", "mother": "female"}.get(role)


def _queue(db, case: Case, user: str, ai_only: bool = False) -> None:
    case.status, case.error = "queued", None
    _event(db, case, "info", "Re-running AI interpretation" if ai_only else "Analysis queued", user)
    db.commit()
    pipeline.submit(case.id, ai_only=ai_only)


@app.get("/cases/{case_id}", response_class=HTMLResponse)
def case_page(case_id: int, request: Request, db=Depends(get_db), user=Depends(current_user)):
    case = _case_or_404(db, case_id)
    mapping = {}
    if case.status == "needs_mapping":
        mapping = {s.role: read_header(Path(s.vcf_path)).samples for s in case.samples}
    return templates.TemplateResponse(request, "case.html", {
        "case": case, "user": user, "mapping": mapping, "flags": FLAGS, "sections": SECTIONS,
        "classes": CLASSES, "ai_available": ai.available(), "nav": "cases",
        "steps": pipeline.progress(case), "step_label": pipeline.STEP_LABEL})


@app.post("/cases/{case_id}/mapping")
async def save_mapping(case_id: int, request: Request, db=Depends(get_db), user=Depends(current_user)):
    case = _case_or_404(db, case_id)
    form = await request.form()
    for s in case.samples:
        val = str(form.get(s.role, "")).strip()
        if val:
            s.vcf_sample = val
    problems = _auto_map(case)
    if problems:
        db.commit()
        raise HTTPException(400, "; ".join(problems))
    used = [(s.vcf_path, s.vcf_sample) for s in case.samples]
    if len(set(used)) != len(used):
        raise HTTPException(400, "The same VCF sample column is assigned to more than one role")
    case.status = "created"
    db.commit()
    _queue(db, case, user)
    return RedirectResponse(f"/cases/{case.id}", status_code=303)


@app.post("/cases/{case_id}/run")
def run(case_id: int, db=Depends(get_db), user=Depends(current_user)):
    case = _case_or_404(db, case_id)
    if case.status in ("processing", "queued"):
        raise HTTPException(409, "Analysis already running")
    if case.status in ("approved", "rejected"):
        raise HTTPException(409, f"Case is {case.status}; reopen it first")
    _queue(db, case, user)
    return RedirectResponse(f"/cases/{case.id}", status_code=303)


@app.post("/cases/{case_id}/rerun-ai")
def rerun_ai(case_id: int, db=Depends(get_db), user=Depends(current_user)):
    case = _case_or_404(db, case_id)
    if not case.results:
        raise HTTPException(409, "Run the full analysis first")
    if case.status in ("processing", "queued", "approved", "rejected"):
        raise HTTPException(409, f"Case is {case.status}")
    _queue(db, case, user, ai_only=True)
    return RedirectResponse(f"/cases/{case.id}", status_code=303)


@app.post("/cases/{case_id}/reopen")
def reopen(case_id: int, db=Depends(get_db), user=Depends(current_user)):
    case = _case_or_404(db, case_id)
    if case.status not in ("approved", "rejected"):
        raise HTTPException(409, "Case is not approved or rejected")
    was = case.status
    case.status = "review"
    _event(db, case, "review", f"{'Approved report v' + str(case.report_version) if was == 'approved' else 'Rejected case'} reopened for amendment", user)
    db.commit()
    return RedirectResponse(f"/cases/{case.id}", status_code=303)


@app.post("/cases/{case_id}/reject")
def reject(case_id: int, db=Depends(get_db), user=Depends(current_user), reason: str = Form(...)):
    c = _case_or_404(db, case_id)
    if c.status != "review":
        raise HTTPException(409, "Only cases awaiting review can be rejected")
    if not reason.strip():
        raise HTTPException(400, "A reason is required to reject a case")
    c.status = "rejected"
    c.rejected_by, c.rejected_at, c.rejection_reason = user, now(), reason.strip()
    _event(db, c, "review", f"Case rejected by {user}: {reason.strip()}", user)
    db.commit()
    return RedirectResponse(f"/cases/{c.id}", status_code=303)


@app.post("/cases/{case_id}/report-details")
async def save_report_details(case_id: int, request: Request, db=Depends(get_db), user=Depends(current_user)):
    """Patient/sample administrative details, filled in at reporting time rather than
    at upload - the analysis never needed them, only the printed report does."""
    c = _case_or_404(db, case_id)
    if c.status not in ("review",):
        raise HTTPException(409, f"Case is {c.status}; details can only be edited while awaiting review")
    form = await request.form()
    c.hospital = str(form.get("hospital", "")).strip() or None
    c.referring_clinician = str(form.get("referring_clinician", "")).strip() or None
    c.indication = str(form.get("indication", "")).strip() or None
    c.seq_total_gb = str(form.get("seq_total_gb", "")).strip() or None
    c.seq_q30 = str(form.get("seq_q30", "")).strip() or None
    for s in c.samples:
        p = s.role + "_"
        s.patient_id = str(form.get(p + "patient_id", s.patient_id or "")).strip() or None
        s.pin = str(form.get(p + "pin", s.pin or "")).strip() or None
        s.age = str(form.get(p + "age", s.age or "")).strip() or None
        s.sample_number = str(form.get(p + "sample_number", s.sample_number or "")).strip() or None
        s.sample_type = str(form.get(p + "sample_type", s.sample_type or "")).strip() or None
        s.collected_on = str(form.get(p + "collected_on", s.collected_on or "")).strip() or None
        s.received_on = str(form.get(p + "received_on", s.received_on or "")).strip() or None
    coverage_file = form.get("coverage_file")
    if coverage_file is not None and getattr(coverage_file, "filename", None):
        cov = settings.uploads_dir / c.accession / "coverage" / _safe_name(coverage_file.filename)
        cov.parent.mkdir(parents=True, exist_ok=True)
        cov.write_bytes(await coverage_file.read())
        c.coverage_path = str(cov)
    _event(db, c, "info", "Report details updated", user)
    db.commit()
    return RedirectResponse(f"/cases/{c.id}", status_code=303)


# ------------------------------------------------------------------ JSON API
@app.get("/api/hpo")
def hpo_search(q: str, user=Depends(current_user)):
    k = kb()
    return k.ontology.search(q) if k.ontology else []


@app.get("/api/cases/{case_id}/status")
def case_status(case_id: int, db=Depends(get_db), user=Depends(current_user)):
    c = _case_or_404(db, case_id)
    return {"status": c.status, "stage": c.stage, "stage_label": pipeline.STEP_LABEL.get(c.stage or "", c.stage),
            "error": c.error, "steps": pipeline.progress(c),
            "events": [{"at": pipeline._iso(e.at), "kind": e.kind, "message": e.message, "user": e.user}
                       for e in c.events[-40:]]}


@app.get("/api/cases/{case_id}/data")
def case_data(case_id: int, db=Depends(get_db), user=Depends(current_user)):
    c = _case_or_404(db, case_id)
    res = c.results or {}
    return {
        "case_type": c.case_type, "status": c.status, "review": c.review, "ai_result": c.ai_result,
        "rule_flag": res.get("rule_flag"), "final_flag": c.final_flag, "qc": c.qc,
        "candidates": res.get("candidates", []), "secondary": res.get("secondary"),
        "carrier": res.get("carrier"), "couple": res.get("couple"), "warnings": res.get("warnings", []),
        "hpo_terms": c.hpo_terms, "phenotype_ai": c.phenotype_ai, "roles": [s.role for s in c.samples],
        "gene_list": res.get("gene_list", []), "has_coverage": bool(c.coverage_path),
    }


@app.post("/api/cases/{case_id}/review")
async def save_review(case_id: int, request: Request, db=Depends(get_db), user=Depends(current_user)):
    c = _case_or_404(db, case_id)
    if c.status != "review":
        raise HTTPException(409, f"Case cannot be edited while '{c.status}'")
    body = await request.json()
    review = dict(c.review or {})
    if body.get("flag") not in FLAGS:
        raise HTTPException(400, "Invalid flag")
    if c.case_type != "couple" and body["flag"] == "CARRIER":
        raise HTTPException(400, "CARRIER flag applies to couple screening only")
    findings = []
    for f in body.get("findings", []):
        if f.get("report_section") not in SECTIONS:
            raise HTTPException(400, "Invalid report section")
        if f.get("classification_override") and f["classification_override"] not in CLASSES:
            raise HTTPException(400, "Invalid classification")
        findings.append(f)
    review.update({
        "flag": body["flag"],
        "summary_line": body.get("summary_line", ""),
        "clinical_history_text": body.get("clinical_history_text", ""),
        "cnv_text": body.get("cnv_text", ""),
        "gene_list": [g.strip().upper() for g in body.get("gene_list", []) if g.strip()],
        "findings": findings,
        "clinical_interpretation": body.get("clinical_interpretation", ""),
        "recommendations": [r for r in body.get("recommendations", []) if r.strip()],
        "limitations": [r for r in body.get("limitations", []) if r.strip()],
        "edited_by": user, "edited_at": now().isoformat(),
    })
    c.review = review
    c.final_flag = body["flag"]
    c.reviewer = user
    _event(db, c, "review", f"Review saved (flag {body['flag']}, {len(findings)} findings)", user)
    db.commit()
    return {"ok": True}


@app.post("/api/cases/{case_id}/hpo")
async def save_hpo(case_id: int, request: Request, db=Depends(get_db), user=Depends(current_user)):
    c = _case_or_404(db, case_id)
    if c.status in ("processing", "queued", "approved", "rejected"):
        raise HTTPException(409, f"Case is {c.status}")
    body = await request.json()
    from ..phenotype import validate_terms
    c.hpo_terms = validate_terms(kb(), body.get("terms", []))
    c.manual_hpo = c.hpo_terms
    _event(db, c, "info", f"HPO terms updated ({len(c.hpo_terms)}) - re-run analysis to re-rank", user)
    db.commit()
    return {"terms": c.hpo_terms}


@app.post("/cases/{case_id}/approve")
def approve(case_id: int, db=Depends(get_db), user=Depends(current_user), approver: str = Form("")):
    c = _case_or_404(db, case_id)
    if c.status != "review" or not c.review:
        raise HTTPException(409, "Only reviewed cases can be approved")
    if not c.final_flag:
        raise HTTPException(409, "Save the review (confirm the result flag) before approving")
    c.approved_by = approver.strip() or user
    c.approved_at = now()
    c.report_version += 1
    c.status = "approved"
    db.commit()
    pdf = report.render_pdf(c, final=True)
    _event(db, c, "report", f"Report v{c.report_version} approved by {c.approved_by}: {pdf.name}", user)
    db.commit()
    return RedirectResponse(f"/cases/{c.id}", status_code=303)


@app.get("/cases/{case_id}/report", response_class=HTMLResponse)
def report_html(case_id: int, db=Depends(get_db), user=Depends(current_user)):
    c = _case_or_404(db, case_id)
    if not c.review:
        raise HTTPException(409, "No results to report yet")
    return HTMLResponse(report.render_html(c, final=c.status == "approved"))


@app.get("/cases/{case_id}/report.pdf")
def report_pdf(case_id: int, db=Depends(get_db), user=Depends(current_user)):
    c = _case_or_404(db, case_id)
    if not c.review:
        raise HTTPException(409, "No results to report yet")
    final = c.status == "approved"
    path = settings.reports_dir / f"{c.accession}_v{c.report_version}.pdf" if final else None
    if not final or not path.exists():
        path = report.render_pdf(c, final=final)
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.post("/cases/{case_id}/delete")
def delete_case(case_id: int, db=Depends(get_db), user=Depends(current_user)):
    c = _case_or_404(db, case_id)
    if c.status in ("approved", "rejected"):
        raise HTTPException(409, f"{c.status.capitalize()} cases cannot be deleted - reopen first if you need to change it")
    if c.status in ("processing", "queued"):
        raise HTTPException(409, "Wait for the analysis to finish")
    for d in (settings.uploads_dir / c.accession, settings.work_dir / c.accession):
        shutil.rmtree(d, ignore_errors=True)
    db.delete(c)
    db.commit()
    return RedirectResponse("/", status_code=303)


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/") or "application/json" in request.headers.get("accept", ""):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)
    return templates.TemplateResponse(request, "error.html", {"detail": exc.detail, "code": exc.status_code},
                                      status_code=exc.status_code, headers=exc.headers)
