"""InterVoiceAI: one page for applying and for reviewing every applicant.

The apply form writes exactly one data/candidates/<id>.json (contract C1, see
src/candidates.py). The phone agent reads it at call start and generates its own
questions, so the form always leaves `plan` null. The same page lists applicants;
clicking one shows the application, the interview answers, the integrity analysis
and the transcript (src/recruiter.py).

  uv run python src/intake_app.py          # http://127.0.0.1:3001

  .env.local (optional)
    AGENT_PHONE     = +1646...   shown as the number to call
    DATA_DIR        = <path>     applicants, call records, decisions (default <repo>/data)
    EVENTS_DUMP_DIR = <path>     the agent's live call logs (default <repo>/calls)
"""

import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.errors import PyPdfError

import candidates
import recruiter

logger = logging.getLogger("intake")

# Only AGENT_PHONE, DATA_DIR and EVENTS_DUMP_DIR are read from it; nothing else
# reaches a template.
load_dotenv(".env.local")

HOST, PORT = "127.0.0.1", 3001
MAX_RESUME_BYTES = 5 * 1024 * 1024
ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
CALL_RE = re.compile(r"[A-Za-z0-9_.+-]+")  # room names carry the caller's "+1..."
YEARS_ERROR = "years_experience must be a whole number between 0 and 60"

app = FastAPI(title="InterVoiceAI", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


def parse_years(raw: str) -> int | None:
    """Whole number 0-60 from the form field; blank means unknown (None)."""
    if not raw.strip():
        return None
    try:
        years = int(raw)
    except ValueError as err:
        raise ValueError(YEARS_ERROR) from err
    if not 0 <= years <= 60:
        raise ValueError(YEARS_ERROR)
    return years


def extract_resume_text(filename: str, data: bytes) -> str:
    """Plain text from a .pdf or .txt upload; ValueError for anything else."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as err:
            raise ValueError("resume .txt must be UTF-8 text") from err
    elif suffix == ".pdf":
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("resume PDF is password-protected")
            text = "\n".join(page.extract_text() for page in reader.pages)
        except PyPdfError as err:  # base of PdfReadError, EmptyFileError, ...
            raise ValueError(f"could not read resume PDF: {err}") from err
    else:
        raise ValueError(f"resume must be a .pdf or .txt file, got '{filename}'")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise ValueError(
            "no text found in resume (scanned image?); upload a .txt instead"
        )
    return text


def render_page(
    request: Request,
    *,
    errors: list[str] | None = None,
    form: dict[str, str] | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="app.html",
        context={
            "errors": errors or [],
            "form": form or {},
            "agent_phone": os.getenv("AGENT_PHONE", ""),
        },
        status_code=status_code,
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return render_page(request)


@app.post("/apply", response_model=None)
async def apply(
    request: Request,
    name: Annotated[str, Form()] = "",
    phone: Annotated[str, Form()] = "",
    role: Annotated[str, Form()] = "",
    email: Annotated[str, Form()] = "",
    company: Annotated[str, Form()] = "",
    education: Annotated[str, Form()] = "",
    years_experience: Annotated[str, Form()] = "",
    current_job: Annotated[str, Form()] = "",
    biggest_project: Annotated[str, Form()] = "",
    why_this_role: Annotated[str, Form()] = "",
    resume: UploadFile
    | None = None,  # File() is inferred; None when the form has no file part
) -> Response:
    """Collect every validation problem, then write one C1 file and open it.

    Problems re-render the page with the form filled in and the errors listed."""
    form = {
        "name": name,
        "phone": phone,
        "role": role,
        "email": email,
        "company": company,
        "education": education,
        "years_experience": years_experience,
        "current_job": current_job,
        "biggest_project": biggest_project,
        "why_this_role": why_this_role,
    }
    errors: list[str] = []
    years: int | None = None
    try:
        years = parse_years(years_experience)
    except ValueError as err:
        errors.append(str(err))

    resume_text = resume_filename = ""
    if resume is None or not resume.filename:
        errors.append("resume file (.pdf or .txt) is required")
    else:
        resume_filename = Path(resume.filename).name
        data = await resume.read()
        if len(data) > MAX_RESUME_BYTES:
            errors.append("resume larger than 5 MB")
        else:
            try:
                resume_text = extract_resume_text(resume_filename, data)
            except ValueError as err:
                errors.append(str(err))

    try:
        candidate = candidates.new_candidate(
            name=name,
            phone=phone,
            role=role,
            email=email,
            company=company,
            education=education,
            years_experience=years,
            answers={
                "current_job": current_job.strip(),
                "biggest_project": biggest_project.strip(),
                "why_this_role": why_this_role.strip(),
            },
            resume_text=resume_text,
            resume_filename=resume_filename,
        )
    except ValueError as err:
        errors.append(str(err))
    if errors:
        return render_page(request, errors=errors, form=form, status_code=400)
    logger.info("saved %s -> %s", candidate["id"], candidates.save(candidate))
    return RedirectResponse(url=f"/#{candidate['id']}", status_code=303)


# ---------- recruiter data (JSON for the page) ----------


@app.get("/api/roster")
async def api_roster() -> dict[str, Any]:
    return recruiter.roster()


@app.get("/api/candidates/{candidate_id}")
async def api_candidate(candidate_id: str) -> dict[str, Any]:
    if not ID_RE.fullmatch(candidate_id):
        raise HTTPException(status_code=404, detail="no such candidate")
    try:
        return recruiter.candidate_report(candidate_id)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="no such candidate") from err


@app.get("/api/calls/{call_id}")
async def api_call(call_id: str) -> dict[str, Any]:
    if not CALL_RE.fullmatch(call_id):
        raise HTTPException(status_code=404, detail="no such call")
    try:
        return recruiter.call_report(call_id)
    except KeyError as err:
        raise HTTPException(status_code=404, detail="no such call") from err


class DecisionIn(BaseModel):
    v: str | None = None
    note: str = ""


@app.post("/api/decisions/{subject}")
async def api_decision(subject: str, body: DecisionIn) -> dict[str, Any]:
    if not CALL_RE.fullmatch(subject) or not recruiter.known_subject(subject):
        raise HTTPException(status_code=404, detail="no such applicant or call")
    try:
        return dict(recruiter.save_decision(subject, body.v, body.note))
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err)) from err


@app.get("/api/detectors")
async def api_detectors() -> dict[str, Any]:
    return recruiter.detector_performance()


# ---------- old URLs ----------


@app.get("/candidates/{candidate_id}")
async def old_candidate_page(candidate_id: str) -> RedirectResponse:
    if not ID_RE.fullmatch(candidate_id):
        raise HTTPException(status_code=404, detail="no such candidate")
    return RedirectResponse(url=f"/#{candidate_id}", status_code=308)


@app.get("/candidates")
async def old_candidates_page() -> RedirectResponse:
    return RedirectResponse(url="/", status_code=308)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT)  # an app object is fine while reload is off
