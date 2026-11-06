"""Candidate intake UI: a server-rendered apply page that writes contract C1.

Submitting the form writes exactly one data/candidates/<id>.json (see
src/candidates.py and docs/plans/02-intake-ui.md). The phone agent reads it at
call start and generates its own questions, so the UI always leaves `plan` null.

  uv run python src/intake_app.py          # http://127.0.0.1:3001

  .env.local (optional)
    AGENT_PHONE = +1646...   shown on the confirmation page as "Now call ..."
    DATA_DIR    = <path>     where the JSON goes (default: <repo>/data)
"""

import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Annotated

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pypdf import PdfReader
from pypdf.errors import PyPdfError

import candidates

logger = logging.getLogger("intake")

# Only AGENT_PHONE and DATA_DIR are read from it; nothing else reaches a template.
load_dotenv(".env.local")

HOST, PORT = "127.0.0.1", 3001
MAX_RESUME_BYTES = 5 * 1024 * 1024
ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
YEARS_ERROR = "years_experience must be a whole number between 0 and 60"

app = FastAPI(title="InterviewAI intake", docs_url=None, redoc_url=None)
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


@app.get("/", response_class=HTMLResponse)
async def apply_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="apply.html", context={})


@app.post("/apply")
async def apply(
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
) -> RedirectResponse:
    """Collect every validation problem, then write one C1 file and redirect to it."""
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
        raise HTTPException(status_code=400, detail=[*errors, str(err)]) from err
    if errors:
        raise HTTPException(status_code=400, detail=errors)
    logger.info("saved %s -> %s", candidate["id"], candidates.save(candidate))
    return RedirectResponse(url=f"/candidates/{candidate['id']}", status_code=303)


@app.get("/candidates/{candidate_id}", response_class=HTMLResponse)
async def candidate_page(request: Request, candidate_id: str) -> HTMLResponse:
    if not ID_RE.fullmatch(candidate_id):
        raise HTTPException(status_code=404, detail="no such candidate")
    try:
        c = candidates.load(candidate_id)
    except FileNotFoundError as err:
        raise HTTPException(status_code=404, detail="no such candidate") from err
    return templates.TemplateResponse(
        request=request,
        name="candidate.html",
        context={
            "c": c,
            "snippet": c["resume_text"][:600],
            "agent_phone": os.getenv("AGENT_PHONE", ""),
        },
    )


@app.get("/candidates", response_class=HTMLResponse)
async def candidates_page(request: Request) -> HTMLResponse:
    called = {call.get("candidate_id") for call in candidates.load_calls()}
    return templates.TemplateResponse(
        request=request,
        name="candidates.html",
        context={"rows": candidates.load_all(), "called": called},
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT)  # an app object is fine while reload is off
