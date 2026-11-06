# PR 2 — Workstream B: candidate intake UI (branch `feat/02-intake-ui`, base `main`)

**Goal.** One server-rendered page where a candidate enters their details and uploads a resume;
submitting writes exactly one `data/candidates/<id>.json` (contract C1, section 1) that the phone
agent (Workstream A, built in parallel on another branch) reads. Nothing else: no auth, no DB, no JS
framework, no build step. The agent generates its own questions from the file, so `plan` stays `null`.

**Stack.** Python 3.13 (3.12+ style, full type hints, ruff-clean under the repo config), FastAPI +
Jinja2 server-rendered HTML, `uvicorn` on **127.0.0.1:3001** (port 3000 belongs to the user). One
Windows laptop, PowerShell 5.1, `uv` and `lk` on PATH. Run everything from the repo root.

**Preconditions.** (1) PR 0 is merged: `src/candidates.py` exists on `main`. Never write your own
`candidates.py` or re-implement the schema. (2) `.gitignore` lists `data/`; `git status` must never show
files under `data/`. (3) Read `src/candidates.py` once: this doc assumes its API (section 1); if a name
or return type differs, follow the module and adapt the snippets. Then `git switch -c feat/02-intake-ui main`.

## 1. Contract recap — C1 `data/candidates/<id>.json`

| Key | Type | Rule |
|---|---|---|
| `schema_version` | int | always `1` (set by `new_candidate`) |
| `id`, `created_at` | str | set by `new_candidate` (`c_20261106-170211_maria-lopez`, ISO 8601 with offset) |
| `name`, `phone`, `role` | str | **required**; `phone` stored digits only, 10 or 11 digits (agent matches on the last 10) |
| `email`, `company`, `education` | str | optional, `""` when blank (`company` falls back to "the company" in the agent prompt) |
| `years_experience` | int or null | whole number 0-60; blank -> `null` |
| `answers` | dict | exactly the keys `current_job`, `biggest_project`, `why_this_role` (str, may be `""`) |
| `resume_text`, `resume_filename` | str | plain text (cut to 6000 chars by the helper); basename of the upload |
| `plan` | null | **always `null` from the UI**; the agent writes `[{id, text, topic, grounded_in}]` back later |

`src/candidates.py` API used here:
- `new_candidate(*, name, phone, role, email="", company="", education="", years_experience=None,
  answers=None, resume_text="", resume_filename="") -> Candidate` — strips, normalises the phone,
  trims the resume, and raises `ValueError` naming every missing/invalid required field.
- `save(c) -> Path` (BOM-less UTF-8 JSON), `load(id) -> Candidate` (raises `FileNotFoundError`),
  `load_all() -> list[Candidate]` (newest first), `load_calls() -> list[dict]` (C2 files, each with
  `candidate_id`; `[]` when `data/calls/` is absent), `load_call(call_id) -> dict`.
- `Candidate` is a TypedDict: use `c["id"]`, not `c.id`, in Python (Jinja accepts `c.id`).
- `data_dir()` = env `DATA_DIR`, default `<repo>/data` (resolved from the module's location, not the CWD).
  Pure stdlib; the UI needs no LiveKit credentials.

## 2. Files

| Path | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | `uv add fastapi "uvicorn[standard]" python-multipart jinja2 pypdf` (commit both; uv may pick an older python-multipart to keep Python 3.10 support — fine) |
| `src/intake_app.py` | flat module like the rest of `src/` (`import candidates`, same mechanism as `from prompts import ...` in `agent.py`): app, validation, resume extraction, 4 routes, `__main__` runs uvicorn |
| `src/templates/base.html` | `<title>`, nav links to `/` and `/candidates`, ~15 lines of inline CSS (system font, 640px column), `{% block content %}` |
| `src/templates/apply.html`, `candidate.html`, `candidates.html` | the three pages (section 4) |
| `examples/resume.sample.txt` | 30-40 line fake resume for "Maria Lopez" (see `examples/candidate.example.json` for her facts); used in section 7 |

ASCII only inside `.py` strings and comments (ruff `RUF001` rejects curly quotes and dashes). Every new
text file is BOM-less UTF-8: write it with your editor tool, not `Out-File` (PowerShell 5.1 adds a BOM).

## 3. `src/intake_app.py` skeleton

Imports and signatures verified against current FastAPI 0.162, Starlette, pypdf 7.0, uvicorn 0.54;
`ruff format` fixes spacing.

```python
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
load_dotenv(".env.local")  # only AGENT_PHONE and DATA_DIR are read; nothing else reaches a template
HOST, PORT = "127.0.0.1", 3001
MAX_RESUME_BYTES = 5 * 1024 * 1024
ID_RE = re.compile(r"[A-Za-z0-9_.-]+")
app = FastAPI(title="InterviewAI intake", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")


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
            reader = PdfReader(BytesIO(data))  # stream: str | IO | Path
            if reader.is_encrypted:
                raise ValueError("resume PDF is password-protected")
            text = "\n".join(page.extract_text() for page in reader.pages)  # -> str
        except PyPdfError as err:  # base of PdfReadError, EmptyFileError, ...
            raise ValueError(f"could not read resume PDF: {err}") from err
    else:
        raise ValueError(f"resume must be a .pdf or .txt file, got '{filename}'")
    text = text.replace("\r\n", "\n").strip()
    if not text:
        raise ValueError("no text found in resume (scanned image?); upload a .txt instead")
    return text


@app.post("/apply")
async def apply(
    name: Annotated[str, Form()] = "",
    phone: Annotated[str, Form()] = "",
    role: Annotated[str, Form()] = "",
    # identical `Annotated[str, Form()] = ""` params for email, company, education,
    # years_experience (parsed to int | None below), current_job, biggest_project, why_this_role
    resume: UploadFile | None = None,  # File() is inferred; an empty <input type=file> arrives as None
) -> RedirectResponse:
    errors: list[str] = []  # section 5: collect every problem, then raise HTTPException(400, detail=errors)
    ...
    try:
        candidate = candidates.new_candidate(name=name, phone=phone, role=role, ..., plan stays None)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=[*errors, str(err)]) from err
    logger.info("saved %s -> %s", candidate["id"], candidates.save(candidate))
    return RedirectResponse(url=f"/candidates/{candidate['id']}", status_code=303)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT)  # an app object is fine while reload is off
```

| Route | Handler signature | Behaviour |
|---|---|---|
| `GET /` | `async def apply_form(request: Request) -> HTMLResponse` with `response_class=HTMLResponse` | `return templates.TemplateResponse(request=request, name="apply.html", context={})` |
| `POST /apply` | `apply(...)` above | validate (section 5) -> `new_candidate(...)` with `resume_filename=Path(resume.filename).name` -> `candidates.save` -> 303 to `/candidates/{id}` |
| `GET /candidates/{candidate_id}` | `async def candidate_page(request: Request, candidate_id: str) -> HTMLResponse` (name it `candidate_id`, not `id`: ruff `A002`) | `ID_RE.fullmatch` else 404; `candidates.load` (`FileNotFoundError` -> `HTTPException(404) from err`, ruff `B904`); context `c`, `snippet=c["resume_text"][:600]`, `agent_phone=os.getenv("AGENT_PHONE", "")` |
| `GET /candidates` | `async def candidates_page(request: Request) -> HTMLResponse` | `rows=candidates.load_all()` (already newest first), `called={call.get("candidate_id") for call in candidates.load_calls()}` |

## 4. Templates

- `apply.html`: `<form method="post" action="/apply" enctype="multipart/form-data">`; inputs, in this
  order, whose `name=` equals the Form parameter: `name` (text, required), `phone` (tel, required,
  placeholder `(212) 555-0142`), `email` (email), `role` (text, required, label "Role applying for"),
  `company` (text, labelled optional), `education` (text), `years_experience` (number, min 0, max 60),
  `current_job` / `biggest_project` / `why_this_role` (textarea rows=3 maxlength=600; labels "What do you
  do in your current job?", "Biggest project you have shipped?", "Why this role?"), `resume` (file,
  required, `accept=".pdf,.txt"`), submit "Apply".
- `candidate.html` (context `c`, `snippet`, `agent_phone`): `<h1>{{ c.name }}</h1>`; `<dl>` with role,
  company, email, phone, education, years, created_at; the three answers; `<pre>{{ snippet }}</pre>` plus
  "{{ c.resume_text|length }} chars from {{ c.resume_filename }}";
  `{% if agent_phone %}<p>Now call {{ agent_phone }} from {{ c.phone }}.</p>{% endif %}`;
  `{% if c.plan %}<ol>{% for q in c.plan %}<li>{{ q.text }} <small>{{ q.topic }} / {{ q.grounded_in }}</small></li>{% endfor %}</ol>{% else %}<p>The interviewer prepares its questions when you call.</p>{% endif %}`.
- `candidates.html` (context `rows`, `called`): table of name (link to `/candidates/{{ c.id }}`), role,
  phone, created_at, "has call" = `{{ "yes" if c.id in called else "no" }}`; an empty-state row
  "No applicants yet".
- Autoescaping is on for `.html` (Starlette passes `jinja2.select_autoescape()`), so resume text is safe
  to interpolate. Only `agent_phone` from the environment ever reaches a template; no secrets.

## 5. Validation and error responses (`POST /apply`)

| Check, in this order (all failures collected) | `detail` entry (HTTP 400 unless noted) |
|---|---|
| `years_experience` not `int()`-parsable or outside 0-60 (blank -> `None`) | `years_experience must be a whole number between 0 and 60` |
| `resume is None or not resume.filename` | `resume file (.pdf or .txt) is required` |
| `len(data) > MAX_RESUME_BYTES` after `data = await resume.read()` (read once) | `resume larger than 5 MB` |
| `extract_resume_text` raises `ValueError` (bad extension, unreadable / encrypted / empty PDF, non-UTF-8 .txt) | its message |
| `candidates.new_candidate` raises `ValueError` (blank name/role, phone not 10-11 digits) | its message |
| `/candidates/{candidate_id}`: id fails `ID_RE` or `load` raises | 404 `no such candidate` |

Raise `HTTPException(status_code=400, detail=errors)` once; FastAPI answers `{"detail": [...]}` as JSON
(the form's `required` / `accept` / `type=number` attributes keep humans from hitting it; curl will). A
malformed multipart body gets FastAPI's own 400 "There was an error parsing the body". Starlette's 1 MB
`max_part_size` applies to text fields only (file parts stream to a spooled temp file), so
`MAX_RESUME_BYTES` is the real cap. No silent fallbacks: no default resume text, no guessed phone, no
swallowed exceptions.

## 6. Run

```powershell
cd D:\Projects\IntervoiceAI\take2-voice-agent-hackathon-starter
uv run python src/intake_app.py                                        # http://127.0.0.1:3001 ; Ctrl+C stops
# auto-reload alternative (import string + --app-dir because src/ is flat)
uv run uvicorn --app-dir src intake_app:app --host 127.0.0.1 --port 3001 --reload
```
Optional: `AGENT_PHONE=+1646...` in `.env.local` shows "Now call ..." on the confirmation page.

## 7. Verify (PowerShell 5.1 aliases `curl` to Invoke-WebRequest, so call `curl.exe`)

```powershell
$base = "D:/Projects/IntervoiceAI/take2-voice-agent-hackathon-starter"
# 1. .txt upload -> expect "HTTP/1.1 303 See Other" and "location: /candidates/c_..."
curl.exe -s -i -F "resume=@$base/examples/resume.sample.txt" -F "name=Maria Lopez" --form-string "phone=(212) 555-0142" `
  -F "email=maria@example.com" -F "role=Registered Nurse, Med-Surg" -F "company=Harbor Health" -F "education=BSN, Hunter College, 2021" `
  -F "years_experience=3" -F "current_job=Night-shift RN on a 32-bed med-surg unit" -F "biggest_project=Led the unit fall-prevention rollout" `
  -F "why_this_role=Move to a teaching hospital" http://127.0.0.1:3001/apply | Select-String "HTTP/|location"
# 2. .pdf upload: any PDF on disk (none? open the sample .txt in Notepad, Ctrl+P, "Microsoft Print to PDF")
curl.exe -s -i -F "resume=@C:/Users/Dell/Downloads/resume-sample.pdf" -F "name=Sam Patel" -F "phone=6465550199" -F "role=Data Analyst" http://127.0.0.1:3001/apply | Select-String "HTTP/|location"
# 3. rejects -> 400 {"detail":["resume must be a .pdf or .txt file, got 'win.ini'", "Missing or invalid: phone (10 or 11 digits)"]}
curl.exe -s -i -F "resume=@C:/Windows/win.ini" -F "name=X" -F "phone=123" -F "role=Y" http://127.0.0.1:3001/apply
# 4. the JSON: plan null, phone digits only, resume_text <= 6000 chars, no BOM
$f = Get-ChildItem "$base/data/candidates/*.json" | Sort-Object LastWriteTime | Select-Object -Last 1
$c = Get-Content $f.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
$c | Select-Object id, name, phone, role, years_experience, resume_filename | Format-List; $null -eq $c.plan; $c.resume_text.Length
[IO.File]::ReadAllBytes($f.FullName)[0..2] -join ","      # must NOT be 239,187,191
# 5. pages: assert on DOM text, not screenshots
Start-Process "http://127.0.0.1:3001/candidates/$($c.id)"   # name, snippet, "Now call ..." only if AGENT_PHONE is set, plan placeholder
(Invoke-WebRequest "http://127.0.0.1:3001/candidates" -UseBasicParsing).Content -match "Maria Lopez"   # True
```

## 8. Lint, commit, hand-off

1. `uv run ruff format; uv run ruff check` -> "All checks passed!" (PowerShell: `;`, not `&&`). Fix
   findings; no `# noqa`.
2. Section 7 all green; `git status` shows only `pyproject.toml`, `uv.lock`, `src/intake_app.py`,
   `src/templates/`, `examples/resume.sample.txt` — never `data/` or `.env.local`.
3. `git add pyproject.toml uv.lock src/intake_app.py src/templates examples/resume.sample.txt;
   git commit -m "feat(intake): candidate apply page writing C1 JSON (PR 2)"` — no AI attribution lines.
4. Do not push. Report the status lines and the new id; the user runs
   `git push -u origin feat/02-intake-ui; gh pr create --fill --base main`, then checks the agent reads the
   file (`lk agent debugger start`, `say "Hi, this is Maria Lopez"`: the first question should cite a
   resume detail — Workstream A owns that path).

## 9. Follow-on: PR 4 scorecard pages (`feat/04-scorecard-ui`, branched from `main` after PR 3 merges)

- Same module and base template; recruiter-only, so never link these pages from the candidate pages.
  `GET /calls` -> `calls.html`: `rows = candidates.load_calls()` (newest first); columns call_id (link),
  candidate_name, matched_by, started_at, verdict (`call.integrity.verdict` if present, else "pending"),
  flagged_answers.
- `GET /calls/{call_id}` -> `call.html`: `candidates.load_call(call_id)` (`FileNotFoundError` -> 404).
  Header: candidate_name linked to `/candidates/{{ call.candidate_id }}` when not null, caller_number,
  matched_by, started/ended. Per answer: question, summary, quality_hint; a two-column table main vs
  follow_up over words, seconds, wpm, fillers_per100, pauses_per100, restarts_per100, disfluency_per100;
  `onset_ms`; `<ul>` of signals; a "sounds read" badge when `sounds_read`. Footer: `integrity.verdict`,
  `flagged_answers`, evidence `<ul>`, and `note` verbatim ("Observations for the recruiter. Not a score,
  not a decision.").
- PR 1 writes `answers[]` without `main` / `follow_up` / `integrity` (PR 3 adds them): guard with
  `{% if a.main %}` and `{% if call.integrity %}` and show "not scored yet" instead of erroring.
- Verify with a hand-made `data/calls/call-demo.json` (C2 shape above), then end to end: apply on :3001
  -> phone call -> `data/calls/<room>.json` -> `/calls/<room>`.
