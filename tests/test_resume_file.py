"""The uploaded resume is kept as a file and served back as uploaded."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import candidates
import intake_app
import recruiter

FORM = {
    "name": "Maria Lopez",
    "phone": "(212) 555-0142",
    "role": "Registered Nurse",
    "years_experience": "3",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return TestClient(intake_app.app)


def test_save_resume_keeps_the_bytes_and_rejects_other_types(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    rel = candidates.save_resume("c_1", "Resume Final.PDF", b"%PDF-1.4 fake")
    assert rel == "resumes/c_1.pdf"
    assert (tmp_path / rel).read_bytes() == b"%PDF-1.4 fake"
    with pytest.raises(ValueError):
        candidates.save_resume("c_1", "resume.docx", b"x")


def test_upload_is_served_back_and_reported(client, tmp_path):
    body = b"MARIA LOPEZ, RN\nStaff nurse, Mount Sinai Queens, 2021 to present.\n"
    r = client.post(
        "/apply",
        data=FORM,
        files={"resume": ("maria.txt", body, "text/plain")},
        follow_redirects=False,
    )
    assert r.status_code == 303
    cid = r.headers["location"].split("#", 1)[1]
    cand = candidates.load(cid)
    assert cand["resume_file"] == f"resumes/{cid}.txt"

    served = client.get(f"/resumes/{cid}")
    assert served.status_code == 200
    assert served.content == body
    assert served.headers["content-type"].startswith("text/plain")
    assert served.headers["content-disposition"] == 'inline; filename="maria.txt"'

    report = recruiter.candidate_report(cid)["candidate"]
    assert report["resume_url"] == f"/resumes/{cid}" and report["resume_kind"] == "txt"


def test_missing_file_is_404(client, tmp_path):
    assert client.get("/resumes/c_nobody").status_code == 404
    old = candidates.new_candidate(name="Old Applicant", phone="2125550100", role="QA")
    candidates.save(old)  # resume_file "" : applied before file storage existed
    assert client.get(f"/resumes/{old['id']}").status_code == 404
    assert recruiter.candidate_report(old["id"])["candidate"]["resume_url"] is None
