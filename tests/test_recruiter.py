"""The one-page recruiter view: joins, transcript, decisions and the JSON API."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import candidates
import intake_app
import recruiter

T0 = "2026-11-06T23:00:00+00:00"


def _iso(seconds: float) -> str:
    return f"2026-11-06T23:{int(seconds // 60):02d}:{seconds % 60:06.3f}+00:00"


def _turn(seq: int, at: float, role: str, text: str, **extra: Any) -> dict[str, Any]:
    epoch = 1794006000.0 + at  # 2026-11-06T23:00:00Z
    return {
        "seq": seq,
        "type": "turn",
        "occurred_at": _iso(at),
        "payload": {
            "turn_id": f"t{seq}-{at}",
            "role": role,
            "text": text,
            "interrupted": False,
            "started_at": epoch,
            "ended_at": epoch + 1,
            **extra,
        },
    }


def _event(seq: int, at: float, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"seq": seq, "type": kind, "occurred_at": _iso(at), "payload": payload}


def _write_log(folder: Path, call_id: str, events: list[dict[str, Any]]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{call_id}.events.jsonl"
    path.write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8", newline="\n"
    )
    return path


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    data, calls = tmp_path / "data", tmp_path / "calls"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("EVENTS_DUMP_DIR", str(calls))
    recruiter._cache.clear()
    return {"data": data, "calls": calls}


def _applicant(
    name: str = "Ana Ruiz", phone: str = "2125550100"
) -> candidates.Candidate:
    c = candidates.new_candidate(
        name=name, phone=phone, role="Phlebotomist", resume_text="Ana resume"
    )
    candidates.save(c)
    return c


def _interview(candidate_id: str | None) -> list[dict[str, Any]]:
    return [
        _event(
            1,
            0.0,
            "call_started",
            {"mode": "inbound", "candidate_id": candidate_id, "matched_by": "phone"},
        ),
        _turn(2, 1.0, "agent", "Hi, I'm Nia. Tell me about your last job."),
        _turn(3, 5.0, "candidate", "I drew blood at a clinic for two years."),
        _event(
            4,
            9.0,
            "tool_record_answer",
            {
                "question_id": "q1",
                "summary": "Two years at a clinic.",
                "quality_hint": "adequate",
            },
        ),
        _event(5, 12.0, "call_ended", {"outcome": "completed"}),
    ]


# ---------- joins ----------


def test_call_joins_its_applicant_through_call_started(env: dict[str, Path]) -> None:
    ana = _applicant()
    _write_log(env["calls"], "call-_+12125550100_A", _interview(ana["id"]))

    roster = recruiter.roster()

    [row] = roster["applicants"]
    assert row["id"] == ana["id"]
    assert row["calls"] == 1
    assert row["latest"]["outcome"] == "completed"
    assert row["latest"]["recorded"] == 1
    assert roster["unmatched"] == []


def test_call_without_an_applicant_is_unmatched(env: dict[str, Path]) -> None:
    _applicant()
    _write_log(env["calls"], "console", _interview(None))

    roster = recruiter.roster()

    assert roster["applicants"][0]["calls"] == 0
    assert [c["id"] for c in roster["unmatched"]] == ["console"]


def test_report_merges_summary_question_and_analysis(env: dict[str, Path]) -> None:
    ana = _applicant()
    _write_log(env["calls"], "call-A", _interview(ana["id"]))

    report = recruiter.candidate_report(ana["id"])

    assert "resume_text" not in report["candidate"]
    assert report["candidate"]["resume_excerpt"] == "Ana resume"
    [call] = report["calls"]
    assert call["duration_s"] == 12
    q1 = next(a for a in call["answers"] if a["question_id"] == "q1")
    assert q1["recorded"] is True
    assert q1["summary"] == "Two years at a clinic."
    # Ana has no plan yet, so question text falls back to the walk-in plan.
    assert q1["question"] == candidates.WALK_IN_PLAN[0]["text"]
    assert q1["analysis"] is not None


def test_duplicate_applicant_ids_are_listed_once(env: dict[str, Path]) -> None:
    ana = _applicant()
    copy = env["data"] / "candidates" / "copy.json"
    copy.write_text(json.dumps(ana), encoding="utf-8")

    assert [a["id"] for a in recruiter.roster()["applicants"]] == [ana["id"]]


# ---------- live logs ----------


def test_half_written_last_line_waits_for_the_next_read(tmp_path: Path) -> None:
    path = tmp_path / "live.events.jsonl"
    path.write_text(
        json.dumps(_turn(1, 0.0, "agent", "Hello")) + "\n" + '{"seq": 2, "ty',
        encoding="utf-8",
    )

    assert [e["seq"] for e in recruiter.read_events(path)] == [1]


def test_log_without_call_ended_is_live_while_fresh(env: dict[str, Path]) -> None:
    ana = _applicant()
    _write_log(env["calls"], "call-live", _interview(ana["id"])[:3])

    assert recruiter.roster()["applicants"][0]["latest"]["in_progress"] is True


def test_record_outcome_beats_call_ended(env: dict[str, Path]) -> None:
    """call_ended says "completed" for any normal close, even a hang-up after one
    question; Nia's record knows the interview was dropped."""
    ana = _applicant()
    _write_log(env["calls"], "call-_+1_A", _interview(ana["id"]))
    candidates.save_call(
        "call-_+1_A",
        {
            "call_id": "call-_+1_A",
            "candidate_id": ana["id"],
            "outcome": "dropped",
            "answers": [],
        },
    )

    assert recruiter.roster()["applicants"][0]["latest"]["outcome"] == "dropped"


# ---------- transcript ----------


def test_transcript_orders_by_time_across_a_session_restart() -> None:
    events = [
        _event(1, 0.0, "call_started", {"candidate_id": None}),
        _turn(2, 1.0, "agent", "Hello, I'm Nia."),
        # The agent restarted: seq starts over, the greeting comes again.
        _event(1, 20.0, "call_started", {"candidate_id": None}),
        _turn(2, 21.0, "agent", "Hello again, I'm Nia."),
        _turn(3, 25.0, "candidate", "Hi.", interrupted=True),
        _event(4, 26.0, "tool_mark_knockout", {"confirmed": "True"}),
        _event(5, 30.0, "tool_record_answer", {"question_id": "q1"}),
    ]

    lines, restarts = recruiter.transcript(events)

    assert restarts == 1
    assert [(x["kind"], x["text"]) for x in lines] == [
        ("turn", "Hello, I'm Nia."),
        ("restart", "Agent session restarted"),
        ("turn", "Hello again, I'm Nia."),
        ("turn", "Hi."),
        ("identity", "Identity confirmed"),
        ("answer", "q1"),
    ]
    assert [x["at_s"] for x in lines] == [1.0, 20.0, 21.0, 25.0, 26.0, 30.0]
    assert lines[3]["interrupted"] is True


def test_transcript_drops_a_repeated_turn_id() -> None:
    turn = _turn(2, 1.0, "agent", "Hello")
    lines, _ = recruiter.transcript([turn, turn])
    assert len(lines) == 1


# ---------- decisions ----------


def test_decision_round_trip(env: dict[str, Path]) -> None:
    saved = recruiter.save_decision("c_1", "go", "  strong answers  ")

    assert saved["v"] == "go"
    assert saved["note"] == "strong answers"
    assert recruiter.load_decision("c_1") == saved
    assert (env["data"] / "decisions" / "c_1.json").is_file()


def test_decision_rejects_an_unknown_value(env: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="decision must be one of"):
        recruiter.save_decision("c_1", "maybe", "")


def test_room_names_become_safe_file_names(env: dict[str, Path]) -> None:
    recruiter.save_decision("call-_+1646_X", None, "note only")

    assert (env["data"] / "decisions" / "call-__1646_X.json").is_file()
    assert recruiter.load_decision("call-_+1646_X")["note"] == "note only"


# ---------- API ----------


@pytest.fixture
def client(env: dict[str, Path]) -> TestClient:
    return TestClient(intake_app.app)


def test_api_roster_and_report(env: dict[str, Path], client: TestClient) -> None:
    ana = _applicant()
    _write_log(env["calls"], "call-_+12125550100_A", _interview(ana["id"]))

    roster = client.get("/api/roster").json()
    assert roster["applicants"][0]["name"] == "Ana Ruiz"

    report = client.get(f"/api/candidates/{ana['id']}").json()
    assert report["kind"] == "candidate"
    assert [
        t["role"] for t in report["calls"][0]["transcript"] if t["kind"] == "turn"
    ] == [
        "agent",
        "candidate",
    ]

    call = client.get("/api/calls/call-_+12125550100_A").json()
    assert call["calls"][0]["id"] == "call-_+12125550100_A"


def test_api_unknown_ids_are_404(client: TestClient) -> None:
    assert client.get("/api/candidates/nobody").status_code == 404
    assert client.get("/api/candidates/..%2Fsecrets").status_code == 404
    assert client.get("/api/calls/nope").status_code == 404


def test_api_decision_only_for_known_subjects(
    env: dict[str, Path], client: TestClient
) -> None:
    ana = _applicant()

    ok = client.post(f"/api/decisions/{ana['id']}", json={"v": "hold", "note": "ask"})
    assert ok.status_code == 200
    assert ok.json()["v"] == "hold"

    bad_value = client.post(f"/api/decisions/{ana['id']}", json={"v": "yes"})
    assert bad_value.status_code == 422

    unknown = client.post("/api/decisions/c_ghost", json={"v": "go"})
    assert unknown.status_code == 404
    assert not (env["data"] / "decisions" / "c_ghost.json").exists()


def test_apply_success_opens_the_new_applicant(
    env: dict[str, Path], client: TestClient
) -> None:
    r = client.post(
        "/apply",
        data={"name": "Lee Park", "phone": "(646) 555-0101", "role": "Nurse"},
        files={"resume": ("cv.txt", b"Lee resume text", "text/plain")},
        follow_redirects=False,
    )

    assert r.status_code == 303
    [saved] = candidates.load_all()
    assert r.headers["location"] == f"/#{saved['id']}"


def test_apply_errors_rerender_the_form_with_values(client: TestClient) -> None:
    r = client.post(
        "/apply",
        data={"name": "Lee <b>Park</b>", "phone": "12", "role": "Nurse"},
        files={"resume": ("cv.doc", b"x", "application/msword")},
    )

    assert r.status_code == 400
    assert "Fix these and submit again" in r.text
    assert "resume must be a .pdf or .txt file" in r.text
    assert "phone (10 or 11 digits)" in r.text
    # The typed name comes back, escaped.
    assert 'value="Lee &lt;b&gt;Park&lt;/b&gt;"' in r.text
    assert candidates.load_all() == []


def test_old_urls_redirect_into_the_page(client: TestClient) -> None:
    r = client.get("/candidates/c_1", follow_redirects=False)
    assert r.status_code == 308
    assert r.headers["location"] == "/#c_1"
