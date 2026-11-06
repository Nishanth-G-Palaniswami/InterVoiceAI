"""Post-call analysis: AI-assisted-answer observations for one finished call.

Input is the call's agent events, either
  - a JSON file holding a list of event objects ({"type", "payload", "seq"...}), e.g. an export of the web's
    agent_events rows for one call, or
  - a worker log file from standalone mode, from which the `event (local) {...}` lines are extracted.

    uv run python evals/ai_detect/analyze_call.py worker.log
    uv run python evals/ai_detect/analyze_call.py call_events.json --json out.json
    uv run python evals/ai_detect/analyze_call.py worker.log --fixture my_call.json --tier human_spontaneous

--fixture writes the call's answers (text, follow-up text, words, onset) as a labelled fixture for the stress test,
so every real call you make becomes evaluation data. Redact names before committing a fixture.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from interviewer.ai_text import answer_texts, observe, report_lines  # noqa: E402
from interviewer.delivery import answer_delivery  # noqa: E402

_LOCAL = re.compile(r"event \(local\) (\{.*\})\s*$")


def load_events(path: pathlib.Path) -> list[dict]:
    text = path.read_text(errors="replace")
    stripped = text.lstrip()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if stripped.startswith("[") or stripped.startswith("{"):
        data = json.loads(stripped)
        if isinstance(data, dict):
            data = data.get("events") or data.get("agent_events") or []
        return list(data)
    events = []
    for line in text.splitlines():
        m = _LOCAL.search(line)
        if not m:
            continue
        raw = m.group(1)
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            # The worker log truncates long events at 500 chars; a cut `turn` with word timings is unusable.
            continue
    return events


def follow_up_texts(events: list[dict]) -> dict[str, str]:
    """question_id -> the candidate's follow-up text (turns after the second agent question, before record_answer)."""
    from interviewer.delivery import is_question
    out: dict[str, str] = {}
    buf: list[str] = []
    asked = 0
    for ev in sorted(events, key=lambda e: e.get("seq") or 0):
        p = ev.get("payload") or {}
        t = ev.get("type")
        if t == "turn" and p.get("role") == "agent" and is_question(str(p.get("text") or "")):
            asked += 1
        elif t == "turn" and p.get("role") == "candidate" and asked >= 2:
            buf.append(str(p.get("text") or ""))
        elif t in ("tool_mark_knockout", "tool_record_answer"):
            if t == "tool_record_answer":
                out[str(p.get("question_id") or "?")] = " ".join(buf).strip()
            buf, asked = [], 0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=pathlib.Path, help="worker log or events JSON")
    ap.add_argument("--json", type=pathlib.Path, help="write observations as JSON")
    ap.add_argument("--fixture", type=pathlib.Path, help="write this call's answers as a labelled fixture")
    ap.add_argument("--tier", default="human_spontaneous",
                    choices=["ai_verbatim", "ai_paraphrased", "human_spontaneous", "human_prepared"])
    args = ap.parse_args()

    events = load_events(args.source)
    turns = sum(1 for e in events if e.get("type") == "turn")
    answers = sum(1 for e in events if e.get("type") == "tool_record_answer")
    print(f"{len(events)} events, {turns} turns, {answers} recorded answers from {args.source}")
    if not answers:
        print("No record_answer events: nothing to analyze. If this is a worker log, long turn events may have been "
              "truncated at 500 chars; prefer an events JSON export.")
        return

    deliveries = {d.question_id: d for d in answer_delivery(events)}
    texts = answer_texts(events)
    obs = [observe(q, texts.get(q, ""), deliveries.get(q)) for q in texts]
    for line in report_lines(obs):
        print(line)

    if args.json:
        args.json.write_text(json.dumps([o.to_dict() for o in obs], indent=1, default=str))
        print(f"wrote {args.json}")
    if args.fixture:
        fus = follow_up_texts(events)
        stem = args.fixture.stem
        rows = []
        for i, (q, text) in enumerate(texts.items(), 1):
            d = deliveries.get(q)
            rows.append({"id": f"{stem}_{i:02d}", "tier": args.tier, "question_id": q, "text": text,
                         "follow_up_text": fus.get(q, ""), "onset_ms": d.onset_ms if d else None,
                         "source": f"real call via {args.source.name}; redact before committing"})
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        args.fixture.write_text(json.dumps({"answers": rows}, indent=1))
        print(f"wrote fixture {args.fixture} ({len(rows)} answers, tier={args.tier}). Redact names before committing.")


if __name__ == "__main__":
    main()
