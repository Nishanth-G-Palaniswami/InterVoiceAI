"""Stress-test the AI-assisted-answer detector on labelled transcripts.

    uv run python evals/ai_detect/run.py                 # synthetic timings, all fixtures
    uv run python evals/ai_detect/run.py --timings none  # text signals only (what a transcript-only review sees)
    uv run python evals/ai_detect/run.py --json out.json

Tiers: ai_verbatim (generated text read aloud), ai_paraphrased (generated, re-spoken in the candidate's words),
human_spontaneous (told live), human_prepared (a real story, rehearsed). Catch rate = flagged share of the ai_* tiers;
false-flag rate = flagged share of the human_* tiers. Every detector row sits beside the do-nothing baselines
(never flag, always flag, coin flip, long-answer-only), because a number without its baseline is not a result.

Real transcripts: drop redacted JSON files with the same shape into fixtures/real/ and they are included.
Word timings for synthetic fixtures are generated from a per-tier speaking profile (seeded, deterministic);
they stand in for Deepgram word timings until real calls supply them, and the report says so.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from interviewer.ai_text import observe  # noqa: E402
from interviewer.delivery import CandTurn, features, judge  # noqa: E402

# Speaking profiles used to synthesize word timings per tier (seconds). Deliberately coarse.
PROFILES = {
    "ai_verbatim":       {"gap": 0.06, "dur": 0.27, "pause_p": 0.02, "pause_s": 0.6, "onset_ms": 4200},
    "ai_paraphrased":    {"gap": 0.11, "dur": 0.29, "pause_p": 0.07, "pause_s": 0.7, "onset_ms": 2600},
    "human_spontaneous": {"gap": 0.15, "dur": 0.30, "pause_p": 0.12, "pause_s": 0.8, "onset_ms": 1100},
    "human_prepared":    {"gap": 0.08, "dur": 0.28, "pause_p": 0.04, "pause_s": 0.6, "onset_ms": 1400},
}
FOLLOW_UP_PROFILE = {  # the unscripted follow-up: readers and paraphrasers get less fluent, humans stay themselves
    "ai_verbatim":       {"gap": 0.16, "dur": 0.30, "pause_p": 0.15, "pause_s": 0.9},
    "ai_paraphrased":    {"gap": 0.14, "dur": 0.30, "pause_p": 0.12, "pause_s": 0.8},
    "human_spontaneous": {"gap": 0.15, "dur": 0.30, "pause_p": 0.12, "pause_s": 0.8},
    "human_prepared":    {"gap": 0.10, "dur": 0.28, "pause_p": 0.06, "pause_s": 0.6},
}
AI_TIERS = ("ai_verbatim", "ai_paraphrased")
HUMAN_TIERS = ("human_spontaneous", "human_prepared")


def synth_words(text: str, prof: dict, seed: str) -> list[list]:
    rng = random.Random(seed)
    t = 0.0
    out = []
    for tok in text.split():
        if rng.random() < prof["pause_p"]:
            t += prof["pause_s"]
        clean = tok.strip(".,?!;:\"'")
        if not clean:
            continue
        out.append([clean, round(t, 2), round(t + prof["dur"], 2)])
        t += prof["dur"] + prof["gap"]
    return out


def load_fixtures() -> list[dict]:
    answers = []
    base = json.loads((HERE / "fixtures" / "answers.json").read_text())
    answers.extend(base["answers"])
    for sub in ("public", "real"):
        real_dir = HERE / "fixtures" / sub
        if not real_dir.exists():
            continue
        for p in sorted(real_dir.glob("*.json")):
            data = json.loads(p.read_text())
            for a in data.get("answers", []):
                a.setdefault("source", p.name)
                answers.append(a)
    return answers


def evaluate(answers: list[dict], timings: str) -> list[dict]:
    rows = []
    for a in answers:
        tier = a["tier"]
        delivery = None
        if timings == "synthetic":
            prof = PROFILES[tier]
            main = CandTurn("m", a["text"], synth_words(a["text"], prof, a["id"] + ":m"), started_at=prof["onset_ms"] / 1000)
            m = features([main])
            f = None
            if a.get("follow_up_text"):
                fu = CandTurn("f", a["follow_up_text"], synth_words(a["follow_up_text"], FOLLOW_UP_PROFILE[tier], a["id"] + ":f"))
                f = features([fu])
            signals, sounds_read, eligible = judge(m, f)
            from interviewer.delivery import AnswerDelivery
            delivery = AnswerDelivery(a["question_id"], prof["onset_ms"], m, f, signals, sounds_read, eligible, ["m", "f"])
        elif timings == "real" and a.get("words"):
            main = CandTurn("m", a["text"], a["words"], started_at=a.get("started_at"))
            m = features([main])
            signals, sounds_read, eligible = judge(m, None)
            from interviewer.delivery import AnswerDelivery
            delivery = AnswerDelivery(a["question_id"], a.get("onset_ms"), m, None, signals, sounds_read, eligible, ["m"])
        obs = observe(a["question_id"], a["text"], delivery)
        rows.append({
            "id": a["id"], "tier": tier, "is_ai": tier in AI_TIERS, "words": obs.text.words,
            "verdict": obs.verdict, "text_signals": len(obs.text.signals), "reads_generated": obs.text.reads_generated,
            "sounds_read": bool(delivery and delivery.sounds_read), "evidence": obs.evidence,
        })
    return rows


def rate(rows: list[dict], pred) -> tuple[float | None, float | None, int, int]:
    ai = [r for r in rows if r["is_ai"]]
    hu = [r for r in rows if not r["is_ai"]]
    catch = sum(pred(r) for r in ai) / len(ai) if ai else None
    false_flag = sum(pred(r) for r in hu) / len(hu) if hu else None
    return catch, false_flag, len(ai), len(hu)


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def report(rows: list[dict], timings: str) -> str:
    rng = random.Random(2026)
    coin = {r["id"]: rng.random() < 0.5 for r in rows}
    median_words = sorted(r["words"] for r in rows)[len(rows) // 2]
    detectors = [
        ("never flag (do nothing)", lambda r: False),
        ("always flag", lambda r: True),
        ("coin flip (seeded)", lambda r: coin[r["id"]]),
        (f"long answer only (> {median_words} words)", lambda r: r["words"] > median_words),
        ("text signals only (reads_generated)", lambda r: r["reads_generated"]),
        ("voice signals only (sounds_read)", lambda r: r["sounds_read"]),
        ("either side", lambda r: r["reads_generated"] or r["sounds_read"]),
        ("both sides agree (ai_assisted)", lambda r: r["verdict"] == "ai_assisted"),
    ]
    lines = [f"# AI-assisted answer detector: stress test ({len(rows)} answers, timings={timings})", ""]
    lines.append("| Detector | Catch rate on AI answers | False-flag rate on human answers | n (ai / human) |")
    lines.append("|---|---|---|---|")
    for name, pred in detectors:
        c, f, na, nh = rate(rows, pred)
        lines.append(f"| {name} | {pct(c)} | {pct(f)} | {na} / {nh} |")
    lines.append("")
    lines.append("## By tier (share flagged by each side)")
    lines.append("| Tier | n | text flags | voice flags | both | median text signals |")
    lines.append("|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in rows:
        by[r["tier"]].append(r)
    for tier in (*AI_TIERS, *HUMAN_TIERS):
        rs = by.get(tier, [])
        if not rs:
            continue
        n = len(rs)
        med = sorted(r["text_signals"] for r in rs)[n // 2]
        lines.append(f"| {tier} | {n} | {sum(r['reads_generated'] for r in rs)}/{n} | {sum(r['sounds_read'] for r in rs)}/{n} "
                     f"| {sum(r['verdict'] == 'ai_assisted' for r in rs)}/{n} | {med} |")
    lines.append("")
    lines.append("## Misses and false flags (text side)")
    for r in rows:
        if r["is_ai"] and not r["reads_generated"]:
            lines.append(f"- MISS {r['id']} ({r['tier']}): {r['text_signals']} text signals: {'; '.join(e for e in r['evidence'] if e.startswith('text')) or 'none'}")
        if not r["is_ai"] and r["reads_generated"]:
            lines.append(f"- FALSE FLAG {r['id']} ({r['tier']}): {'; '.join(e for e in r['evidence'] if e.startswith('text'))}")
    lines.append("")
    lines.append("## Read this honestly")
    lines.append("- Human fixtures are synthetic transcripts and the word timings are generated from per-tier profiles. "
                 "The voice-side numbers therefore measure the detector against our own assumptions, not against callers. "
                 "Real redacted calls in fixtures/real/ replace that assumption.")
    lines.append("- The ai_paraphrased tier is the one that matters: a candidate who re-speaks generated text in their own "
                 "words. Text signals are expected to degrade there; the unscripted follow-up is the signal that should hold.")
    lines.append("- human_prepared is the false-flag risk: a rehearsed true story is fluent and structured. A detector that "
                 "flags it is punishing preparation, which is legitimate.")
    lines.append("- Observations only. Nothing here scores a candidate or is said on the call.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timings", choices=["synthetic", "none", "real"], default="synthetic")
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "report.md")
    args = ap.parse_args()
    rows = evaluate(load_fixtures(), args.timings)
    text = report(rows, args.timings)
    print(text)
    args.out.write_text(text + "\n")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
