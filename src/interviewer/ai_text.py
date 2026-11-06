"""Text-side signals for an AI-generated answer, and the combined "AI-assisted answer" observation.

Companion to delivery.py (voice side). Delivery asks "was this read aloud?"; this module asks "does the text itself
read as generated?" Neither is proof. Prepared notes are legitimate, fluent speakers exist, and generated text can be
paraphrased. So: observations with evidence, for a recruiter, never a score and never said to the caller.

Everything here is a pure function of the transcript text, so the eval harness (evals/ai_detect) can measure catch
rate and false-flag rate against labelled fixtures, with do-nothing baselines beside every number.
Thresholds are UNCALIBRATED starting points; the harness is how they get calibrated.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from .delivery import FILLERS, AnswerDelivery

THRESHOLDS = {
    "min_words": 25,  # below this, text signals are noise
    "min_signals": 3,  # of the 6 below
    "list_markers_min": 2,
    "concrete_per100_max": 1.0,
    "mean_sentence_min": 14.0,
    "sentence_cv_max": 0.35,  # coefficient of variation of sentence length: generated text is uniform
    "formal_per100_min": 2.0,
    "generic_phrases_min": 2,
}

LIST_MARKERS = {
    "first",
    "firstly",
    "second",
    "secondly",
    "third",
    "additionally",
    "furthermore",
    "moreover",
    "finally",
    "lastly",
    "overall",
    "in conclusion",
    "to summarize",
    "in summary",
}
FORMAL = {
    "ensure",
    "ensuring",
    "utilize",
    "utilizing",
    "prioritize",
    "prioritizing",
    "effectively",
    "appropriate",
    "appropriately",
    "communicate",
    "communicating",
    "collaborate",
    "collaborating",
    "facilitate",
    "implement",
    "implemented",
    "maintain",
    "maintaining",
    "comprehensive",
    "proactively",
    "leverage",
    "optimal",
    "demonstrate",
    "demonstrated",
    "adhere",
    "adhering",
    "escalate",
    "escalating",
    "utilization",
    "foster",
}
GENERIC_PHRASES = [
    "it's important to",
    "it is important to",
    "i always make sure",
    "i make sure to",
    "i believe that",
    "i believe",
    "ultimately",
    "in order to",
    "as a nurse",
    "as a professional",
    "patient safety is",
    "i understand the importance",
    "i am committed to",
    "i strive to",
    "my approach is",
    "i take pride in",
    "i remain calm",
    "clear communication",
    "i would",
    "i always",
]
EPISODIC = [
    "one time",
    "there was",
    "i remember",
    "last year",
    "last month",
    "that night",
    "that day",
    "one night",
    "one shift",
    "this one",
    "this guy",
    "this lady",
    "this patient",
    "a couple",
    "the other",
    "we had a",
    "i had a",
    "so i",
    "and then",
    "turned out",
    "ended up",
    "actually",
]
NUMBER_WORDS = {
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "fifteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "hundred",
    "half",
    "twice",
    "once",
}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z0-9']+")


@dataclass
class TextSignals:
    words: int
    eligible: bool
    fillers: int
    restarts: int
    list_markers: int
    concrete_per100: float
    mean_sentence_len: float
    sentence_cv: float | None
    formal_per100: float
    generic_phrases: int
    episodic_markers: int
    signals: list[str] = field(default_factory=list)
    reads_generated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def text_signals(text: str) -> TextSignals:
    T = THRESHOLDS
    toks = _tokens(text)
    low = [t.lower() for t in toks]
    n = len(toks)
    lower_text = " " + re.sub(r"\s+", " ", text.lower()) + " "

    fillers = sum(1 for t in low if t in FILLERS)
    restarts = sum(
        1 for a, b in zip(low, low[1:]) if a == b and a not in FILLERS
    ) + lower_text.count(" i mean ")
    list_markers = sum(
        1
        for m in LIST_MARKERS
        if (" " + m + " ") in lower_text or (" " + m + ",") in lower_text
    )

    # Concreteness: digits, number words, and capitalised words that are not sentence-initial (names, drugs, units).
    sentence_starts = {0}
    for i, t in enumerate(toks[:-1]):
        if _PUNCT_END_WORD(text, t):
            sentence_starts.add(i + 1)
    concrete = 0
    for i, t in enumerate(toks):
        if (
            t.isdigit()
            or t.lower() in NUMBER_WORDS
            or (t[:1].isupper() and i not in sentence_starts and t.lower() != "i")
        ):
            concrete += 1
    concrete_per100 = round(concrete / n * 100, 1) if n else 0.0

    sents = [s for s in _SENT_SPLIT.split(text.strip()) if _tokens(s)]
    lens = [len(_tokens(s)) for s in sents]
    mean_len = round(statistics.fmean(lens), 1) if lens else 0.0
    cv = (
        round(statistics.pstdev(lens) / mean_len, 2)
        if len(lens) >= 2 and mean_len
        else None
    )

    formal = sum(1 for t in low if t in FORMAL)
    formal_per100 = round(formal / n * 100, 1) if n else 0.0
    generic = sum(
        lower_text.count(" " + p + " ") + lower_text.count(" " + p + ",")
        for p in GENERIC_PHRASES
    )
    episodic = sum(
        lower_text.count(" " + p + " ") + lower_text.count(" " + p + ",")
        for p in EPISODIC
    )

    signals: list[str] = []
    if n >= T["min_words"]:
        if fillers == 0 and restarts == 0:
            signals.append("no fillers or restarts in the transcript")
        if list_markers >= T["list_markers_min"]:
            signals.append(f"enumerating markers ({list_markers})")
        if concrete_per100 <= T["concrete_per100_max"] and episodic == 0:
            signals.append("no concrete details or episode markers")
        if (
            mean_len >= T["mean_sentence_min"]
            and cv is not None
            and cv <= T["sentence_cv_max"]
        ):
            signals.append(f"long, uniform sentences (mean {mean_len} words, cv {cv})")
        if formal_per100 >= T["formal_per100_min"]:
            signals.append(f"formal register ({formal_per100} formal words per 100)")
        if generic >= T["generic_phrases_min"]:
            signals.append(f"generic competency phrasing ({generic} stock phrases)")
    eligible = n >= T["min_words"]
    return TextSignals(
        words=n,
        eligible=eligible,
        fillers=fillers,
        restarts=restarts,
        list_markers=list_markers,
        concrete_per100=concrete_per100,
        mean_sentence_len=mean_len,
        sentence_cv=cv,
        formal_per100=formal_per100,
        generic_phrases=generic,
        episodic_markers=episodic,
        signals=signals,
        reads_generated=eligible and len(signals) >= T["min_signals"],
    )


def _PUNCT_END_WORD(text: str, token: str) -> bool:
    # Cheap: a token is sentence-final if it is followed by sentence punctuation somewhere in the text.
    return bool(re.search(re.escape(token) + r"[.!?]", text))


@dataclass
class Observation:
    question_id: str
    text: TextSignals
    delivery: AnswerDelivery | None
    verdict: (
        str  # "ai_assisted" | "reads_generated" | "sounds_read" | "none" | "ineligible"
    )
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "text": self.text.to_dict(),
            "delivery": self.delivery.to_dict() if self.delivery else None,
        }


def observe(
    question_id: str, answer_text: str, delivery: AnswerDelivery | None
) -> Observation:
    """Combine the text read and the voice read into one observation with its evidence.

    ai_assisted     = text reads generated AND delivery sounds read (both sides agree)
    reads_generated = text only
    sounds_read     = voice only (a prepared script in the candidate's own words, or a fluent speaker)
    """
    ts = text_signals(answer_text)
    voice = bool(delivery and delivery.sounds_read)
    if not ts.eligible and not (delivery and delivery.eligible):
        verdict = "ineligible"
    elif ts.reads_generated and voice:
        verdict = "ai_assisted"
    elif ts.reads_generated:
        verdict = "reads_generated"
    elif voice:
        verdict = "sounds_read"
    else:
        verdict = "none"
    evidence = [f"text: {s}" for s in ts.signals] + (
        [f"voice: {s}" for s in delivery.signals] if delivery else []
    )
    return Observation(
        question_id=question_id,
        text=ts,
        delivery=delivery,
        verdict=verdict,
        evidence=evidence,
    )


def answer_texts(events: list[dict[str, Any]]) -> dict[str, str]:
    """question_id -> the candidate's main-answer text (turns between the question and record_answer), from events."""
    from .delivery import is_question

    out: dict[str, str] = {}
    buf: list[str] = []
    questions_asked = 0
    for ev in sorted(events, key=lambda e: e.get("seq") or 0):
        p = ev.get("payload") or {}
        typ = ev.get("type")
        if typ == "turn" and p.get("role") == "agent":
            if is_question(str(p.get("text") or "")):
                questions_asked += 1
        elif typ == "turn" and p.get("role") == "candidate":
            if questions_asked <= 1:
                buf.append(str(p.get("text") or ""))
        elif typ == "tool_mark_knockout":
            buf, questions_asked = [], 0
        elif typ == "tool_record_answer":
            out[str(p.get("question_id") or "?")] = " ".join(buf).strip()
            buf, questions_asked = [], 0
    return out


def report_lines(observations: list[Observation]) -> list[str]:
    """Human-readable post-call block for the worker log (standalone mode) or the demo screen."""
    lines = ["ANSWER DELIVERY (observations, not scored; never shown to the caller)"]
    for o in observations:
        d = o.delivery
        voice = (
            f"onset {d.onset_ms} ms, {d.main.wpm} wpm, fillers {d.main.fillers_per100}/100, pauses "
            f"{d.main.pauses_per100}/100"
            if d
            else "no timings"
        )
        lines.append(
            f"  [{o.question_id}] {o.verdict.upper():16s} words {o.text.words:3d} | {voice}"
        )
        for e in o.evidence:
            lines.append(f"      - {e}")
    return lines


# ---------- call-level confidence and problem areas (dashboard) ----------
TEXT_SIGNAL_COUNT = 6
VOICE_SIGNAL_COUNT = 5


def answer_confidence(o: Observation) -> float:
    """0..1: how strongly this answer's evidence points at AI assistance. Text and voice each contribute half;
    agreement between the two sides adds a bonus, disagreement caps the score. Ineligible answers score 0."""
    text_frac = len(o.text.signals) / TEXT_SIGNAL_COUNT if o.text.eligible else 0.0
    voice_frac = (
        (len(o.delivery.signals) / VOICE_SIGNAL_COUNT)
        if (o.delivery and o.delivery.eligible)
        else 0.0
    )
    score = 0.5 * text_frac + 0.5 * voice_frac
    if o.verdict == "ai_assisted":
        score = min(1.0, score + 0.15)
    elif o.verdict in ("reads_generated", "sounds_read"):
        score = min(score, 0.6)  # one side only never reads as high confidence
    elif o.verdict == "ineligible":
        score = 0.0
    return round(score, 2)


def call_confidence(observations: list[Observation]) -> dict[str, Any]:
    """Call-level confidence that AI assistance occurred, from the two strongest answers, discounted when few
    answers were eligible. Bands: low < 0.35, medium < 0.6, high otherwise. The band is what a recruiter reads;
    the number is what they can check."""
    scores = sorted((answer_confidence(o) for o in observations), reverse=True)
    eligible = sum(1 for o in observations if o.verdict != "ineligible")
    top = scores[:2]
    raw = sum(top) / len(top) if top else 0.0
    coverage = min(
        1.0, eligible / 3
    )  # three eligible answers = full confidence in the coverage
    score = round(raw * (0.6 + 0.4 * coverage), 2)
    band = "low" if score < 0.35 else "medium" if score < 0.6 else "high"
    flagged = [
        o.question_id
        for o in observations
        if o.verdict in ("ai_assisted", "reads_generated", "sounds_read")
    ]
    return {
        "score": score,
        "band": band,
        "eligible_answers": eligible,
        "answers": len(observations),
        "flagged": flagged,
        "strong": [o.question_id for o in observations if o.verdict == "ai_assisted"],
    }


def problem_areas(
    observations: list[Observation], events: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """What a recruiter should look at first: flagged answers, then call-mechanics issues from the events."""
    out: list[dict[str, str]] = []
    for o in observations:
        if o.verdict == "ai_assisted":
            out.append(
                {
                    "area": o.question_id,
                    "severity": "high",
                    "note": "text reads generated and delivery sounds read; "
                    + "; ".join(o.evidence[:3]),
                }
            )
        elif o.verdict == "reads_generated":
            out.append(
                {
                    "area": o.question_id,
                    "severity": "medium",
                    "note": "transcript reads generated; delivery did not confirm: "
                    + "; ".join(o.text.signals[:3]),
                }
            )
        elif o.verdict == "sounds_read":
            out.append(
                {
                    "area": o.question_id,
                    "severity": "medium",
                    "note": "sounded read; text looks natural (prepared notes?): "
                    + "; ".join(o.delivery.signals[:3] if o.delivery else []),
                }
            )
    ended = next(
        (e.get("payload") or {} for e in events if e.get("type") == "call_ended"), {}
    )
    silence = ended.get("silence_events") or []
    if silence:
        out.append(
            {
                "area": "silence",
                "severity": "low",
                "note": f"{len(silence)} silence event(s): {', '.join(silence)}",
            }
        )
    if (ended.get("interruptions") or 0) >= 2:
        out.append(
            {
                "area": "interruptions",
                "severity": "low",
                "note": f"{ended['interruptions']} agent turns were interrupted",
            }
        )
    if ended.get("guardrail_events"):
        out.append(
            {
                "area": "guardrail",
                "severity": "medium",
                "note": f"{ended['guardrail_events']} sentence(s) replaced before being spoken (pay, offer or protected-class content)",
            }
        )
    for kid, k in (ended.get("knockouts") or {}).items():
        if k.get("result") == "fail":
            out.append(
                {
                    "area": kid,
                    "severity": "high",
                    "note": f"knockout failed: {k.get('evidence', '')[:120]}",
                }
            )
        elif k.get("result") == "unclear":
            out.append(
                {
                    "area": kid,
                    "severity": "medium",
                    "note": f"knockout unclear: {k.get('evidence', '')[:120]}",
                }
            )
    if ended.get("outcome") and ended["outcome"] not in ("completed",):
        out.append(
            {
                "area": "outcome",
                "severity": "medium",
                "note": f"call outcome: {ended['outcome']}",
            }
        )
    return out
