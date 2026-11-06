"""Pre-TTS guardrail (BUILD_PLAN §5.2, CLAUDE.md rule 5).

`check_text` is the single rule function. `SentenceGate` turns a stream of text deltas into
checked sentences. Both are pure so they can be unit-tested without LiveKit.

Limits (documented, not claimed away): rules are regex/keyword based, so paraphrases that avoid
every listed pattern can pass. The adversarial persona set is the test of record.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

SAFE_LINE = "I can't speak to that, but the recruiting team can follow up."

_TAG = re.compile(r"<[^>]{0,200}>")  # expressive-mode markup; stripped before checking

# Only these tags may reach TTS: Cartesia's markup (livekit tts/_provider_format.py `_CARTESIA_TAGS`) plus the
# SDK's own `<expr/>` transcript marker. Anything else is dropped before speech: on live call dfa187cd the model
# wrote tool calls as tags (`<default_api:mark_knockout{...}/>`) under expressive mode.
_SPEAKABLE_TAGS = {"emotion", "speed", "volume", "break", "spell", "expr"}
_ANY_TAG = re.compile(r"</?\s*([A-Za-z_][\w:.-]*)[^>]{0,800}>")
TOOL_NAMES = ("mark_knockout", "record_answer", "create_followup", "request_reschedule", "recording_refused",
              "escalate_to_human", "wrong_person", "end_call")
_TOOL_TEXT = re.compile(
    rf"(?:\bdefault_api[:.]\s*)?\b(?:{'|'.join(TOOL_NAMES)})\b\s*[({{][^)}}]{{0,800}}[)}}]|\bdefault_api[:.]\S*", re.I)


def scrub_markup(text: str) -> str:
    """Remove tool-call-shaped text and tags TTS does not understand; keep speakable markup."""
    out = _ANY_TAG.sub(lambda m: m.group(0) if m.group(1).lower() in _SPEAKABLE_TAGS else "", text)
    return _TOOL_TEXT.sub("", out)

_NUM_WORDS = (
    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|k\b)"
)

RULES: list[tuple[str, re.Pattern[str]]] = [
    # Pay figures: currency, "$40", "40 an hour", "forty dollars", "80k", ranges.
    ("pay_figure", re.compile(
        r"(\$\s?\d)|(\b\d[\d,.]*\s*(?:k\b|dollars?|bucks|an hour|per hour|/\s?hr|a year|per year|annually))"
        rf"|(\b{_NUM_WORDS}[\w\s-]{{0,20}}\b(?:dollars?|bucks|an hour|per hour|a year))",
        re.I)),
    ("pay_disclosure", re.compile(
        r"\b(?:the )?(?:pay|salary|rate|wage|compensation|differential|bonus)\b[^.?!]{0,40}\b(?:is|are|will be|starts at|ranges?)\b[^.?!]{0,30}\d",
        re.I)),
    ("offer_promise", re.compile(
        r"\b(?:you(?:'re| are| have)?\s+(?:been\s+)?(?:hired|got the job|getting the job|selected|a shoo-in))"
        r"|\b(?:we(?:'d| would| will|'ll)\s+(?:love to\s+)?(?:hire|offer)\s+you)"
        r"|\b(?:offer(?:ing)? you (?:the|a) (?:job|position|role))"
        r"|\b(?:top candidate|best candidate)\b",
        re.I)),
    ("start_date", re.compile(
        r"\b(?:you (?:can|could|will|'ll) start|your start date (?:is|will be)|start(?:ing)? (?:on|next) (?:monday|tuesday|wednesday|thursday|friday|week|month))",
        re.I)),
    ("protected_class", re.compile(
        r"\b(?:how old are you|your age|are you (?:married|pregnant|religious|disabled)|"
        r"do you have (?:kids|children)|plan(?:ning)? to have (?:kids|children)|where are you (?:really )?from|"
        r"what(?:'s| is) your (?:religion|nationality|ethnicity|race)|your (?:pregnancy|disability|religion|race|ethnicity))",
        re.I)),
    ("advice", re.compile(
        r"\b(?:you should (?:take|stop taking|increase|decrease) (?:your )?(?:medication|meds|dose)|"
        r"legally,? you (?:can|should|must)|you should (?:sue|file a lawsuit))",
        re.I)),
]


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    text: str
    rule: str | None = None


def check_text(sentence: str) -> CheckResult:
    scrubbed = scrub_markup(sentence)
    plain = _TAG.sub("", scrubbed)
    for name, pat in RULES:
        if pat.search(plain):
            return CheckResult(False, SAFE_LINE, name)
    if scrubbed != sentence:  # the rest of the sentence is fine; only the leaked markup is dropped
        return CheckResult(False, scrubbed.strip(), "tool_text")
    return CheckResult(True, sentence, None)


_BOUNDARY = re.compile(r"([.!?…]+[\"')\]]*\s+|\n+)")


@dataclass
class SentenceGate:
    """Buffers text deltas until a sentence boundary, a token cap, or a time cap, then checks.

    feed() returns the list of CheckResults ready to release (possibly empty).
    flush() checks and returns whatever is buffered (end of stream).
    drop() discards the buffer (interruption / cancellation): never release it later.
    """

    max_tokens: int = 40
    max_wait_s: float = 0.6
    clock: callable = field(default=time.monotonic)
    _buf: str = ""
    _started: float | None = None
    replaced_any: bool = False
    first_release_at: float | None = None

    def feed(self, delta: str) -> list[CheckResult]:
        if not delta:
            return []
        if self._started is None:
            self._started = self.clock()
        self._buf += delta
        out: list[CheckResult] = []
        while True:
            m = next((m for m in _BOUNDARY.finditer(self._buf) if not self._inside_tag(m.end())), None)
            if not m:
                break
            sentence, self._buf = self._buf[: m.end()], self._buf[m.end():]
            out.append(self._check(sentence))
        if self._buf and (len(self._buf.split()) >= self.max_tokens
                          or (self._started is not None and self.clock() - self._started >= self.max_wait_s)):
            out.append(self._check(self._buf))
            self._buf = ""
        if not self._buf:
            self._started = None
        return out

    def flush(self) -> list[CheckResult]:
        if not self._buf.strip():
            self._buf = ""
            return []
        res = [self._check(self._buf)]
        self._buf = ""
        self._started = None
        return res

    def drop(self) -> None:
        self._buf = ""
        self._started = None

    def _inside_tag(self, idx: int) -> bool:
        head = self._buf[:idx]
        return head.rfind("<") > head.rfind(">")

    def _check(self, sentence: str) -> CheckResult:
        r = check_text(sentence)
        if not r.ok:
            self.replaced_any = True
            # keep spacing so the next sentence doesn't run into the safe line
            r = CheckResult(False, r.text + " ", r.rule)
        if self.first_release_at is None:
            self.first_release_at = self.clock()
        return r
