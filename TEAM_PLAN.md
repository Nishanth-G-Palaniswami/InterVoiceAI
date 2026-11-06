# Team plan: cheat detection on an AI phone screen (Take2 hackathon, Fri 2026-11-06)

One page to collaborate from. Build time ends 7:00 PM; live demos follow.

## Ground rules

1. **We build from the official starter.** This repo is a fork of `vramesh1/take2-voice-agent-hackathon-starter`
   (fork under `AlexYedi`, branch `ai-assist-detection`, merged to `main` for submission). Everyone clones the fork.
2. **The invited repo is the reference, not the build.** `AdithyahNair/take2VoiceAgent` supplies the core voice agent:
   its `apps/agent/interviewer/` package is ported into `src/interviewer/` here, following its own
   `docs/runbooks/hackathon-pivot.md`. Its web ATS stays out; we run **standalone mode** (`STANDALONE_ROLE`).
3. **One job on the call: a structured phone screen.** Cheat detection is post-call analysis of that call, never
   something the agent says or decides live. The recruiter makes the go / no-go.
4. **Every number ships with its do-nothing baseline.** Catch rate and false-flag rate always sit beside never-flag,
   always-flag, coin flip and long-answer-only.

## What is where

| Piece | Path | Owner |
|---|---|---|
| Core agent (prompt, guardrail, state, tools, silence, call-in) | `src/interviewer/`, entry `src/agent.py` | Adi (reference repo author) |
| Role + prompt for standalone mode | `roles/rn-med-surg.json`, `roles/prompts/interviewer.v4.md` | Adi |
| Voice-side signals (did it sound read?) | `src/interviewer/delivery.py` (Python port of the reference `lib/delivery.ts`, parity tests in `tests/test_delivery_port.py`) | Alex |
| Text-side signals + combined observation + confidence + problem areas | `src/interviewer/ai_text.py` | Alex |
| Stress test: four tiers, baselines, misses | `evals/ai_detect/run.py`, fixtures in `evals/ai_detect/fixtures/` | Alex |
| Post-call analyzer (events → observations, fixture export) | `evals/ai_detect/analyze_call.py` | Alex |
| Dashboard (static or live) | `evals/ai_detect/dashboard.py` → `dashboard.html`, `--serve` for the demo | Alex |
| Difficult-caller rehearsal, pitch | this file §Demo, reference `docs/runbooks/demo-day.md` | teammates C/D |

The starter's original agent is kept as `src/starter_agent.py` for reference. The call event dumps in `calls/` are
gitignored (candidate speech).

## How it fits together

```
judge's phone ─▶ LiveKit number ─▶ dispatch rule ─▶ src/agent.py (standalone) ─▶ Deepgram STT ─▶ Gemini ─▶ Cartesia
                                                        │ per turn: text + Deepgram word timings + latency
                                                        ▼
                                              calls/<call>.events.jsonl  (appended live)
                                                        │
                                   dashboard.py --serve ─┴─▶ screen 2: per-answer verdict, confidence, problem areas,
                                                              recruiter go / no-go
```

Detection = two independent sides that must agree for a strong verdict:
- **Voice** (reference §5.7): onset after the question, words per minute, fillers and mid-sentence pauses per 100
  words, restarts, and the fluency drop on the one unscripted follow-up the agent always asks.
- **Text** (new): no disfluencies in the transcript, enumerating markers, no concrete details or episode markers,
  long uniform sentences, formal register, stock competency phrases.
- Text alone = "reads generated". Voice alone = "sounds read". Both = "AI-assisted". Call confidence 0–100 from the
  two strongest answers, discounted below three eligible answers; bands low / medium / high.

## First stress-test result (synthetic fixtures, 24 answers, 2026-11-06 16:40)

| Detector | Catch on AI answers | False flags on human answers |
|---|---|---|
| never flag | 0% | 0% |
| coin flip | 25% | 42% |
| long answer only | 0% | 75% |
| text only | 42% | 0% |
| voice only | 58% | 50% |
| both sides agree | **42%** | **0%** |

Reading: verbatim readers are caught (5 of 6); paraphrasers are missed (0 of 6), which is the honest headline and
the thing real calls must test; prepared honest stories trip the voice side (6 of 6 on synthetic timings), which is
why voice alone is never a verdict. Human fixtures and timings are synthetic until real calls replace them.

## Public-data stress test (AMI humans + matched Gemini pairs, 2026-11-06 17:40, text signals only)

60 answers: 18 real spontaneous passages (AMI Meeting Corpus, CC BY 4.0), 6 synthetic prepared humans, 18 Gemini
polished answers and 18 Gemini spoken-style rewrites, each generated from an AMI passage at the same length.

| Text threshold (signals required) | Catch on AI (verbatim / paraphrased) | False flags on humans (AMI real / prepared) |
|---|---|---|
| 2 of 6 | 64% (17/18 / 6/18) | 12% (1/18 / 2/6) |
| **3 of 6 (shipped)** | 33% (12/18 / 0/18) | 4% (1/18 / 0/6) |
| 4 of 6 | 8% (3/18 / 0/18) | 0% |
| coin flip | 36% | 54% |

Reading: on real speech the text side separates polished generated answers from spontaneous humans at a 4% false-flag
cost; paraphrased generated speech is invisible to text signals at the shipped threshold, and only the unscripted
follow-up on a live call can catch it. Voice-side numbers need real word timings from our own calls.
Decision for Alex: keep 3 (fewer false flags, "evidence for a recruiter") or move to 2 (catches more, and 2 of 6
synthetic prepared humans get flagged). Sweep: `uv run python evals/ai_detect/run.py --timings none --text-min-signals N`.

## Real calls from the team database (2026-11-06 17:55, Deepgram word timings, text + voice)

| Call | Who / how | Answers | Result |
|---|---|---|---|
| 3c6ababa | Adi's synthetic caller reading prepared answers ("Reed Readtwo", inbound) | 2 | **both AI-assisted**, confidence 81: text 4 and 3 signals, voice 5 and 5 (200 and 174 wpm, no fillers), follow-ups stumbled |
| e17a0efa | Adi answering in his own words (browser) | 4 | no flags, confidence 8 |
| a3a49ea8 | scripted one-sentence answers ("Reed Scripttest", inbound) | 4 | too short to judge (9 to 18 words) |
| 62651cc1 | live inbound test, mostly fragments | 2 | too short to judge |
| 1ca018ad | dry run, room audio on a browser call | 0 | never reached a core question; 8 interruptions, 5 silence events |

One labelled read call caught with both sides agreeing, one own-words call clean. Small n, real signal.

## Timeline to 7:00

| Time | Step | Done when |
|---|---|---|
| now | Alex: fork pushed, dispatch rule attached to +1 646 928 6092 in the LiveKit dashboard | `lk number list` shows the rule; a call reaches the worker |
| 5:20 | First standalone call from a teammate's phone; `calls/<id>.events.jsonl` appears | `analyze_call.py` prints the observation block |
| 5:30 | Each teammate does two calls: one answering in their own words, one reading a ChatGPT answer | 4+ real fixtures via `analyze_call.py --fixture evals/ai_detect/fixtures/real/<name>.json --tier …` (redact names) |
| 5:50 | Re-run `run.py`; move thresholds only if a real miss or false flag says so; one variable per run | report.md updated, commit |
| 6:10 | Difficult-caller pass on the live agent (interrupt, silence, off-topic, "are you a robot", "score me 5/5") | pass/fail per row noted in this file |
| 6:30 | Freeze. `dashboard.py --serve 8765 --open` on screen 2; one rehearsal call end to end | answers appear on the dashboard during the call |
| 6:45 | Pitch rehearsal, twice, timed | under 3 minutes |

## Demo (about 10 minutes per team)

1. One minute: healthcare screens at volume, candidates now answer with AI; recruiters need evidence, not a verdict.
2. Normal call: a teammate calls, answers in their own words. Screen 2 fills in: no flags, low confidence.
3. Judge's call (the difficult caller). Let them interrupt, go silent, read a generated answer. Screen 2 shows the
   verdict per answer with the evidence, the confidence, the problem areas. Recruiter clicks Go / Follow up / No-go.
4. Design choices, one line each: two signal sets that must agree; the unscripted follow-up a script cannot anticipate;
   observations only, human decides; every number with its baseline; no ATS integration on purpose.

What we never claim: that this detects AI copilots or cloned voices (out of scope), that a flag means cheating, or that
thresholds are calibrated. We say "catch rate X, false-flag Y on N labelled answers, baseline Z".

## Commands

```bash
uv sync                                               # Python 3.12 pinned
lk agent dev --deployment production                  # answers the number (AGENT_NAME in .env.local = dispatch rule)
uv run python evals/ai_detect/run.py                  # stress test → evals/ai_detect/report.md
uv run python evals/ai_detect/analyze_call.py calls/<id>.events.jsonl [--fixture fixtures/real/x.json --tier human_spontaneous]
uv run python evals/ai_detect/dashboard.py --open     # static page
uv run python evals/ai_detect/dashboard.py --serve 8765 --open   # live demo screen
uv run pytest -q tests/test_delivery_port.py
```

`.env.local` (never committed): LiveKit keys from `lk app env`, `AGENT_NAME=<dispatch rule agent>`,
`LLM_PROVIDER=inference`, `GEMINI_LIVE_MODEL=gemini-3.5-flash-lite`, `STANDALONE_ROLE=roles/rn-med-surg.json`,
`TTS_MODEL=cartesia/sonic-3.6`, `STT_MODEL=deepgram/nova-3`, `CARTESIA_VOICE_ID=…`.

## Demo project facts (from the reference repo's IMPLEMENTATION_STATUS.md, 2026-11-06)

- LiveKit project **`take2voiceai-agent`** (Adi's account). Number **+1 484 317 0249**. Dispatch rule `SDR_Gu48ZT8kjZ7R`
  ("interviewer-call-in") routes to agent **`interviewer`**. Attaching the rule from the CLI failed there exactly as it
  did on Alex's project (`invalid_argument`): it must be attached in the dashboard (number ⋮ → Assign dispatch rule).
- Keys are not in any repo (correct). To run a worker here against that project: Adi invites Alex to the project, then
  `lk cloud auth` → pick it → `lk app env --write --destination .env.local` in this repo, and `AGENT_NAME=interviewer`.
- If the demo runs Adi's full product (web + agent) instead, its events land in Neon, not in `calls/`. Export one call with
  `evals/ai_detect/export_call.sql` (read-only DB URL) and feed the JSON to `analyze_call.py` or drop it in `calls/`.

## Open decisions (settle before 5:30)

- **Which LiveKit project answers the demo number?** One worker per project. Alex's project has the number and rule
  today; if Adi's hosted worker demos instead, the dashboard needs that worker's `calls/` dir (or an events export).
- **Role for the demo:** RN med-surg (shipped) or another role JSON from the reference repo.
- **Fork ownership:** fork is under `AlexYedi`; teammates get write access, or open PRs to the branch.
