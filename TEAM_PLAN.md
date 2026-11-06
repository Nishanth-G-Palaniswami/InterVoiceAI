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

## Open decisions (settle before 5:30)

- **Which LiveKit project answers the demo number?** One worker per project. Alex's project has the number and rule
  today; if Adi's hosted worker demos instead, the dashboard needs that worker's `calls/` dir (or an events export).
- **Role for the demo:** RN med-surg (shipped) or another role JSON from the reference repo.
- **Fork ownership:** fork is under `AlexYedi`; teammates get write access, or open PRs to the branch.
