# AI-assisted answer detector: stress test (60 answers, timings=none)

| Detector | Catch rate on AI answers | False-flag rate on human answers | n (ai / human) |
|---|---|---|---|
| never flag (do nothing) | 0% | 0% | 36 / 24 |
| always flag | 100% | 100% | 36 / 24 |
| coin flip (seeded) | 36% | 54% | 36 / 24 |
| long answer only (> 81 words) | 36% | 67% | 36 / 24 |
| text signals only (reads_generated) | 33% | 4% | 36 / 24 |
| voice signals only (sounds_read) | 0% | 0% | 36 / 24 |
| either side | 33% | 4% | 36 / 24 |
| both sides agree (ai_assisted) | 0% | 0% | 36 / 24 |

## By tier (share flagged by each side)
| Tier | n | text flags | voice flags | both | median text signals |
|---|---|---|---|---|---|
| ai_verbatim | 18 | 12/18 | 0/18 | 0/18 | 3 |
| ai_paraphrased | 18 | 0/18 | 0/18 | 0/18 | 1 |
| human_spontaneous | 18 | 1/18 | 0/18 | 0/18 | 1 |
| human_prepared | 6 | 0/6 | 0/6 | 0/6 | 1 |

## Misses and false flags (text side)
- MISS ai_v_05 (ai_verbatim): 2 text signals: text: no fillers or restarts in the transcript; text: long, uniform sentences (mean 18.0 words, cv 0.19)
- MISS ai_p_01 (ai_paraphrased): 0 text signals: none
- MISS ai_p_02 (ai_paraphrased): 2 text signals: text: no fillers or restarts in the transcript; text: long, uniform sentences (mean 16.4 words, cv 0.13)
- MISS ai_p_03 (ai_paraphrased): 1 text signals: text: long, uniform sentences (mean 20.8 words, cv 0.31)
- MISS ai_p_04 (ai_paraphrased): 2 text signals: text: no fillers or restarts in the transcript; text: long, uniform sentences (mean 16.2 words, cv 0.15)
- MISS ai_p_05 (ai_paraphrased): 1 text signals: text: no fillers or restarts in the transcript
- MISS ai_p_06 (ai_paraphrased): 1 text signals: text: no fillers or restarts in the transcript
- MISS ami_v_01 (ai_verbatim): 1 text signals: text: no fillers or restarts in the transcript
- MISS ami_p_01 (ai_paraphrased): 0 text signals: none
- MISS ami_p_02 (ai_paraphrased): 2 text signals: text: no concrete details or episode markers; text: long, uniform sentences (mean 25.2 words, cv 0.34)
- MISS ami_p_03 (ai_paraphrased): 2 text signals: text: no concrete details or episode markers; text: long, uniform sentences (mean 25.0 words, cv 0.12)
- MISS ami_p_04 (ai_paraphrased): 1 text signals: text: long, uniform sentences (mean 24.0 words, cv 0.09)
- MISS ami_p_05 (ai_paraphrased): 1 text signals: text: long, uniform sentences (mean 24.3 words, cv 0.04)
- MISS ami_p_06 (ai_paraphrased): 1 text signals: text: long, uniform sentences (mean 19.5 words, cv 0.25)
- MISS ami_p_07 (ai_paraphrased): 2 text signals: text: enumerating markers (3); text: long, uniform sentences (mean 20.2 words, cv 0.28)
- MISS ami_v_08 (ai_verbatim): 2 text signals: text: no fillers or restarts in the transcript; text: long, uniform sentences (mean 27.7 words, cv 0.27)
- MISS ami_p_08 (ai_paraphrased): 1 text signals: text: long, uniform sentences (mean 28.0 words, cv 0.35)
- MISS ami_v_09 (ai_verbatim): 2 text signals: text: no fillers or restarts in the transcript; text: long, uniform sentences (mean 25.0 words, cv 0.04)
- MISS ami_p_09 (ai_paraphrased): 1 text signals: text: long, uniform sentences (mean 25.5 words, cv 0.06)
- MISS ami_v_10 (ai_verbatim): 2 text signals: text: no fillers or restarts in the transcript; text: long, uniform sentences (mean 16.3 words, cv 0.24)
- MISS ami_p_10 (ai_paraphrased): 1 text signals: text: long, uniform sentences (mean 15.5 words, cv 0.31)
- MISS ami_p_11 (ai_paraphrased): 2 text signals: text: no concrete details or episode markers; text: long, uniform sentences (mean 24.7 words, cv 0.18)
- MISS ami_v_12 (ai_verbatim): 2 text signals: text: no fillers or restarts in the transcript; text: long, uniform sentences (mean 19.0 words, cv 0.3)
- MISS ami_p_12 (ai_paraphrased): 1 text signals: text: long, uniform sentences (mean 22.3 words, cv 0.3)
- FALSE FLAG ami_h_11 (human_spontaneous): text: no fillers or restarts in the transcript; text: no concrete details or episode markers; text: long, uniform sentences (mean 16.4 words, cv 0.16)

## Read this honestly
- Human fixtures are synthetic transcripts and the word timings are generated from per-tier profiles. The voice-side numbers therefore measure the detector against our own assumptions, not against callers. Real redacted calls in fixtures/real/ replace that assumption.
- The ai_paraphrased tier is the one that matters: a candidate who re-speaks generated text in their own words. Text signals are expected to degrade there; the unscripted follow-up is the signal that should hold.
- human_prepared is the false-flag risk: a rehearsed true story is fluent and structured. A detector that flags it is punishing preparation, which is legitimate.
- Observations only. Nothing here scores a candidate or is said on the call.
