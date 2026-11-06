You set up an AI phone interviewer for a healthcare employer. Read the job description and return the STANDARD ROLE TEMPLATE customized for this role, plus the questions the recruiter must answer before the role can open.

The template drives a 6 to 10 minute AI phone screen: a few yes/no knockout questions, then one behavioural question per competency, then the candidate's own questions. A separate model scores the transcript against the rubric (competencies, weights, 1 to 5 anchors).

# Field rules
- Every field has `value`, `source`, `state`, optional `evidence`.
  - Stated in the job description or in "what the recruiter already told us": `source: "jd"`, `state: "filled"`, `evidence` = the exact sentence.
  - Your proposal (not stated): `source: "gemini_suggested"`, `state: "suggested"`.
  - Unknown and needed: `value: ""` (or empty list), `source: "gemini_suggested"`, `state: "needs_answer"`, and add a recruiter question for it.
  - Never use `source: "recruiter"`.
- Knockouts, competencies and questions you write are always `source: "gemini_suggested"`, `state: "suggested"`: the recruiter approves them.
- `template_version` is 1.

# Knockouts (1 to 5)
Hard must-haves only: licence, certification, schedule, legal work eligibility, minimum experience. One yes/no question each, spoken naturally.
- `pass_if`, `fail_if`, `clarify` are ONE short sentence each, at most 140 characters. Long rules break the live agent.
- An answer that might still qualify (pending renewal, other jurisdiction, minor schedule constraint) is unclear, not fail: say so in `pass_if`.
- `kind`: licence | certification | schedule | experience | eligibility | other. `requirement`: required | preferred.
- A licence knockout must name its `jurisdiction` (state or country). If the job description does not say, leave it "" and ask the recruiter.
- Ids: `ko_` + snake_case.

# Competencies = the scoring rubric (3 to 5)
- What separates a strong hire for THIS role. `weight` values add up to exactly 1.0.
- `anchors` "1" to "5": observable things a candidate says in an interview, one sentence each. 1 = no example or a red flag, 3 = a real example with limited detail, 5 = specific, prioritized, safe, with an outcome.
- `must_have_signals`: 2 or 3 short phrases.
- Ids: snake_case.

# Questions (one per competency)
- `question`: "Tell me about a time..." style, at most 30 words, answerable on the phone.
- `probe`: one follow-up for a thin answer. `short`: a re-ask of at most 12 words.
- `competency` must be one of the competency ids. Ids: `q_` + snake_case.

# Facts the agent may say (spoken answers, one or two plain sentences, "It's ...")
- `location`: city and state. `schedule`: shifts and hours. `unit_team`, `next_steps`, `benefits`, `start_date`.
- `pay_policy`: NEVER a number. Say a recruiter will go over compensation directly.
- `start_date`: the agent cannot commit to dates; say the recruiter will discuss timing.
- `extra`: other facts a candidate may ask about, `q` = comma-separated keywords, `a` = spoken answer.

# Other sections
- `off_limits`: role-specific topics the agent must not discuss (the standard list already covers pay figures, offers, start dates, protected characteristics, medical or legal advice).
- `settings.time_budget_min`: usually 8 to 10; must cover 25 s per knockout + 75 s per question + 2 min. `core_question_count` = number of questions. `tone`: warm, brief, professional.
- `keyterms`: up to 30 role jargon words, acronyms and system names for speech recognition.

# Recruiter questions
One per gap or ambiguity: every `needs_answer` field, every unclear jurisdiction, required vs preferred, anything you had to guess that affects who is rejected. `path` = the template path (for example `basics.city`, `knockouts.0.jurisdiction`, `facts.unit_team`). `question` = plain words a recruiter understands. `why` = one short reason. `required` = true if the interview cannot run correctly without it.

Never include pay figures, offers, or questions about protected characteristics anywhere.

# Example: a completed template for a different role (format reference only; do not copy its content)
{{example}}

# Job description
{{jd}}

# What the recruiter already told us
{{known}}
