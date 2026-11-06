You write test lines for an automated readiness check of an AI phone interviewer. A simulated candidate will say these lines, word for word, in reply to the interviewer's questions for the role below. Each line must be something a real candidate could say on the phone: one or two short spoken sentences, first person, no names other than the candidate's, no phone numbers, no pay figures.

For EVERY knockout id, write:
- `pass`: a clear yes that clearly meets the pass rule, with one concrete detail (for a licence or certification, name the jurisdiction from the rule).
- `fail`: a clear no that clearly meets the fail rule.
- `unclear`: a plausible borderline answer that is neither a clear pass nor a clear fail (another jurisdiction, pending renewal, "mostly"), so the interviewer should ask the clarify line.
- `question_back`: instead of answering, a short logistics question about this knockout that the approved facts can answer (for example about hours or which certificate counts). Must not contain a yes or no.

For EVERY question id, write:
- `strong`: a specific two- or three-sentence example that would score 4 or 5 on that competency's anchors: what happened, what the candidate did, the outcome.
- `thin`: one vague sentence that would score 1 or 2.

`first_name`: a common, gender-neutral first name.

Use each id exactly once, exactly as given.

# Role
{{role}}
