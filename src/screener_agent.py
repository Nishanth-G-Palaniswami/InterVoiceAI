"""Example 2: a recruiter that runs a short phone screen.

Shows a structured conversation: fixed questions, one tool that records each
answer, and a final summary. This is close to what Take2 runs in production.
"""

import json
import logging

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.beta.tools import EndCallTool

from prompts import build_instructions

logger = logging.getLogger("screener-agent")

ROLE = "Weekend Barista at Corner Cup Coffee in Brooklyn"

QUESTIONS = [
    "Are you available to work Saturday and Sunday mornings from six to two?",
    "Tell me about any experience you have making espresso drinks or working in food service.",
    "Describe a time you dealt with an unhappy customer. What did you do?",
    "Are you able to start within the next two weeks?",
]

JOB_PROMPT = f"""
You are Sam, a friendly recruiter running a five-minute phone screen for the
role of {ROLE}.

1. Greet the candidate, confirm their name, and say the screen takes about
   five minutes.
2. Ask these questions in order, one at a time:
{chr(10).join(f"   {i + 1}. {q}" for i, q in enumerate(QUESTIONS))}
3. After each answer, call record_answer with a one-sentence summary. If an
   answer is vague, ask one short follow-up before recording it.
4. After the last question, call finish_screen, thank them, tell them the team
   will follow up by text within two days, and say goodbye.

Fair hiring rules: never ask about or comment on age, race, religion,
disability, pregnancy, family plans, nationality or immigration status, or
anything else unrelated to the job. If the candidate raises one of these, do not
record it. You cannot tell the candidate whether they passed, and you cannot
promise them the job.
"""


class ScreenerAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_instructions(JOB_PROMPT),
            tools=[EndCallTool(ignore_on_enter=True)],
        )
        self.answers: dict[int, str] = {}

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the candidate as Sam, confirm their name, and explain the screen takes about five minutes."
        )

    @function_tool
    async def record_answer(
        self, context: RunContext, question_number: int, summary: str
    ) -> str:
        """Save the candidate's answer to one screening question.

        Args:
            question_number: 1-based number of the question that was answered.
            summary: One neutral sentence summarizing the answer.
        """
        self.answers[question_number] = summary
        logger.info("answer %s: %s", question_number, summary)
        remaining = [n for n in range(1, len(QUESTIONS) + 1) if n not in self.answers]
        return f"Saved. Questions remaining: {remaining or 'none'}."

    @function_tool
    async def finish_screen(self, context: RunContext) -> str:
        """Call once every question has an answer, before saying goodbye."""
        report = {"role": ROLE, "answers": self.answers}
        # In a real product this goes to an ATS. Here it lands in your logs.
        logger.info("SCREEN COMPLETE %s", json.dumps(report))
        return "Screen saved. Thank the candidate and close the call."
