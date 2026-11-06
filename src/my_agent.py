"""YOUR AGENT. Start here.

Copy the shape of booking_agent.py or screener_agent.py:
  1. Write a JOB_PROMPT for one job, with numbered steps and clear limits.
  2. Add a tool for every action the agent takes (look something up, save
     something, check a rule). The docstring is what the LLM reads.
  3. Set AGENT_PROFILE=mine in .env.local and run `lk agent console`.
"""

import logging

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.beta.tools import EndCallTool

from prompts import build_instructions

logger = logging.getLogger("my-agent")

JOB_PROMPT = """
You are <NAME>, <WHO YOU WORK FOR>. You answer the phone to <THE ONE JOB>.

Your goal on every call: <WHAT DONE LOOKS LIKE>.

1. <First step>
2. <Second step>
3. Confirm the key details out loud before you save anything.
4. Say goodbye and end the call.

You cannot <THINGS OUT OF SCOPE>.
"""


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_instructions(JOB_PROMPT),
            tools=[EndCallTool(ignore_on_enter=True)],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the caller and ask how you can help."
        )

    @function_tool
    async def save_request(self, context: RunContext, details: str) -> str:
        """Save what the caller asked for. Replace this with your own tool.

        Args:
            details: One sentence describing the request.
        """
        logger.info("saved request: %s", details)
        return "Saved."
