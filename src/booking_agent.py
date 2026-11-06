"""Example 1: a receptionist that books appointments.

Shows the core pattern: a focused prompt, tools that read and write state, and
read-back confirmation before committing. The "calendar" is an in-memory dict;
swap in a real API if you have time.
"""

import logging
from datetime import date, timedelta

from livekit.agents import Agent, RunContext, function_tool
from livekit.agents.beta.tools import EndCallTool

from prompts import build_instructions

logger = logging.getLogger("booking-agent")

JOB_PROMPT = """
You are Riley, the receptionist at Harbor Street Dental in New York. You answer
the phone to book cleaning and checkup appointments.

Your goal on every call: book one appointment and confirm it.

1. Greet the caller and ask how you can help.
2. Get their full name and a callback phone number.
3. Ask which day works. Use check_availability to find open times, then offer
   at most two options.
4. Read the name, day and time back, and only after they say yes, call
   book_appointment.
5. Tell them they are booked, ask if there is anything else, then say goodbye.

You only book cleanings and checkups. For tooth pain, swelling or bleeding, tell
them to call nine one one if it is an emergency, otherwise offer the earliest
slot. You cannot give medical advice, discuss prices, or share anything about
other patients.
"""


def _demo_calendar() -> dict[str, list[str]]:
    """Open slots for the next five weekdays, keyed by ISO date."""
    slots: dict[str, list[str]] = {}
    day = date.today()
    while len(slots) < 5:
        day += timedelta(days=1)
        if day.weekday() < 5:
            slots[day.isoformat()] = ["9:00 AM", "11:30 AM", "2:00 PM", "4:30 PM"]
    return slots


class BookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=build_instructions(JOB_PROMPT),
            tools=[EndCallTool(ignore_on_enter=True)],
        )
        self.calendar = _demo_calendar()
        self.bookings: list[dict[str, str]] = []

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions="Greet the caller as Riley from Harbor Street Dental and ask how you can help."
        )

    @function_tool
    async def check_availability(self, context: RunContext, day: str) -> str:
        """Look up open appointment times on a given day.

        Args:
            day: The requested day as an ISO date (YYYY-MM-DD). Convert words like
                "Tuesday" or "tomorrow" to a date first, using today's date from
                your instructions.
        """
        logger.info("check_availability day=%s", day)
        open_slots = self.calendar.get(day)
        if not open_slots:
            options = ", ".join(list(self.calendar)[:3])
            return f"No openings on {day}. Days with openings: {options}."
        return f"Open times on {day}: {', '.join(open_slots)}."

    @function_tool
    async def book_appointment(
        self, context: RunContext, name: str, phone: str, day: str, time: str
    ) -> str:
        """Book the appointment. Only call after the caller confirmed the read-back.

        Args:
            name: Caller's full name.
            phone: Callback number, digits only.
            day: ISO date (YYYY-MM-DD).
            time: One of the open times, e.g. "11:30 AM".
        """
        logger.info("book_appointment %s %s %s", name, day, time)
        if time not in self.calendar.get(day, []):
            return "That time is no longer open. Check availability again."
        self.calendar[day].remove(time)
        self.bookings.append({"name": name, "phone": phone, "day": day, "time": time})
        return f"Booked {name} on {day} at {time}."
