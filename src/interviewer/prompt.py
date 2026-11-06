"""Render the live instructions from runtime_config.prompt_text + plan + state (BUILD_PLAN §5.3)."""

from __future__ import annotations

from .metadata import DispatchMetadata
from .state import InterviewState


def _knockouts(meta: DispatchMetadata) -> str:
    return "\n".join(
        f"   - [{k.id}] \"{k.question}\" Pass if: {k.pass_if} Fail if: {k.fail_if or 'clearly no'}."
        + (f" Clarify once: \"{k.clarify}\"" if k.clarify else "")
        for k in meta.plan.knockouts
    )


def _questions(meta: DispatchMetadata) -> str:
    return "\n".join(
        f"   - [{q.id}] \"{q.question}\"" + (f" Probe once if thin: \"{q.probe}\"" if q.probe else "")
        + (f" If they are silent, re-ask as: \"{q.short}\"" if q.short else "")
        for q in meta.plan.questions
    )


def _faq(meta: DispatchMetadata) -> str:
    return "\n".join(f"- About {f.q}: \"{f.a}\"" for f in meta.runtime_config.faq) or "- (none)"


def render_base(meta: DispatchMetadata) -> str:
    return meta.runtime_config.prompt_text.format(
        org_name=meta.org_name, role_title=meta.role_title, first_name=meta.candidate_first_name,
        knockouts=_knockouts(meta), questions=_questions(meta), faq=_faq(meta),
    )


def render_instructions(meta: DispatchMetadata, state: InterviewState) -> str:
    return render_base(meta) + "\n\n" + state.render()


def opening_line(meta: DispatchMetadata) -> str:
    """Scripted opening: AI disclosure, length, recording stated (not asked), time check. One breath."""
    who = (f"Hi {meta.candidate_first_name}, thanks for calling. This is the AI interviewer for {meta.org_name}"
           if meta.mode == "inbound" else
           f"Hi {meta.candidate_first_name}, this is the AI interviewer from {meta.org_name}")
    return (
        f"{who} about the {meta.role_title} position. It takes about six to eight minutes. "
        "This call is recorded so the hiring team can review your application. "
        + ("Ready to begin?" if meta.mode == "inbound" else "Is now still a good time?")
    )
