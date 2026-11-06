"""Build interview DispatchMetadata from a compiled role (mirror of apps/web/src/lib/metadata.ts)."""

from __future__ import annotations

from ..metadata import DispatchMetadata


def metadata_from_role(role: dict, *, prompt_text: str, prompt_version: int, first_name: str,
                       call_id: str = "readiness", mode: str = "browser") -> DispatchMetadata:
    return DispatchMetadata.model_validate({
        "schema_version": 1,
        "application_id": f"app-{call_id}", "attempt_id": f"att-{call_id}", "call_id": call_id,
        "mode": mode, "phone_e164": "+15555550100" if mode == "inbound" else None,
        "candidate_identity": f"candidate-app-{call_id}", "candidate_first_name": first_name,
        "role_title": role["title"], "org_name": role["org_name"],
        "plan": {**role["plan"], "plan_source": "template", "knockouts": role["knockouts"], "questions": role["question_bank"]},
        "keyterms": list(dict.fromkeys([first_name, *role.get("keyterms", [])]))[:50],
        "runtime_config": {
            "prompt_version": prompt_version, "prompt_text": prompt_text, "rubric_version": role["version"],
            "question_bank_version": role["question_bank_version"],
            "competencies": [{"id": c["id"], "name": c["name"], "anchors": c["anchors"]} for c in role["competencies"]],
            "faq": role.get("faq", []),
        },
    })
