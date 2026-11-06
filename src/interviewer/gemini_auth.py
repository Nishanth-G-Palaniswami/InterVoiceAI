"""Gemini access selection (BUILD_PLAN §2 "Gemini access"). Vertex AI when GOOGLE_GENAI_USE_VERTEXAI is set,
else an AI Studio GOOGLE_API_KEY. The google plugin and google-genai read the env themselves; this only
materializes hosted credentials (GOOGLE_CREDENTIALS_JSON) as a file, since both SDKs load ADC from a path."""

from __future__ import annotations

import os
import tempfile


def use_vertex() -> bool:
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true")


def configure() -> None:
    raw = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if use_vertex() and raw and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        fd, path = tempfile.mkstemp(prefix="gcp-sa-", suffix=".json")  # created 0600
        with os.fdopen(fd, "w") as f:
            f.write(raw)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


def configured() -> bool:
    if use_vertex():
        return bool(os.environ.get("GOOGLE_CLOUD_PROJECT")) and bool(
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GOOGLE_CREDENTIALS_JSON"))
    return bool(os.environ.get("GOOGLE_API_KEY"))
