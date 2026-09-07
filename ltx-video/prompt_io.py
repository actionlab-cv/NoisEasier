"""Load user-facing prompt lists without benchmark-specific schemas."""

from __future__ import annotations

import json
from pathlib import Path


def load_user_prompts(prompt: str, prompt_file: str | None) -> list[str]:
    """Return one CLI prompt or a small UTF-8 TXT/JSON prompt list."""
    if not prompt_file:
        cleaned = prompt.strip()
        if not cleaned:
            raise ValueError("--prompt must not be empty")
        return [cleaned]

    path = Path(prompt_file)
    if not path.is_file():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("prompts")
        if not isinstance(payload, list):
            raise ValueError("JSON prompt files must be a list or {'prompts': [...]} object")
        prompts = [item.strip() for item in payload if isinstance(item, str) and item.strip()]
    else:
        prompts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not prompts:
        raise ValueError(f"Prompt file contains no usable prompts: {path}")
    return prompts
