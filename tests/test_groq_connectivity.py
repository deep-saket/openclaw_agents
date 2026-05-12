from __future__ import annotations

import os
from pathlib import Path

import pytest
from groq import Groq


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@pytest.mark.integration
def test_groq_chat_completion_stream_returns_text() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _load_env_file(repo_root / "agents" / "collection_agent" / ".env")

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GROQ_API_KEY is not set")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": "Who is the chief minister of west bengal",
            }
        ],
        temperature=1,
        max_completion_tokens=256,
        top_p=1,
        stream=True,
        stop=None,
    )

    chunks: list[str] = []
    for chunk in completion:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            chunks.append(delta)

    text = "".join(chunks).strip()
    assert text, "Groq stream returned no text"
