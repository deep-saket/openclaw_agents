"""Local file-backed repository for Collection Agent conversation and tool logs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.schemas.domain import ToolExecutionLog
from src.schemas.messages import ConversationMessage


@dataclass(slots=True)
class CollectionRepository:
    """Persists working conversation state and tool logs under `runtime/`."""

    runtime_dir: Path
    _messages: dict[str, list[ConversationMessage]] = field(default_factory=dict)
    _states: dict[str, dict[str, object]] = field(default_factory=dict)
    messages_path: Path = field(init=False)
    states_path: Path = field(init=False)
    tool_logs_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.messages_path = self.runtime_dir / "conversation_messages.json"
        self.states_path = self.runtime_dir / "conversation_states.json"
        self.tool_logs_path = self.runtime_dir / "tool_logs.json"
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        messages_payload = self._read_json(self.messages_path, {})
        states_payload = self._read_json(self.states_path, {})
        self._messages = {
            session_id: [ConversationMessage.model_validate(row) for row in rows]
            for session_id, rows in dict(messages_payload).items()
        }
        self._states = {
            str(session_id): dict(state)
            for session_id, state in dict(states_payload).items()
        }
        if not self.tool_logs_path.exists():
            self._write_json(self.tool_logs_path, [])

    @staticmethod
    def _read_json(path: Path, default: object) -> object:
        if not path.exists():
            return default
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return default
        return json.loads(content)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _flush_messages(self) -> None:
        payload = {
            session_id: [message.model_dump(mode="json") for message in messages]
            for session_id, messages in self._messages.items()
        }
        self._write_json(self.messages_path, payload)

    def _flush_states(self) -> None:
        self._write_json(self.states_path, self._states)

    def list_conversation_messages(self, session_id: str) -> list[ConversationMessage]:
        return list(self._messages.get(session_id, []))

    def get_conversation_state(self, session_id: str) -> dict[str, object] | None:
        state = self._states.get(session_id)
        if state is None:
            return None
        return dict(state)

    def add_conversation_message(self, message: ConversationMessage) -> None:
        self._messages.setdefault(message.session_id, []).append(message)
        self._flush_messages()

    def save_conversation_state(self, session_id: str, state: dict[str, object]) -> None:
        self._states[session_id] = dict(state)
        self._flush_states()

    def reset_conversation_session(self, session_id: str) -> None:
        self._messages.pop(session_id, None)
        self._states.pop(session_id, None)
        self._flush_messages()
        self._flush_states()

    def save_tool_log(self, log: ToolExecutionLog) -> None:
        logs = list(self._read_json(self.tool_logs_path, []))
        logs.append(log.model_dump(mode="json"))
        self._write_json(self.tool_logs_path, logs)


__all__ = ["CollectionRepository"]
