"""Created: 2026-03-31

Purpose: Defines shared protocol and state types for agent graph nodes.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict


class MemoryProtocol(Protocol):
    """Describes the working-memory contract needed by shared graph nodes."""

    session_id: str
    state: dict[str, Any]
    recent_items: list[dict[str, Any]]

    def add_user_message(self, content: str) -> None:
        """Persists one user message in session working memory."""
        ...

    def add_agent_message(self, content: str) -> None:
        """Persists one agent message in session working memory."""
        ...

    def set_state(self, **kwargs: object) -> None:
        """Stores session-scoped state updates for later turns."""
        ...


class SessionStoreProtocol(Protocol):
    """Describes the session store used to load working memory by session id."""

    def load(self, session_id: str) -> MemoryProtocol:
        """Loads or creates the conversation working memory for a session."""
        ...


class MemoryTargetSpec(TypedDict, total=False):
    """Describes one memory target that a node may retrieve or update.

    This is the state-level configuration shape for memory-aware nodes. It lets
    callers add or change memory targets without changing the node class
    constructor or hardcoding per-agent logic into the node itself.
    """

    type: str
    layer: str
    scope: str
    limit: int
    enabled: bool
    metadata: dict[str, Any]
    agent_id: str
    source_type: str
    source_id: str
    tags: list[str] | str
    query: str
    query_candidates: list[str]
    created_before: str
    created_after: str
    stop_on_first_hit: bool
    min_results: int
    max_queries: int


class AgentState(TypedDict, total=False):
    """Represents the neutral shared state passed between graph nodes.

    The state is intentionally generic so the platform can compose different
    agent graphs from the same node vocabulary. Although some current nodes are
    inspired by a ReAct-style loop, the state itself should not be named after
    a single reasoning pattern.
    """

    session_id: str
    turn_id: str
    trigger_type: str
    user_input: str
    memory: MemoryProtocol | None
    memory_targets: list[MemoryTargetSpec]
    memory_context: dict[str, Any]
    memory_retrievals: list[dict[str, Any]]
    intent: dict[str, Any]
    decision: Any
    memory_updates: list[dict[str, Any]]
    stored_memories: list[Any]
    observation: dict[str, Any] | None
    reflection_feedback: dict[str, Any] | None
    reflection_complete: bool
    error: dict[str, Any] | None
    available_tools: list[Any] | None
    response: str
    steps: int
    route: str
    final: bool
    waiting: bool
    trace: dict[str, Any]
    approval_item: Any
    approval_result: Any
    confidence: float
    channel_result: dict[str, Any]
    routing_context: dict[str, Any]
    response_target: str
    handoff_payload: dict[str, Any]
    additional_targets: list[str]
    memory_helper_trigger: dict[str, Any]
    pending_tool_calls: list[dict[str, Any]]


class NodeUpdate(TypedDict, total=False):
    """Represents a partial state update emitted by one graph node.

    LangGraph merges these updates into the shared state after each node
    finishes. Keeping this as a dedicated alias makes the node contract easier
    to read and document than reusing the full state type everywhere.
    """

    session_id: str
    turn_id: str
    trigger_type: str
    user_input: str
    memory: MemoryProtocol | None
    memory_targets: list[MemoryTargetSpec]
    memory_context: dict[str, Any]
    memory_retrievals: list[dict[str, Any]]
    intent: dict[str, Any]
    decision: Any
    memory_updates: list[dict[str, Any]]
    stored_memories: list[Any]
    observation: dict[str, Any] | None
    reflection_feedback: dict[str, Any] | None
    reflection_complete: bool
    error: dict[str, Any] | None
    available_tools: list[Any] | None
    response: str
    steps: int
    route: str
    final: bool
    waiting: bool
    trace: dict[str, Any]
    approval_item: Any
    approval_result: Any
    confidence: float
    channel_result: dict[str, Any]
    routing_context: dict[str, Any]
    response_target: str
    handoff_payload: dict[str, Any]
    additional_targets: list[str]
    memory_helper_trigger: dict[str, Any]
    pending_tool_calls: list[dict[str, Any]]


ReActState = AgentState
