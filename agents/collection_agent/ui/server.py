"""Collection Agent debug UI server.

Purpose: Provides a local web UI for text-mode testing with execution timeline,
thinking events, and graph-state snapshots.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agents.collection_agent.agent import CollectionAgent
from agents.collection_agent.main import DEFAULT_CONFIG_PATH, build_llm, load_collection_config, load_env_file
from agents.collection_agent.repository import CollectionRepository
from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_memory_helper_agent.agent import CollectionMemoryHelperAgent
from agents.collection_memory_helper_agent.repository import CollectionMemoryRepository
from agents.discount_planning_agent.agent import DiscountPlanningAgent


STATIC_DIR = Path(__file__).resolve().parent


class InMemoryTraceSink:
    """Captures per-turn trace events for the debug UI."""

    def __init__(
        self,
        *,
        hop_index: int | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.events: list[dict[str, Any]] = []
        self.hop_index = hop_index
        self.event_callback = event_callback

    def emit(self, event: dict[str, Any]) -> None:
        safe_event = _json_safe(event)
        self.events.append(safe_event)
        if callable(self.event_callback):
            self.event_callback(
                {
                    "event": "trace_event",
                    "payload": {
                        "hop": self.hop_index,
                        "trace_event": safe_event,
                    },
                }
            )


class RunTurnRequest(BaseModel):
    """Input payload for one UI chat turn."""

    message: str = Field(min_length=1)
    session_id: str = Field(default="collection-ui-session")
    soft_cap: int = Field(default=10, ge=1)
    hard_cap: int = Field(default=50, ge=1)
    timeout_seconds: float = Field(default=20.0, ge=1.0)
    sender: str = Field(default="customer", description="customer|admin")


class SessionStateResponse(BaseModel):
    """Session snapshot response for right-side state panel refresh."""

    session_id: str
    conversation_state: dict[str, Any]
    working_memory_state: dict[str, Any]


class StartConversationRequest(BaseModel):
    """Input payload for demo-user initiated collection conversation."""

    user_code: str = Field(min_length=1, description="One of: user_a, user_b, user_c")
    session_id: str | None = Field(default=None)
    soft_cap: int = Field(default=10, ge=1)
    hard_cap: int = Field(default=50, ge=1)


class ResetConversationRequest(BaseModel):
    """Reset payload for demo-user conversation state/history."""

    user_code: str = Field(min_length=1, description="One of: user_a, user_b, user_c")
    session_id: str | None = Field(default=None)


class StartVoiceCallRequest(BaseModel):
    """Starts Pipecat voice runtime and returns client URL for call mode."""

    user_code: str = Field(min_length=1, description="One of: user_a, user_b, user_c")
    session_id: str | None = Field(default=None)
    port: int = Field(default=8788, ge=1024, le=65535)
    transport: str = Field(default="webrtc")
    soft_cap: int = Field(default=10, ge=1)
    hard_cap: int = Field(default=50, ge=1)


class StopVoiceCallRequest(BaseModel):
    """Stops active Pipecat voice runtime for the UI."""

    force: bool = Field(default=False)


@dataclass(slots=True)
class VoiceProcessState:
    """Tracks active Pipecat process metadata."""

    process: subprocess.Popen[Any]
    user_code: str
    session_id: str
    transport: str
    port: int
    client_url: str
    log_path: str
    started_at: float
    log_handle: Any


class CollectionVoiceProcessManager:
    """Starts/stops Pipecat bot process used by collection UI call mode."""

    def __init__(self, *, repo_root: Path, base_dir: Path) -> None:
        self._repo_root = repo_root
        self._base_dir = base_dir
        self._lock = threading.Lock()
        self._state: VoiceProcessState | None = None
        self._log_dir = self._base_dir / "runtime" / "voice"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def start(
        self,
        *,
        user_code: str,
        session_id: str,
        greeting: str,
        port: int = 8788,
        transport: str = "webrtc",
    ) -> dict[str, Any]:
        with self._lock:
            self._stop_locked(force=True)
            safe_session = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in session_id)
            log_path = self._log_dir / f"pipecat_{safe_session}.log"
            log_handle = log_path.open("a", encoding="utf-8")

            env = os.environ.copy()
            env["COLLECTION_VOICE_SESSION_ID"] = session_id
            env["COLLECTION_VOICE_USER_CODE"] = user_code
            env["COLLECTION_VOICE_GREETING"] = greeting
            cmd = [
                sys.executable,
                "-m",
                "agents.collection_agent.pipecat_bot",
                "-t",
                transport,
                "--port",
                str(port),
            ]
            process = subprocess.Popen(
                cmd,
                cwd=str(self._repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            client_url = f"http://127.0.0.1:{port}/client/"
            self._state = VoiceProcessState(
                process=process,
                user_code=user_code,
                session_id=session_id,
                transport=transport,
                port=port,
                client_url=client_url,
                log_path=str(log_path),
                started_at=time.time(),
                log_handle=log_handle,
            )
            return self._status_payload_locked()

    def stop(self, *, force: bool = False) -> dict[str, Any]:
        with self._lock:
            self._stop_locked(force=force)
            return self._status_payload_locked()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_payload_locked()

    def _stop_locked(self, *, force: bool) -> None:
        state = self._state
        if state is None:
            return
        process = state.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0 if force else 8.0)
            except Exception:
                process.kill()
                process.wait(timeout=3.0)
        try:
            state.log_handle.close()
        except Exception:
            pass
        self._state = None

    def _status_payload_locked(self) -> dict[str, Any]:
        state = self._state
        if state is None:
            return {
                "active": False,
                "session_id": None,
                "user_code": None,
                "transport": None,
                "port": None,
                "client_url": None,
                "log_path": None,
                "pid": None,
            }
        process = state.process
        running = process.poll() is None
        return {
            "active": running,
            "session_id": state.session_id,
            "user_code": state.user_code,
            "transport": state.transport,
            "port": state.port,
            "client_url": state.client_url,
            "log_path": state.log_path,
            "pid": process.pid,
            "started_at": state.started_at,
            "exit_code": process.poll(),
        }


@dataclass(slots=True)
class CollectionDebugRuntime:
    """Owns runtime agents and executes hop-by-hop internal routing."""

    base_dir: Path
    config_path: Path
    llm: Any | None
    llm_error: str | None
    collection_agent: CollectionAgent
    discount_agent: DiscountPlanningAgent
    memory_helper_agent: CollectionMemoryHelperAgent
    voice_manager: CollectionVoiceProcessManager
    demo_users: list[dict[str, Any]]

    @classmethod
    def create(cls, base_dir: Path) -> "CollectionDebugRuntime":
        config_path = base_dir / "config.yml"
        load_env_file(base_dir / ".env")
        raw_config = load_collection_config(config_path if config_path.exists() else DEFAULT_CONFIG_PATH)

        llm_error: str | None = None
        try:
            llm = build_llm(raw_config)
        except Exception as exc:  # pragma: no cover - fallback path for local demo use
            llm = None
            llm_error = str(exc)

        collection_repo = CollectionRepository(runtime_dir=base_dir / "runtime")
        collection_store = CollectionDataStore(base_dir=base_dir)
        memory_repo = CollectionMemoryRepository(collection_runtime_dir=base_dir / "runtime")
        demo_users = _load_demo_users(base_dir)

        collection_agent = CollectionAgent(
            repository=collection_repo,
            data_store=collection_store,
            llm=llm,
            trace_output_dir=base_dir / "runtime" / "traces",
        )
        discount_agent = DiscountPlanningAgent(llm=llm)
        memory_helper_agent = CollectionMemoryHelperAgent(repository=memory_repo, llm=llm)

        repo_root = Path(__file__).resolve().parents[3]
        voice_manager = CollectionVoiceProcessManager(repo_root=repo_root, base_dir=base_dir)

        return cls(
            base_dir=base_dir,
            config_path=config_path,
            llm=llm,
            llm_error=llm_error,
            collection_agent=collection_agent,
            discount_agent=discount_agent,
            memory_helper_agent=memory_helper_agent,
            voice_manager=voice_manager,
            demo_users=demo_users,
        )

    def list_demo_users(self) -> list[dict[str, Any]]:
        return _json_safe(self.demo_users)

    def start_conversation(self, request: StartConversationRequest) -> dict[str, Any]:
        user_code = request.user_code.strip().lower()
        matched = next((item for item in self.demo_users if item.get("user_code") == user_code), None)
        if matched is None:
            raise ValueError(f"Unknown user_code={user_code}. Use one of: user_a, user_b, user_c")

        requested_session = request.session_id.strip() if isinstance(request.session_id, str) else ""
        session_id = requested_session or f"collection-{user_code}"

        memory = self.collection_agent.session_store.load(session_id)
        memory.set_state(
            mode="strict_collections",
            active_channel="voice",
            active_case_id=str(matched["case"]["case_id"]),
            active_user_id=str(matched["customer"]["customer_id"]),
            active_customer_name=str(matched["customer"]["name"]),
            agent_loop_blocked=False,
            agent_loop_count=0,
        )

        opener_input = (
            "Start outbound collections call now. "
            f"Customer={matched['customer']['name']} "
            f"customer_id={matched['customer']['customer_id']} "
            f"case_id={matched['case']['case_id']} "
            f"overdue_amount={matched['case']['overdue_amount']} "
            "Generate the first call pitch by introducing yourself and requesting identity verification only. "
            "Do not disclose overdue amount, dues details, or payment options before verification."
        )

        turn = self.run_turn(
            RunTurnRequest(
                message=opener_input,
                session_id=session_id,
                soft_cap=request.soft_cap,
                hard_cap=request.hard_cap,
            )
        )
        return {
            "session_id": session_id,
            "user": _json_safe(matched),
            "opener_input": opener_input,
            "turn": turn,
        }

    def reset_conversation(self, request: ResetConversationRequest) -> dict[str, Any]:
        user_code = request.user_code.strip().lower()
        matched = next((item for item in self.demo_users if item.get("user_code") == user_code), None)
        if matched is None:
            raise ValueError(f"Unknown user_code={user_code}. Use one of: user_a, user_b, user_c")

        requested_session = request.session_id.strip() if isinstance(request.session_id, str) else ""
        session_id = requested_session or f"collection-{user_code}"
        self.collection_agent.repository.reset_conversation_session(session_id)
        return {
            "session_id": session_id,
            "user_code": user_code,
            "status": "reset",
        }

    def run_turn(
        self,
        request: RunTurnRequest,
        *,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        allow_deterministic_fallback = bool(getattr(self.collection_agent, "allow_deterministic_fallback", False))
        if self.llm is None and not allow_deterministic_fallback:
            error_text = (
                "LLM is not available for this session. "
                f"Startup error: {self.llm_error or 'unknown'}"
            )
            payload = {
                "session_id": request.session_id,
                "final_response": error_text,
                "final_target": "customer",
                "hops": [],
                "final_state": {
                    "error": {
                        "kind": "llm_unavailable",
                        "message": error_text,
                    }
                },
                "final_working_memory_state": self._memory_state(session_id=request.session_id),
                "llm": self._llm_meta(),
            }
            self._emit_progress(
                event_callback,
                {
                    "event": "turn_error",
                    "payload": {"error": error_text},
                },
            )
            return payload

        session_id = request.session_id.strip() or "collection-ui-session"
        soft_cap = max(1, int(request.soft_cap))
        hard_cap = max(soft_cap, int(request.hard_cap))
        timeout_seconds = max(1.0, float(request.timeout_seconds))

        current_input = request.message
        current_sender = str(request.sender).strip().lower() or "customer"
        hops: list[dict[str, Any]] = []
        started_at = time.monotonic()
        self._emit_progress(
            event_callback,
            {
                "event": "turn_started",
                "payload": {
                    "session_id": session_id,
                    "message": request.message,
                    "soft_cap": soft_cap,
                    "hard_cap": hard_cap,
                    "timeout_seconds": timeout_seconds,
                    "sender": current_sender,
                },
            },
        )

        for hop_index in range(1, hard_cap + 1):
            if (time.monotonic() - started_at) >= timeout_seconds:
                timeout_response = (
                    "I am still processing internal steps and do not want to keep you waiting. "
                    "Please choose one next step: pay now, request arrangement, or schedule follow-up."
                )
                self.collection_agent.session_store.load(session_id).set_state(
                    agent_loop_blocked=True,
                    agent_loop_count=hop_index,
                    agent_timeout_triggered=True,
                )
                return {
                    "session_id": session_id,
                    "final_response": timeout_response,
                    "final_target": "customer",
                    "hops": hops,
                    "final_state": {"timeout": {"seconds": timeout_seconds, "hop": hop_index}},
                    "final_working_memory_state": self._memory_state(session_id=session_id),
                    "llm": self._llm_meta(),
                }

            self._emit_progress(
                event_callback,
                {
                    "event": "hop_started",
                    "payload": {"hop": hop_index, "input": current_input, "sender": current_sender},
                },
            )
            trace_sink = InMemoryTraceSink(hop_index=hop_index, event_callback=event_callback)
            self.collection_agent.trace_sink = trace_sink
            state = self.collection_agent.run_turn(current_input, session_id=session_id, sender=current_sender)
            trace = self.collection_agent.last_trace

            response = str(state.get("response", "No response generated."))
            target = str(state.get("response_target", "customer")).strip().lower()
            memory_state = self._memory_state(session_id=session_id)

            hop_payload: dict[str, Any] = {
                "hop": hop_index,
                "input": current_input,
                "response": response,
                "response_target": target,
                "elapsed_ms": self._trace_elapsed_ms(trace),
                "node_history": _json_safe(state.get("node_history", [])),
                "conversation_phase": str(state.get("conversation_phase", "")),
                "thinking": self._thinking_lines(trace_sink.events),
                "events": trace_sink.events,
                "trace_summary": trace.summary() if trace is not None else {},
                "state": self._sanitize_state(state),
                "working_memory_state": memory_state,
            }
            hops.append(hop_payload)
            self._emit_progress(event_callback, {"event": "hop_update", "payload": hop_payload})

            if target == "customer":
                helper_result = self._run_memory_helper_if_requested(session_id=session_id, state=state)
                if helper_result is not None:
                    hop_payload["memory_helper"] = _json_safe(helper_result)
                return {
                    "session_id": session_id,
                    "final_response": response,
                    "final_target": "customer",
                    "hops": hops,
                    "final_state": self._sanitize_state(state),
                    "final_working_memory_state": self._memory_state(session_id=session_id),
                    "llm": self._llm_meta(),
                }

            if target == "discount_planning_agent":
                handoff_payload = state.get("handoff_payload") if isinstance(state.get("handoff_payload"), dict) else {}
                recommendation = self.discount_agent.run(handoff_payload)
                self.collection_agent.session_store.load(session_id).set_state(
                    discount_recommendation=recommendation,
                    last_tool_used="discount_planning_handoff",
                )
                hop_payload["discount_handoff"] = _json_safe(
                    {
                        "handoff_payload": handoff_payload,
                        "recommendation": recommendation,
                    }
                )
                current_input = "Discount recommendation ready. Continue and respond to customer."
                current_sender = "self"
                continue

            if hop_index >= soft_cap:
                self.collection_agent.session_store.load(session_id).set_state(
                    agent_loop_blocked=True,
                    agent_loop_count=hop_index,
                )
                guard_sink = InMemoryTraceSink(hop_index=hop_index + 1, event_callback=event_callback)
                self.collection_agent.trace_sink = guard_sink
                guard_state = self.collection_agent.run_turn("system loop guard", session_id=session_id, sender="self")
                guard_response = str(guard_state.get("response", response))
                helper_result = self._run_memory_helper_if_requested(session_id=session_id, state=guard_state)
                guard_payload = {
                    "hop": hop_index + 1,
                    "input": "system loop guard",
                    "response": guard_response,
                    "response_target": str(guard_state.get("response_target", "customer")),
                    "elapsed_ms": self._trace_elapsed_ms(self.collection_agent.last_trace),
                    "node_history": _json_safe(guard_state.get("node_history", [])),
                    "conversation_phase": str(guard_state.get("conversation_phase", "")),
                    "thinking": self._thinking_lines(guard_sink.events),
                    "events": guard_sink.events,
                    "trace_summary": self.collection_agent.last_trace.summary()
                    if self.collection_agent.last_trace is not None
                    else {},
                    "state": self._sanitize_state(guard_state),
                    "working_memory_state": self._memory_state(session_id=session_id),
                    "memory_helper": _json_safe(helper_result) if helper_result is not None else None,
                }
                hops.append(guard_payload)
                self._emit_progress(event_callback, {"event": "hop_update", "payload": guard_payload})
                return {
                    "session_id": session_id,
                    "final_response": guard_response,
                    "final_target": "customer",
                    "hops": hops,
                    "final_state": self._sanitize_state(guard_state),
                    "final_working_memory_state": self._memory_state(session_id=session_id),
                    "llm": self._llm_meta(),
                }

            if target == "self":
                current_input = response
                current_sender = "self"
                continue

            return {
                "session_id": session_id,
                "final_response": response,
                "final_target": target,
                "hops": hops,
                "final_state": self._sanitize_state(state),
                "final_working_memory_state": self._memory_state(session_id=session_id),
                "llm": self._llm_meta(),
            }

        # Should not be reached due to hard cap + returns above.
        return {
            "session_id": session_id,
            "final_response": "No terminal response generated.",
            "final_target": "customer",
            "hops": hops,
            "final_state": {},
            "final_working_memory_state": self._memory_state(session_id=session_id),
            "llm": self._llm_meta(),
        }

    def start_voice_call(self, request: StartVoiceCallRequest) -> dict[str, Any]:
        user_code = request.user_code.strip().lower()
        matched = next((item for item in self.demo_users if item.get("user_code") == user_code), None)
        if matched is None:
            raise ValueError(f"Unknown user_code={user_code}. Use one of: user_a, user_b, user_c")
        requested_session = request.session_id.strip() if isinstance(request.session_id, str) else ""
        session_id = requested_session or f"collection-{user_code}"

        start_payload = self.start_conversation(
            StartConversationRequest(
                user_code=user_code,
                session_id=session_id,
                soft_cap=request.soft_cap,
                hard_cap=request.hard_cap,
            )
        )
        opener = str(((start_payload.get("turn") or {}).get("final_response")) or "").strip()
        greeting = opener or "Hello. This is Alex from the collections team. How can I help you today?"
        voice = self.voice_manager.start(
            user_code=user_code,
            session_id=session_id,
            greeting=greeting,
            port=int(request.port),
            transport=str(request.transport).strip().lower() or "webrtc",
        )
        return {
            "session_id": session_id,
            "user": _json_safe(matched),
            "turn": start_payload.get("turn"),
            "voice": voice,
        }

    def stop_voice_call(self, *, force: bool = False) -> dict[str, Any]:
        return self.voice_manager.stop(force=force)

    def voice_status(self) -> dict[str, Any]:
        return self.voice_manager.status()

    def conversation_messages(self, session_id: str, *, limit: int = 120) -> list[dict[str, Any]]:
        sid = session_id.strip() or "collection-ui-session"
        safe_limit = max(1, int(limit))
        messages = self.collection_agent.repository.list_conversation_messages(sid)
        window = messages[-safe_limit:]
        return [_json_safe(message.model_dump(mode="json")) for message in window]

    def latest_trace_for_session(self, session_id: str) -> dict[str, Any] | None:
        sid = session_id.strip()
        if not sid:
            return None
        trace_dir = self.base_dir / "runtime" / "traces"
        if not trace_dir.exists():
            return None

        candidates = sorted(trace_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates[:300]:
            if path.name == "latest_trace.json":
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("session_id", "")).strip() != sid:
                continue
            return _json_safe(payload)
        return None

    def run_turn_stream(self, request: RunTurnRequest) -> StreamingResponse:
        sentinel = object()
        events: queue.Queue[object] = queue.Queue()

        def publish(payload: dict[str, Any]) -> None:
            events.put(_json_safe(payload))

        def worker() -> None:
            try:
                result = self.run_turn(request, event_callback=publish)
                publish({"event": "turn_complete", "payload": result})
            except Exception as exc:  # pragma: no cover - runtime stream error path
                publish({"event": "turn_error", "payload": {"error": str(exc)}})
            finally:
                events.put(sentinel)

        threading.Thread(target=worker, daemon=True).start()

        def stream() -> Any:
            yield self._sse("stream_open", {"session_id": request.session_id})
            while True:
                item = events.get()
                if item is sentinel:
                    break
                if not isinstance(item, dict):
                    continue
                event_name = str(item.get("event", "message"))
                payload = item.get("payload", {})
                yield self._sse(event_name, payload)
            yield self._sse("stream_close", {"session_id": request.session_id})

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def session_state(self, session_id: str) -> SessionStateResponse:
        sid = session_id.strip() or "collection-ui-session"
        state = self.collection_agent.repository.get_conversation_state(sid) or {}
        return SessionStateResponse(
            session_id=sid,
            conversation_state=_json_safe(state),
            working_memory_state=self._memory_state(session_id=sid),
        )

    def _run_memory_helper_if_requested(self, *, session_id: str, state: dict[str, Any]) -> dict[str, Any] | None:
        targets = state.get("additional_targets")
        if not isinstance(targets, list):
            return None

        lowered_targets = {str(item).strip().lower() for item in targets}
        if "collection_memory_helper_agent" not in lowered_targets:
            return None

        trigger = state.get("memory_helper_trigger") if isinstance(state.get("memory_helper_trigger"), dict) else {}
        payload = self._build_memory_helper_payload(session_id=session_id, trigger=trigger)
        return self.memory_helper_agent.run(payload)

    def _build_memory_helper_payload(self, *, session_id: str, trigger: dict[str, Any]) -> dict[str, Any]:
        repository = self.collection_agent.repository
        messages = [message.model_dump(mode="json") for message in repository.list_conversation_messages(session_id)]
        conversation_state = repository.get_conversation_state(session_id) or {}

        user_id = str(conversation_state.get("active_user_id", "")).strip()
        if not user_id:
            case_id = str(conversation_state.get("active_case_id", "")).strip()
            if case_id:
                case_row = self.collection_agent.data_store.get_case(case_id=case_id)
                if isinstance(case_row, dict) and case_row.get("customer_id"):
                    user_id = str(case_row["customer_id"])

        return {
            "session_id": session_id,
            "user_id": user_id or None,
            "trigger": trigger,
            "conversation_messages": messages,
            "conversation_state": conversation_state,
        }

    def _sanitize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in state.items():
            if key == "memory":
                continue
            payload[str(key)] = _json_safe(value)
        return payload

    def _memory_state(self, *, session_id: str) -> dict[str, Any]:
        return _json_safe(dict(self.collection_agent.session_store.load(session_id).state))

    def _thinking_lines(self, events: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for event in events:
            kind = str(event.get("event", "")).strip()
            if kind == "node_started":
                lines.append(f"node start: {event.get('node_name')}")
            elif kind == "node_finished":
                status = str(event.get("status", "completed"))
                lines.append(f"node finish: {event.get('node_name')} [{status}]")
            elif kind == "tool_call":
                status = str(event.get("status", ""))
                lines.append(f"tool: {event.get('tool_name')} [{status}]")
            elif kind == "llm_call":
                model_name = str(event.get("model_name", ""))
                tokens = event.get("total_tokens")
                token_text = f", tokens={tokens}" if tokens is not None else ""
                lines.append(f"llm: {model_name}{token_text}")
            elif kind == "node_state":
                route = event.get("route")
                if route is not None:
                    lines.append(f"route selected: {route}")
                human_message = str(event.get("human_message", "")).strip()
                if human_message:
                    lines.append(f"node update: {human_message}")
        return lines

    def _llm_meta(self) -> dict[str, Any]:
        provider = None
        model_name = None
        if self.llm is not None:
            provider = str(self.llm.__class__.__name__)
            model_name = getattr(self.llm, "model_name", None)
        return {
            "provider": provider,
            "model_name": model_name,
            "startup_error": self.llm_error,
        }

    @staticmethod
    def _trace_elapsed_ms(trace: Any | None) -> float | None:
        if trace is None:
            return None
        started = getattr(trace, "started_at", None)
        finished = getattr(trace, "finished_at", None)
        if not isinstance(started, datetime) or not isinstance(finished, datetime):
            return None
        return round((finished - started).total_seconds() * 1000, 3)

    @staticmethod
    def _emit_progress(
        callback: Callable[[dict[str, Any]], None] | None,
        payload: dict[str, Any],
    ) -> None:
        if callable(callback):
            callback(payload)

    @staticmethod
    def _sse(event_name: str, payload: dict[str, Any]) -> str:
        return f"event: {event_name}\ndata: {json.dumps(_json_safe(payload), ensure_ascii=True)}\n\n"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "_asdict"):
        return _json_safe(value._asdict())
    if hasattr(value, "__dict__"):
        return {key: _json_safe(item) for key, item in vars(value).items() if not key.startswith("_")}
    return repr(value)


def _load_demo_users(base_dir: Path) -> list[dict[str, Any]]:
    customers_path = base_dir / "data" / "customers.json"
    cases_path = base_dir / "data" / "cases.json"
    customers_payload = json.loads(customers_path.read_text(encoding="utf-8"))
    cases_payload = json.loads(cases_path.read_text(encoding="utf-8"))

    customers = [dict(row) for row in customers_payload if isinstance(row, dict)]
    cases = [dict(row) for row in cases_payload if isinstance(row, dict)]
    case_by_customer = {str(case.get("customer_id")): case for case in cases}

    labels = ["user_a", "user_b", "user_c"]
    result: list[dict[str, Any]] = []
    for index, customer in enumerate(customers[:3]):
        user_code = labels[index]
        challenge = customer.get("challenge", {}) if isinstance(customer.get("challenge"), dict) else {}
        case_row = case_by_customer.get(str(customer.get("customer_id", "")), {})
        result.append(
            {
                "user_code": user_code,
                "display_name": f"User {chr(ord('A') + index)}",
                "customer": {
                    "customer_id": str(customer.get("customer_id", "")),
                    "name": str(customer.get("name", "")),
                    "phone": str(customer.get("phone", "")),
                    "email": str(customer.get("email", "")),
                    "dob": str(challenge.get("dob", "")),
                    "zip": str(challenge.get("zip", "")),
                    "last4_pan": str(challenge.get("last4_pan", "")),
                },
                "case": {
                    "case_id": str(case_row.get("case_id", "")),
                    "loan_id": str(case_row.get("loan_id", "")),
                    "product": str(case_row.get("product", "")),
                    "dpd": case_row.get("dpd"),
                    "emi_amount": case_row.get("emi_amount"),
                    "overdue_amount": case_row.get("overdue_amount"),
                    "late_fee": case_row.get("late_fee"),
                    "risk_band": str(case_row.get("risk_band", "")),
                },
            }
        )
    return result


def create_router(runtime: CollectionDebugRuntime) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["collection_agent_ui"])

    @router.post("/run-turn")
    async def run_turn(body: RunTurnRequest) -> dict[str, Any]:
        return runtime.run_turn(body)

    @router.get("/run-turn-stream")
    async def run_turn_stream(
        message: str,
        session_id: str = "collection-ui-session",
        soft_cap: int = 10,
        hard_cap: int = 50,
        timeout_seconds: float = 20.0,
        sender: str = "customer",
    ) -> StreamingResponse:
        request = RunTurnRequest(
            message=message,
            session_id=session_id,
            soft_cap=soft_cap,
            hard_cap=hard_cap,
            timeout_seconds=timeout_seconds,
            sender=sender,
        )
        return runtime.run_turn_stream(request)

    @router.get("/session/{session_id}")
    async def session_state(session_id: str) -> SessionStateResponse:
        return runtime.session_state(session_id)

    @router.get("/demo-users")
    async def demo_users() -> dict[str, Any]:
        return {"users": runtime.list_demo_users()}

    @router.post("/start-conversation")
    async def start_conversation(body: StartConversationRequest) -> dict[str, Any]:
        try:
            return runtime.start_conversation(body)
        except Exception as exc:
            return {"error": str(exc), "users": runtime.list_demo_users()}

    @router.post("/reset-conversation")
    async def reset_conversation(body: ResetConversationRequest) -> dict[str, Any]:
        try:
            return runtime.reset_conversation(body)
        except Exception as exc:
            return {"error": str(exc), "users": runtime.list_demo_users()}

    @router.post("/start-call")
    async def start_call(body: StartVoiceCallRequest) -> dict[str, Any]:
        try:
            return runtime.start_voice_call(body)
        except Exception as exc:
            return {"error": str(exc), "users": runtime.list_demo_users()}

    @router.get("/voice/status")
    async def voice_status() -> dict[str, Any]:
        return runtime.voice_status()

    @router.post("/voice/stop")
    async def stop_voice(body: StopVoiceCallRequest) -> dict[str, Any]:
        return runtime.stop_voice_call(force=bool(body.force))

    @router.get("/session/{session_id}/messages")
    async def session_messages(session_id: str, limit: int = 120) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "messages": runtime.conversation_messages(session_id=session_id, limit=limit),
        }

    @router.get("/session/{session_id}/latest-trace")
    async def session_latest_trace(session_id: str) -> dict[str, Any]:
        return {
            "session_id": session_id,
            "trace": runtime.latest_trace_for_session(session_id=session_id),
        }

    return router


def create_app(base_dir: Path | None = None) -> FastAPI:
    resolved_base_dir = (base_dir or Path(__file__).resolve().parents[1]).resolve()
    runtime = CollectionDebugRuntime.create(resolved_base_dir)

    app = FastAPI(title="Collection Agent Debug UI")
    app.include_router(create_router(runtime))
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="collection-agent-ui-static")

    @app.get("/")
    async def home() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.on_event("shutdown")
    async def _shutdown_voice_runtime() -> None:
        runtime.stop_voice_call(force=True)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8060)
