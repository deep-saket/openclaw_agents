"""Shared ReAct-style decision node with hook-based lifecycle."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


@dataclass(slots=True)
class ReactNode(BaseGraphNode):
    """Decides act/respond/end with explicit pre/post override hooks.

    Lifecycle:
    1) build context
    2) pre-LLM override (optional short-circuit)
    3) LLM or fallback decision
    4) post-LLM override
    """

    llm: Any | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    available_tools: Any | None = None
    max_steps: int = 4
    allow_deterministic_fallback: bool = True
    last_plan_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def execute(self, state: AgentState) -> NodeUpdate:
        self._record_llm_usage(state, node_name="react")
        before_updates = self._update_node_owned_keys_before(state=state)
        emitted_before_updates: dict[str, Any] = {}
        if isinstance(before_updates, dict) and before_updates:
            state.update(before_updates)
            emitted_before_updates = dict(before_updates)
        context = self._build_context_for_react(state)
        pre_rule = self._apply_pre_llm_override(state=state, context=context)
        if pre_rule is None:
            pre_rule = {}
        if not isinstance(pre_rule, dict):
            raise TypeError("_apply_pre_llm_override must return dict or None.")
        if pre_rule.get("context_updates") and isinstance(pre_rule.get("context_updates"), dict):
            context = {**context, **dict(pre_rule["context_updates"])}

        if pre_rule.get("skip_llm", False):
            decision = pre_rule.get("decision")
            if decision is None:
                decision = self._fallback_decision(
                    state=state,
                    reason=str(pre_rule.get("reason", "pre_rule_skip_without_decision")),
                )
            self.last_plan_debug = {
                "prompt": None,
                "system_prompt": None,
                "llm_response": None,
                "llm_error": None,
            }
        else:
            decision = self._plan_with_llm_or_fallback(state=state, context=context)

        decision = self._apply_post_llm_override(state=state, context=context, decision=decision)
        tool_calls = self._decision_tool_calls(decision)
        pending_tool_calls = self._pending_from_decision(decision)
        if isinstance(pre_rule.get("pending_tool_calls"), list):
            pending_tool_calls = list(pre_rule.get("pending_tool_calls", []))
        llm_status = "used_llm" if self.last_plan_debug.get("llm_response") is not None else "fallback"
        update: NodeUpdate = {
            "decision": decision,
            "steps": state.get("steps", 0) + 1,
            "prompt": self.last_plan_debug.get("prompt"),
            "system_prompt": self.last_plan_debug.get("system_prompt"),
            "llm_response": self.last_plan_debug.get("llm_response"),
            "llm_error": self.last_plan_debug.get("llm_error"),
            "llm_status": llm_status,
            "tool_calls": tool_calls,
        }
        if emitted_before_updates:
            update.update(emitted_before_updates)
        update["pending_tool_calls"] = pending_tool_calls or []
        update["no_tools_required"] = not self._decision_requires_tool(decision)
        after_updates = self._update_node_owned_keys_after(state=state, update=update)
        if isinstance(after_updates, dict) and after_updates:
            state.update(after_updates)
            update.update(after_updates)
        return update

    def route(self, state: AgentState) -> str:
        if state.get("steps", 0) > self.max_steps:
            state["response"] = "I reached the tool limit for this turn. Please narrow the request."
            return "respond"
        decision = state.get("decision")
        if decision is None:
            state["response"] = "I need a bit more detail to continue."
            return "respond"
        if getattr(decision, "respond_directly", False) or getattr(decision, "done", False):
            return "respond"
        tool_call = getattr(decision, "tool_call", None)
        if tool_call is not None:
            return "act"
        tool_calls = getattr(decision, "tool_calls", None)
        if isinstance(tool_calls, list) and tool_calls:
            first = tool_calls[0]
            if isinstance(first, dict) and str(first.get("tool_name", "")).strip():
                # Current tool node executes one call at a time. Promote first call.
                decision.tool_call = SimpleNamespace(
                    tool_name=str(first.get("tool_name", "")).strip(),
                    arguments=first.get("arguments", {}) if isinstance(first.get("arguments"), dict) else {},
                )
                return "act"
        state["response"] = "I need a bit more detail to continue."
        return "respond"

    def _build_context_for_react(self, state: AgentState) -> dict[str, Any]:
        memory = state.get("memory")
        raw_memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        memory_state = self._compact_memory_state(raw_memory_state)
        observations = list(state.get("observations", [])) if isinstance(state.get("observations"), list) else []
        if not observations and isinstance(state.get("observation"), dict):
            observations = [dict(state.get("observation", {}))]
        latest_observation = None
        for item in reversed(observations):
            if isinstance(item, dict):
                latest_observation = item
                break
        return {
            "session_id": state.get("session_id"),
            "user_input": state.get("user_input"),
            "observation": latest_observation,
            "observations": observations,
            "memory_context": state.get("memory_context"),
            "available_tools": self.available_tools if self.available_tools is not None else state.get("available_tools"),
            "memory_state": memory_state,
            "intent": state.get("intent"),
            "route": state.get("route"),
            "steps": state.get("steps", 0),
        }

    @staticmethod
    def _compact_memory_state(memory_state: dict[str, Any]) -> dict[str, Any]:
        if not memory_state:
            return {}
        keep_keys = {
            "mode",
            "active_case_id",
            "active_user_id",
            "active_channel",
            "identity_verified",
            "verification_missing_fields",
            "verification_verified_fields",
            "last_tool_observation",
            "last_response_target",
            "turn_index",
        }
        compact: dict[str, Any] = {}
        for key in keep_keys:
            if key in memory_state:
                compact[key] = memory_state.get(key)
        if isinstance(memory_state.get("tool_observations_history"), list):
            compact["tool_observations_history"] = memory_state.get("tool_observations_history", [])[-8:]
        return compact

    def _apply_pre_llm_override(self, *, state: AgentState, context: dict[str, Any]) -> dict[str, Any] | None:
        queued = state.get("pending_tool_calls")
        if isinstance(queued, list) and queued:
            first = queued[0] if isinstance(queued[0], dict) else {}
            tool_name = str(first.get("tool_name", "")).strip()
            if tool_name:
                arguments = first.get("arguments") if isinstance(first.get("arguments"), dict) else {}
                return {
                    "skip_llm": True,
                    "reason": "queued_tool_chain",
                    "decision": SimpleNamespace(
                        thought=f"Continue queued tool chain with `{tool_name}`.",
                        tool_call=SimpleNamespace(tool_name=tool_name, arguments=arguments),
                        respond_directly=False,
                        response_text=None,
                        done=False,
                    ),
                    "pending_tool_calls": queued[1:],
                }
        del state, context
        return None

    def _apply_post_llm_override(self, *, state: AgentState, context: dict[str, Any], decision: Any) -> Any:
        del state, context
        return decision

    def _update_node_owned_keys_before(self, *, state: AgentState) -> dict[str, Any] | None:
        del state
        return None

    def _update_node_owned_keys_after(self, *, state: AgentState, update: NodeUpdate) -> dict[str, Any] | None:
        del state, update
        return None

    def _plan_with_llm_or_fallback(self, *, state: AgentState, context: dict[str, Any]) -> Any:
        rendered_system_prompt = self.system_prompt or ""
        rendered_user_prompt = self._render_user_prompt(
            user_prompt=self.user_prompt or "{user_input}",
            context=context,
        )
        self.last_plan_debug = {
            "prompt": rendered_user_prompt,
            "system_prompt": rendered_system_prompt or None,
            "llm_response": None,
            "llm_error": None,
        }
        if self.llm is None:
            return self._fallback_decision(state=state, reason="llm_not_configured")
        try:
            raw = self.llm.generate(rendered_system_prompt, rendered_user_prompt).strip()
            self.last_plan_debug["llm_response"] = raw
            return self._parse_decision(raw)
        except Exception as exc:
            self.last_plan_debug["llm_error"] = str(exc)
            return self._fallback_decision(state=state, reason="llm_error")

    def _fallback_decision(self, *, state: AgentState, reason: str) -> Any:
        del reason
        if not self.allow_deterministic_fallback:
            raise RuntimeError("ReactNode failed to produce decision and fallback is disabled.")
        prior = str(state.get("response", "")).strip()
        text = prior or "I need a bit more detail to continue."
        return SimpleNamespace(
            thought="ReactNode fallback direct response.",
            tool_call=None,
            respond_directly=True,
            response_text=text,
            done=True,
        )

    def _render_user_prompt(self, *, user_prompt: str, context: dict[str, Any]) -> str:
        values = {
            "user_input": self._stringify_context(context.get("user_input")),
            "recent_conversation": self._stringify_context(context.get("recent_conversation")),
            "observation": self._stringify_context(context.get("observation")),
            "observations": self._stringify_context(context.get("observations")),
            "memory_context": self._stringify_context(context.get("memory_context")),
            "memory_contents": self._stringify_context(context.get("memory_contents")),
            "available_tools": self._stringify_context(context.get("available_tools")),
        }
        rendered_lines: list[str] = []
        for line in user_prompt.splitlines():
            rendered_line = line
            skip_line = False
            for key, value in values.items():
                placeholder = f"{{{key}}}"
                if placeholder not in rendered_line:
                    continue
                if value is None:
                    skip_line = True
                    break
                rendered_line = rendered_line.replace(placeholder, value)
            if skip_line:
                continue
            if re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", rendered_line):
                continue
            rendered_lines.append(rendered_line)
        rendered = "\n".join(rendered_lines).strip() or user_prompt
        context_block = self._build_context_block(context)
        if context_block:
            rendered = f"{rendered}\n\n{context_block}"
        return rendered

    @staticmethod
    def _build_context_block(context: dict[str, Any]) -> str:
        lines: list[str] = []
        for key in sorted(context.keys()):
            if key in {"available_tools", "memory_state"}:
                continue
            value = context.get(key)
            if value is None:
                continue
            text = ReactNode._stringify_context(value)
            if text is None:
                continue
            compact = text.strip()
            if not compact:
                continue
            lines.append(f"- {key}: {compact}")
        memory_state = context.get("memory_state")
        if isinstance(memory_state, dict) and memory_state:
            lines.append(f"- memory_state: {ReactNode._stringify_context(memory_state)}")
        if not lines:
            return ""
        return "React context:\n" + "\n".join(lines)

    @staticmethod
    def _stringify_context(value: Any | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str, ensure_ascii=True)
        except TypeError:
            return str(value)

    @staticmethod
    def _parse_decision(raw: str) -> Any:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return SimpleNamespace(
                thought="No JSON returned; fallback to direct response.",
                tool_call=None,
                respond_directly=True,
                response_text=raw,
                done=True,
            )
        payload = json.loads(match.group(0))
        tool_name = payload.get("tool_name")
        tool_calls = payload.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized_calls = []
            for item in tool_calls:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("tool_name", "")).strip()
                if not name:
                    continue
                args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
                normalized_calls.append({"tool_name": name, "arguments": args})
            if normalized_calls:
                first = normalized_calls[0]
                return SimpleNamespace(
                    thought=payload.get("thought", f"Use {first['tool_name']}."),
                    tool_call=SimpleNamespace(tool_name=first["tool_name"], arguments=first["arguments"]),
                    tool_calls=normalized_calls,
                    respond_directly=False,
                    response_text=None,
                    done=False,
                    no_tools_required=bool(payload.get("no_tools_required", False)),
                )
        if tool_name is None:
            return SimpleNamespace(
                thought=payload.get("thought", "Respond directly."),
                tool_call=None,
                respond_directly=bool(payload.get("respond_directly", True)),
                response_text=payload.get("response_text", raw),
                done=bool(payload.get("done", True)),
                no_tools_required=bool(payload.get("no_tools_required", False)),
            )
        return SimpleNamespace(
            thought=payload.get("thought", f"Use {tool_name}."),
            tool_call=SimpleNamespace(
                tool_name=tool_name,
                arguments=payload.get("arguments", {}),
            ),
            respond_directly=False,
            response_text=None,
            done=False,
            no_tools_required=bool(payload.get("no_tools_required", False)),
        )

    @staticmethod
    def _decision_tool_calls(decision: Any) -> list[dict[str, Any]] | None:
        tool_call = getattr(decision, "tool_call", None)
        if tool_call is None:
            tool_calls = getattr(decision, "tool_calls", None)
            if isinstance(tool_calls, list):
                out: list[dict[str, Any]] = []
                for item in tool_calls:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("tool_name", "")).strip()
                    if not name:
                        continue
                    args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
                    out.append({"tool_name": name, "arguments": args})
                return out or None
            return None
        return [
            {
                "tool_name": str(getattr(tool_call, "tool_name", "")),
                "arguments": getattr(tool_call, "arguments", {}) if isinstance(getattr(tool_call, "arguments", {}), dict) else {},
            }
        ]

    @staticmethod
    def _pending_from_decision(decision: Any) -> list[dict[str, Any]] | None:
        tool_calls = getattr(decision, "tool_calls", None)
        if not isinstance(tool_calls, list) or len(tool_calls) <= 1:
            return None
        pending: list[dict[str, Any]] = []
        for item in tool_calls[1:]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("tool_name", "")).strip()
            if not name:
                continue
            args = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
            pending.append({"tool_name": name, "arguments": args})
        return pending or None

    @staticmethod
    def _decision_requires_tool(decision: Any) -> bool:
        tool_call = getattr(decision, "tool_call", None)
        if tool_call is not None and str(getattr(tool_call, "tool_name", "")).strip():
            return True
        tool_calls = getattr(decision, "tool_calls", None)
        if isinstance(tool_calls, list):
            for item in tool_calls:
                if not isinstance(item, dict):
                    continue
                if str(item.get("tool_name", "")).strip():
                    return True
        return False
