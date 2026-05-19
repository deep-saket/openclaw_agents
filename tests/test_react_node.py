from __future__ import annotations

from src.nodes.react_node import ReactNode


class _PreHookReactNode(ReactNode):
    def _update_node_owned_keys_before(self, *, state):  # type: ignore[override]
        del state
        return {
            "identity_verified": True,
            "verification_verified_fields": ["dob", "phone"],
        }


def test_react_node_emits_pre_hook_updates_in_node_update() -> None:
    node = _PreHookReactNode(llm=None)

    update = node.execute(
        {
            "user_input": "hello",
            "response": "fallback",
            "steps": 0,
        }
    )

    assert update["identity_verified"] is True
    assert update["verification_verified_fields"] == ["dob", "phone"]
