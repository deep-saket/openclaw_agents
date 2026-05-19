"""Prompt loaders for Collection Agent demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_collection_agent_prompts() -> dict[str, Any]:
    path = Path(__file__).with_name("agent_prompts.yml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_collection_tool_catalog() -> dict[str, Any]:
    path = Path(__file__).with_name("tool_catalog.yml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def render_collection_tool_catalog_yaml(
    *,
    include_tool_names: list[str] | None = None,
    exclude_tool_names: list[str] | None = None,
) -> str:
    catalog = load_collection_tool_catalog()
    tools = catalog.get("tools")
    if not isinstance(tools, list):
        return yaml.safe_dump(catalog, sort_keys=False, allow_unicode=False).strip()

    include = {str(name).strip() for name in include_tool_names or [] if str(name).strip()}
    exclude = {str(name).strip() for name in exclude_tool_names or [] if str(name).strip()}

    filtered_tools: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_name = str(tool.get("name", "")).strip()
        if include and tool_name not in include:
            continue
        if exclude and tool_name in exclude:
            continue
        filtered_tools.append(tool)

    filtered_catalog = {**catalog, "tools": filtered_tools}
    return yaml.safe_dump(filtered_catalog, sort_keys=False, allow_unicode=False).strip()


__all__ = [
    "load_collection_agent_prompts",
    "load_collection_tool_catalog",
    "render_collection_tool_catalog_yaml",
]
