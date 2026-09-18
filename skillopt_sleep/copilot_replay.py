"""Validation for opt-in, local Copilot replay profiles (no server startup)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

EXACT_TOOL_NAME = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"


class CopilotReplayError(RuntimeError):
    """A replay profile cannot be used without weakening its execution boundary."""


def validate_replay_tools(tools: object) -> tuple[str, ...]:
    if not isinstance(tools, list) or not tools:
        raise CopilotReplayError(
            "copilot_replay_tools must be a non-empty array of exact Copilot tool names "
            "(CLI: repeat --copilot-replay-tool NAME)."
        )
    if any(not isinstance(tool, str) or not re.fullmatch(EXACT_TOOL_NAME, tool) for tool in tools):
        raise CopilotReplayError(
            "copilot_replay_tools entries must be exact tool names, not wildcards, "
            "permission selectors, whitespace, or comma-separated lists."
        )
    if len(set(tools)) != len(tools):
        raise CopilotReplayError("copilot_replay_tools entries must be unique.")
    return tuple(tools)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not any(ord(c) < 32 for c in value)


@dataclass(frozen=True)
class CopilotReplayProfile:
    directory: str
    tools: tuple[str, ...]

    def validate(self) -> None:
        if os.environ.get("SKILLOPT_SLEEP_COPILOT_FULL_ENV", "") not in {"", "0"}:
            raise CopilotReplayError(
                "SKILLOPT_SLEEP_COPILOT_FULL_ENV must be unset or 0 with copilot_replay_profile; "
                "replay profiles never use the full personal environment."
            )
        if not os.path.isdir(self.directory):
            raise CopilotReplayError("copilot_replay_profile must name an existing local directory.")
        path = os.path.join(self.directory, "mcp-config.json")
        try:
            with open(path, encoding="utf-8-sig") as handle:
                config = json.load(handle, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except OSError as exc:
            raise CopilotReplayError(
                "copilot_replay_profile must contain a readable mcp-config.json; "
                "no personal-home fallback is allowed."
            ) from exc
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise CopilotReplayError(
                "Replay mcp-config.json must be valid UTF-8 JSON without duplicate keys or non-finite numbers."
            ) from exc
        servers = config.get("mcpServers") if isinstance(config, dict) else None
        if not isinstance(servers, dict) or not servers:
            raise CopilotReplayError("Replay mcp-config.json must contain a non-empty mcpServers object.")
        for name, server in servers.items():
            if not _nonempty_text(name) or not isinstance(server, dict):
                raise CopilotReplayError("Replay mcpServers entries must have non-empty names and object definitions.")
            transport = server.get("type", "http" if "url" in server else "local")
            if not isinstance(transport, str):
                raise CopilotReplayError("Replay MCP server type must be a string.")
            if transport in {"local", "stdio"}:
                if not _nonempty_text(server.get("command")) or "url" in server:
                    raise CopilotReplayError("A local replay MCP server needs a command, not a URL.")
            elif transport in {"http", "sse"}:
                if not _nonempty_text(server.get("url")) or "command" in server:
                    raise CopilotReplayError("A remote replay MCP server needs a URL, not a command.")
            else:
                raise CopilotReplayError("Replay MCP server type must be local, stdio, http, or sse.")
            for key in ("args", "tools"):
                if key in server and (
                    not isinstance(server[key], list)
                    or any(not isinstance(item, str) for item in server[key])
                ):
                    raise CopilotReplayError(f"Replay MCP server {key} must be an array of strings.")
            for key in ("env", "headers"):
                if key in server and (
                    not isinstance(server[key], dict)
                    or any(not isinstance(value, str) for value in server[key].values())
                ):
                    raise CopilotReplayError(f"Replay MCP server {key} must be an object of strings.")


def resolve_replay_profile(
    profile: object = "",
    tools: object = None,
    *,
    project_dir: str = "",
) -> CopilotReplayProfile | None:
    if profile in ("", None) and (tools is None or tools == []):
        return None
    if not isinstance(profile, str) or not _nonempty_text(profile):
        raise CopilotReplayError(
            "copilot_replay_tools requires a non-empty copilot_replay_profile directory "
            "(CLI: --copilot-replay-profile PATH)."
        )
    names = validate_replay_tools(tools)
    directory = os.path.expanduser(profile)
    if not os.path.isabs(directory):
        directory = os.path.join(project_dir or os.getcwd(), directory)
    resolved = CopilotReplayProfile(os.path.abspath(directory), names)
    resolved.validate()
    return resolved
