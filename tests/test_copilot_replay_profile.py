"""Offline replay-profile contract: no Copilot model or MCP server is started."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from skillopt_sleep import __main__ as cli
from skillopt_sleep import backend as backend_mod
from skillopt_sleep import cycle
from skillopt_sleep.backend import CopilotCliBackend, DualBackend, MockBackend, build_backend, get_backend
from skillopt_sleep.config import DEFAULTS, SleepConfig, load_config
from skillopt_sleep.copilot_replay import CopilotReplayError, resolve_replay_profile
from skillopt_sleep.mine import mine
from skillopt_sleep.multi_skill import SkillGroup, consolidate_groups
from skillopt_sleep.replay import replay_one
from skillopt_sleep.staging import latest_staging
from skillopt_sleep.types import ReplayResult, TaskRecord

TOOLS = ["catalog-get_item", "catalog-list_items"]
MCP_CONFIG = {
    "mcpServers": {
        "catalog": {
            "command": "example-catalog-mcp",
            "args": ["--read-only"],
            "tools": ["get_item", "list_items"],
        },
    },
}


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch, tmp_path):
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: None)
    monkeypatch.delenv("SKILLOPT_SLEEP_COPILOT_FULL_ENV", raising=False)
    monkeypatch.delenv("COPILOT_AVAILABLE_TOOLS", raising=False)
    monkeypatch.setenv("SKILLOPT_SLEEP_COPILOT_HOME", str(tmp_path / "legacy-home"))
    monkeypatch.setenv("COPILOT_HOME", str(tmp_path / "personal-home"))
    monkeypatch.setattr(backend_mod, "resolve_copilot_path", lambda _: "unused-copilot")
    monkeypatch.setattr(backend_mod, "resolve_copilot_argv", lambda _: ["unused-copilot"])

    def no_spawn(*args, **kwargs):
        pytest.fail("offline test unexpectedly tried to start a subprocess")

    monkeypatch.setattr(backend_mod.subprocess, "run", no_spawn)


@pytest.fixture
def profile(tmp_path):
    directory = tmp_path / "replay profile"
    directory.mkdir()
    (directory / "mcp-config.json").write_text(json.dumps(MCP_CONFIG), encoding="utf-8")
    return directory


def _backend(profile):
    return CopilotCliBackend(
        copilot_replay_profile=str(profile), copilot_replay_tools=TOOLS,
    )


def _task(**kwargs):
    return TaskRecord(id="task", project="example", intent="Look up an item.", **kwargs)


def _stream(text="answer", events=()):
    return "\n".join(
        [json.dumps(event) for event in events]
        + [json.dumps({"type": "assistant.message", "data": {"content": text}})]
    )


def _capture(monkeypatch, *, text="answer", events=(), returncode=0):
    calls = []
    responses = iter(text) if isinstance(text, list) else None

    def run(cmd, **kwargs):
        cwd = Path(kwargs["cwd"])
        home = Path(kwargs["env"]["COPILOT_HOME"])
        calls.append({
            "cmd": cmd, **kwargs, "files": list(cwd.iterdir()),
            "home_has_mcp": (home / "mcp-config.json").is_file(),
        })
        response = next(responses) if responses is not None else text
        return subprocess.CompletedProcess(cmd, returncode, _stream(response, events), "private diagnostic")

    monkeypatch.setattr(backend_mod.subprocess, "run", run)
    return calls


def _scope(call):
    start = call["cmd"].index("--available-tools") + 1
    return ",".join(call["cmd"][start:call["cmd"].index("-C")])


def _parser():
    parser = argparse.ArgumentParser()
    cli._add_common(parser)
    cli._add_copilot_replay(parser)
    return parser


def _config(tmp_path, profile, **overrides):
    return SleepConfig(data={
        **DEFAULTS, "backend": "copilot",
        "projects": "invoked", "invoked_project": str(tmp_path),
        "claude_home": str(tmp_path / "claude"), "state_dir": str(tmp_path / "state"),
        "copilot_replay_profile": str(profile), "copilot_replay_tools": list(TOOLS),
        **overrides,
    })


def test_defaults_do_not_select_profile_or_enable_adoption():
    cfg = load_config()
    assert cfg.copilot_replay_profile == ""
    assert cfg.copilot_replay_tools == []
    assert cfg.auto_adopt is False
    assert resolve_replay_profile("", []) is None
    assert get_backend("copilot").replay_profile is None


@pytest.mark.parametrize("alias", ["copilot", "github_copilot", "copilot_cli", "gh_copilot"])
def test_backend_factory_resolves_profile_relative_to_project(profile, alias):
    backend = get_backend(
        alias, project_dir=str(profile.parent),
        copilot_replay_profile=profile.name, copilot_replay_tools=TOOLS,
    )
    assert backend.replay_profile.directory == str(profile)
    assert backend.replay_profile.tools == tuple(TOOLS)
    assert backend.full_env is False


@pytest.mark.parametrize("tools", [
    None, [], "", "catalog-get_item,catalog-list_items", True, 1, {},
    ["*"], ["catalog-*"], ["catalog-get_?"], ["catalog[read]"], ["catalog(get_item)"],
    ["catalog-get_item,catalog-list_items"], ["--allow-all"], [" tool"], ["tool "],
    ["tool\n"], [""], [False], [None], ["catalog-get_item", "catalog-get_item"],
])
def test_profile_requires_a_nonempty_exact_tool_array(profile, tools):
    with pytest.raises(CopilotReplayError, match="copilot_replay_tools"):
        CopilotCliBackend(copilot_replay_profile=str(profile), copilot_replay_tools=tools)


@pytest.mark.parametrize("value", ["", " ", None, False, 1, [], {}])
def test_tools_without_a_valid_profile_never_use_legacy_home(value):
    with pytest.raises(CopilotReplayError, match="copilot_replay_profile"):
        resolve_replay_profile(value, TOOLS)


def test_missing_profile_and_missing_config_fail_without_creating_them(tmp_path):
    directory = tmp_path / "missing"
    with pytest.raises(CopilotReplayError, match="existing local directory"):
        _backend(directory)
    assert not directory.exists()
    directory.mkdir()
    with pytest.raises(CopilotReplayError, match="readable mcp-config.json"):
        _backend(directory)
    assert list(directory.iterdir()) == []
    with pytest.raises(CopilotReplayError, match="existing local directory"):
        _backend(directory / "mcp-config.json")


@pytest.mark.parametrize("config", [
    None, [], {}, {"servers": {}}, {"mcpServers": []}, {"mcpServers": {}},
    {"mcpServers": {"catalog": None}}, {"mcpServers": {"catalog": {}}},
    {"mcpServers": {"": {"command": "server"}}},
    {"mcpServers": {"catalog": {"command": 1}}},
    {"mcpServers": {"catalog": {"command": "server", "args": "not-an-array"}}},
    {"mcpServers": {"catalog": {"command": "server", "tools": [True]}}},
    {"mcpServers": {"catalog": {"command": "server", "env": {"TOKEN": 1}}}},
    {"mcpServers": {"catalog": {"command": "server", "headers": []}}},
    {"mcpServers": {"catalog": {"command": "server", "type": []}}},
    {"mcpServers": {"catalog": {"command": "server", "type": "unknown"}}},
    {"mcpServers": {"catalog": {"type": "http"}}},
    {"mcpServers": {"catalog": {"type": "http", "url": "https://example.invalid", "command": "server"}}},
])
def test_malformed_mcp_configuration_fails_before_spawn(profile, config):
    (profile / "mcp-config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CopilotReplayError, match="MCP|mcpServers"):
        _backend(profile)


@pytest.mark.parametrize("raw", [
    b"{broken", b"\xff", b'{"mcpServers": {}, "mcpServers": {}}',
    b'{"mcpServers": {"catalog": {"command": "server", "timeout": NaN}}}',
])
def test_invalid_json_is_not_a_personal_home_fallback(profile, raw):
    (profile / "mcp-config.json").write_bytes(raw)
    with pytest.raises(CopilotReplayError, match="valid UTF-8 JSON"):
        _backend(profile)


def test_valid_remote_profile_with_utf8_bom_does_not_need_local_credentials(profile):
    (profile / "mcp-config.json").write_text(json.dumps({
        "mcpServers": {"catalog": {"type": "http", "url": "https://example.invalid/mcp"}},
    }), encoding="utf-8-sig")
    assert _backend(profile).copilot_home == str(profile)


@pytest.mark.parametrize("setting", ["1", "true", "unexpected"])
def test_full_environment_conflict_fails_before_model_resolution(monkeypatch, profile, setting):
    monkeypatch.setenv("SKILLOPT_SLEEP_COPILOT_FULL_ENV", setting)
    with mock.patch.object(backend_mod, "resolve_copilot_argv") as resolve:
        with pytest.raises(CopilotReplayError, match="must be unset or 0"):
            _backend(profile)
    resolve.assert_not_called()


@pytest.mark.parametrize("backend", ["mock", "handoff", "claude"])
def test_profile_cannot_be_silently_ignored_by_other_targets(profile, backend):
    with pytest.raises(CopilotReplayError, match="Copilot replay/target backend"):
        build_backend(
            backend=backend, copilot_replay_profile=str(profile), copilot_replay_tools=TOOLS,
        )


def test_profile_and_tool_array_persist_and_cli_overrides_replace_them(monkeypatch, tmp_path, profile):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "backend": "copilot", "copilot_replay_profile": str(profile),
        "copilot_replay_tools": TOOLS,
    }), encoding="utf-8")
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: str(path))
    saved = cli._cfg_from_args(_parser().parse_args([]))
    assert saved.copilot_replay_profile == str(profile)
    assert saved.copilot_replay_tools == TOOLS
    assert saved.to_dict()["copilot_replay_tools"] == TOOLS

    args = _parser().parse_args([
        "--copilot-replay-profile", "another-profile",
        "--copilot-replay-tool", "catalog-list_items",
        "--copilot-replay-tool", "catalog-read_metadata",
    ])
    changed = cli._cfg_from_args(args)
    assert changed.copilot_replay_profile == "another-profile"
    assert changed.copilot_replay_tools == ["catalog-list_items", "catalog-read_metadata"]
    assert changed.auto_adopt is False


@pytest.mark.parametrize("action", ["run", "dry-run"])
def test_cli_exposes_and_forwards_both_options(monkeypatch, tmp_path, profile, action):
    with mock.patch.object(cli, "run_sleep_cycle", return_value=object()) as run, \
            mock.patch.object(cli, "_print_run_report"):
        result = cli.main([
            action, "--backend", "copilot", "--project", str(tmp_path),
            "--copilot-replay-profile", str(profile),
            "--copilot-replay-tool", TOOLS[0], "--copilot-replay-tool", TOOLS[1],
        ])
    assert result == 0
    cfg = run.call_args.args[0]
    assert cfg.copilot_replay_profile == str(profile)
    assert cfg.copilot_replay_tools == TOOLS
    assert cfg.auto_adopt is False
    assert run.call_args.kwargs["dry_run"] == (action == "dry-run")


@pytest.mark.parametrize("action", ["run", "dry-run"])
@pytest.mark.parametrize("backend", ["copilot", "handoff"])
def test_cli_profile_errors_are_actionable_json_without_model_calls(tmp_path, capsys, action, backend):
    result = cli.main([
        action, "--backend", backend, "--project", str(tmp_path),
        "--claude-home", str(tmp_path / "claude"), "--json",
        "--copilot-replay-profile", str(tmp_path / "missing"),
        "--copilot-replay-tool", TOOLS[0],
    ])
    assert result == 1
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is False
    assert output["error"] == "backend_failed"
    assert "existing local directory" in output["message"]
    assert latest_staging(str(tmp_path)) is None


@pytest.mark.parametrize("action", ["status", "harvest", "adopt", "schedule"])
def test_cli_rejects_replay_flags_on_actions_that_cannot_use_them(action):
    with pytest.raises(SystemExit) as error:
        cli.main([action, "--copilot-replay-profile", "unused"])
    assert error.value.code == 2


@pytest.mark.parametrize("dry_run", [False, True])
def test_cycle_and_diagnostic_construction_forward_settings(tmp_path, profile, dry_run):
    cfg = _config(tmp_path, profile, evidence_log=False)
    with mock.patch.object(cycle, "build_backend", return_value=MockBackend()) as builder:
        cycle._make_model_key(cfg)
        cycle.run_sleep_cycle(cfg, seed_tasks=[], dry_run=dry_run)
    assert builder.call_count == 2
    for call in builder.call_args_list:
        assert call.kwargs["copilot_replay_profile"] == str(profile)
        assert call.kwargs["copilot_replay_tools"] == TOOLS


@pytest.mark.parametrize("split", [False, True])
def test_only_attempts_get_profile_and_exact_tools(monkeypatch, profile, split):
    legacy_home = Path(os.environ["SKILLOPT_SLEEP_COPILOT_HOME"])
    legacy_home.mkdir()
    (legacy_home / "mcp-config.json").write_text(json.dumps(MCP_CONFIG), encoding="utf-8")
    monkeypatch.setenv("COPILOT_AVAILABLE_TOOLS", "bash,write,*")
    monkeypatch.setenv("COPILOT_ALLOW_ALL", "true")
    monkeypatch.setenv("SKILLOPT_SLEEP_COPILOT_FULL_ENV", "0")
    calls = _capture(monkeypatch, text=[
        "answer", "[]", '{"score": 1, "reason": "ok"}',
        '[{"op": "add", "content": "Look up item metadata when requested."}]',
    ])
    backend = build_backend(
        backend="copilot", model="unchanged-model",
        optimizer_model="optimizer-model" if split else "",
        copilot_replay_profile=str(profile), copilot_replay_tools=TOOLS,
    )
    assert isinstance(backend, DualBackend if split else CopilotCliBackend)
    task = _task(reference_kind="rubric", reference="look up an item")
    backend.attempt(task, "", "")
    backend._call("mine tasks")
    backend.judge(task, "answer")
    backend.reflect(
        [(task, ReplayResult(id=task.id, hard=0, soft=0, response="answer", fail_reason="incomplete"))],
        [], "", "", edit_budget=1, evolve_skill=True, evolve_memory=False,
    )
    assert len(calls) == 4
    assert _scope(calls[0]) == ",".join(TOOLS)
    assert calls[0]["env"]["COPILOT_HOME"] == str(profile)
    assert calls[0]["files"] == []
    for call in calls:
        assert call["cmd"][call["cmd"].index("-C") + 1] == call["cwd"]
        assert call["cwd"] not in {str(profile), os.getcwd()}
        assert "--disable-builtin-mcps" in call["cmd"]
        assert "--no-custom-instructions" in call["cmd"]
        assert "--allow-all-tools" in call["cmd"]
        assert "--allow-all" not in call["cmd"]
        assert "COPILOT_ALLOW_ALL" not in call["env"]
        assert "SKILLOPT_SLEEP_COPILOT_HOME" not in call["env"]
        assert "SKILLOPT_SLEEP_COPILOT_FULL_ENV" not in call["env"]
        assert "COPILOT_AVAILABLE_TOOLS" not in call["env"]
        assert not Path(call["cwd"]).exists()
    for call in calls[1:]:
        assert call["env"]["COPILOT_HOME"] == str(Path(call["cwd"]) / ".copilot")
        assert call["home_has_mcp"] is False
        assert _scope(call) == ""
        assert call["cmd"][call["cmd"].index("--available-tools") + 1] == "-C"
    for index, call in enumerate(calls):
        model = call["cmd"][call["cmd"].index("--model") + 1]
        assert model == ("optimizer-model" if split and index else "unchanged-model")


def test_split_can_use_non_copilot_optimizer_without_profile_leak(profile):
    backend = build_backend(
        backend="mock", target_backend="copilot",
        copilot_replay_profile=str(profile), copilot_replay_tools=TOOLS,
    )
    assert isinstance(backend.optimizer, MockBackend)
    assert backend.target.replay_profile.directory == str(profile)


def test_prebuilt_backend_cannot_bypass_profile_or_optimizer_isolation(tmp_path, profile):
    cfg = _config(tmp_path, profile, evidence_log=False)
    for backend in (MockBackend(), DualBackend(_backend(profile), CopilotCliBackend())):
        with pytest.raises(CopilotReplayError, match="supplied backend must match"):
            cycle.run_sleep_cycle(cfg, seed_tasks=[], backend=backend, dry_run=True)
    outcome = cycle.run_sleep_cycle(cfg, seed_tasks=[], backend=_backend(profile), dry_run=True)
    assert outcome.report.n_tasks == 0


def test_explicit_empty_profile_cli_value_does_not_reenable_legacy_mode(tmp_path, capsys):
    result = cli.main([
        "dry-run", "--backend", "copilot", "--project", str(tmp_path),
        "--copilot-replay-profile", "", "--json",
    ])
    assert result == 1
    assert "non-empty local directory" in json.loads(capsys.readouterr().out)["message"]


def test_tool_attempts_share_profile_scope_and_use_real_execution_events(monkeypatch, profile):
    calls = _capture(monkeypatch, events=[
        {"type": "tool.execution_start", "data": {"toolName": TOOLS[0], "toolCallId": "call-1"}},
        {"type": "tool.execution_start", "data": {"toolName": "unrelated-get_item"}},
        {"type": "assistant.message", "data": {"toolRequests": [{"name": TOOLS[1]}]}},
        {"type": "tool.execution_start", "data": ["bad"]},
        {"type": "tool.execution_start", "data": {"toolName": None}},
    ])
    backend = _backend(profile)
    task = _task(reference_kind="rule", judge={
        "kind": "rule", "checks": [{"op": "tool_called", "arg": TOOLS[0]}],
    })
    result = replay_one(backend, task, "", "")
    assert result.tools_called == [TOOLS[0]]
    assert result.hard == 1
    assert _scope(calls[0]) == ",".join(TOOLS)
    assert calls[0]["env"]["COPILOT_HOME"] == str(profile)
    assert calls[0]["files"] == []  # No synthetic shell shims.
    assert "--disable-builtin-mcps" in calls[0]["cmd"]
    assert "--no-custom-instructions" in calls[0]["cmd"]


def test_self_report_and_similarly_named_tools_do_not_count(monkeypatch, profile):
    _capture(monkeypatch, text=f"TOOL_CALL: {TOOLS[0]}", events=[
        {"type": "tool.execution_start", "data": {"toolName": TOOLS[0] + "_other"}},
        {"type": "tool.execution_start", "data": {"toolName": "get_item"}},
    ])
    response, called = _backend(profile).attempt_with_tools(_task(), "", "", TOOLS)
    assert response
    assert called == []


def test_tool_attempt_and_plain_attempt_share_task_system_and_prompt(monkeypatch, profile):
    calls = _capture(monkeypatch)
    backend = _backend(profile)
    task = _task(system="Use the supplied item context.\n{skill_section}", context_excerpt="item context")
    backend.attempt(task, "skill guidance", "memory guidance")
    backend.attempt_with_tools(task, "skill guidance", "memory guidance", TOOLS)
    prompts = [call["cmd"][call["cmd"].index("-p") + 1] for call in calls]
    assert prompts[0] == prompts[1]
    for text in ("Use the supplied item context.", "skill guidance", "memory guidance", "item context"):
        assert text in prompts[0]


def test_tool_judge_cannot_expand_allowlist_or_add_bash(profile):
    with pytest.raises(CopilotReplayError, match="exact names in copilot_replay_tools"):
        _backend(profile).attempt_with_tools(_task(), "", "", ["bash"])


@pytest.mark.parametrize("tool_attempt", [False, True])
def test_profile_removal_after_construction_still_fails_closed(profile, tool_attempt):
    backend = _backend(profile)
    (profile / "mcp-config.json").unlink()
    with pytest.raises(CopilotReplayError, match="readable mcp-config.json"):
        if tool_attempt:
            backend.attempt_with_tools(_task(), "", "", TOOLS)
        else:
            backend.attempt(_task(), "", "")


def test_late_full_env_conflict_is_not_used_even_by_optimizer(monkeypatch, profile):
    backend = _backend(profile)
    monkeypatch.setenv("SKILLOPT_SLEEP_COPILOT_FULL_ENV", "1")
    with pytest.raises(CopilotReplayError, match="must be unset or 0"):
        backend._call("mine tasks")


@pytest.mark.parametrize("tool_attempt", [False, True])
def test_cli_nonzero_is_fatal_even_with_assistant_text(monkeypatch, profile, tool_attempt):
    calls = _capture(monkeypatch, returncode=7)
    backend = _backend(profile)
    with pytest.raises(CopilotReplayError, match="exited 7") as error:
        if tool_attempt:
            backend.attempt_with_tools(_task(), "", "", TOOLS)
        else:
            backend.attempt(_task(), "", "")
    assert len(calls) == 1
    assert "private diagnostic" not in str(error.value)
    assert not Path(calls[0]["cwd"]).exists()


@pytest.mark.parametrize("failure", [OSError("private detail"), subprocess.TimeoutExpired("copilot", 1)])
def test_spawn_failure_is_fatal_with_no_fallback(monkeypatch, profile, failure):
    with mock.patch.object(backend_mod.subprocess, "run", side_effect=failure) as run:
        with pytest.raises(CopilotReplayError, match="No fallback"):
            _backend(profile).attempt(_task(), "", "")
    assert run.call_count == 1


def test_unreadable_profile_is_an_actionable_error(profile):
    with mock.patch("builtins.open", side_effect=PermissionError("private filesystem detail")):
        with pytest.raises(CopilotReplayError, match="readable mcp-config.json") as error:
            _backend(profile)
    assert "private filesystem detail" not in str(error.value)


def test_failed_optimizer_home_creation_cannot_use_personal_home(monkeypatch, profile):
    backend = _backend(profile)
    mkdir = os.mkdir

    def fail_optimizer_home(path, *args, **kwargs):
        if Path(path).name == ".copilot":
            raise PermissionError("cannot create home")
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", fail_optimizer_home)
    with pytest.raises(CopilotReplayError, match="Cannot create an isolated Copilot home"):
        backend._call("mine tasks")


@pytest.mark.parametrize("tool_attempt", [False, True])
def test_empty_profile_response_is_not_a_zero_score(monkeypatch, profile, tool_attempt):
    calls = _capture(monkeypatch, text="")
    backend = _backend(profile)
    with pytest.raises(CopilotReplayError, match="empty"):
        if tool_attempt:
            backend.attempt_with_tools(_task(), "", "", TOOLS)
        else:
            backend.attempt(_task(), "", "")
    assert len(calls) == (1 if tool_attempt else 2)


def test_legacy_environment_and_full_env_remain_compatible(monkeypatch, tmp_path):
    calls = _capture(monkeypatch)
    monkeypatch.setenv("COPILOT_AVAILABLE_TOOLS", "bash,write")
    legacy = CopilotCliBackend()
    legacy._call("legacy")
    legacy.attempt_with_tools(_task(), "", "", ["search"])
    for call in calls:
        assert call["env"]["COPILOT_HOME"] == str(tmp_path / "legacy-home")
        assert _scope(call) == "bash,write"
    assert any(path.name.startswith("search") for path in calls[1]["files"])
    monkeypatch.setenv("SKILLOPT_SLEEP_COPILOT_FULL_ENV", "1")
    full = CopilotCliBackend()
    full._call("full")
    assert full.copilot_home == ""
    assert calls[-1]["env"]["COPILOT_HOME"] == str(tmp_path / "personal-home")
    assert "--disable-builtin-mcps" not in calls[-1]["cmd"]
    assert "--no-custom-instructions" not in calls[-1]["cmd"]


def test_profile_errors_are_not_swallowed_by_mining_or_group_fanout():
    def fail(*args, **kwargs):
        raise CopilotReplayError("invalid replay profile")

    with pytest.raises(CopilotReplayError):
        mine([], llm_miner=fail)
    with pytest.raises(CopilotReplayError):
        consolidate_groups(MockBackend(), [SkillGroup("skill", tasks=[_task()])], consolidate_fn=fail)


def test_cycle_records_scope_but_never_profile_contents(monkeypatch, tmp_path, profile):
    config = json.loads((profile / "mcp-config.json").read_text(encoding="utf-8"))
    config["mcpServers"]["catalog"]["env"] = {"AUTH": "local-auth-placeholder"}
    (profile / "mcp-config.json").write_text(json.dumps(config), encoding="utf-8")
    _capture(monkeypatch)
    outcome = cycle.run_sleep_cycle(
        _config(tmp_path, profile, evolve_skill=False, evolve_memory=False),
        seed_tasks=[_task(reference_kind="exact", reference="answer", split="val")],
    )
    assert outcome.adopted is False
    staging = Path(outcome.staging_dir)
    for path in [staging / "diagnostics.json", staging / "evidence.jsonl"]:
        text = path.read_text(encoding="utf-8")
        assert "local-auth-placeholder" not in text
        assert "example-catalog-mcp" not in text
        assert str(profile) not in text
    diagnostics = json.loads((staging / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics["copilot_replay_profile_enabled"] is True
    assert diagnostics["copilot_replay_tools"] == TOOLS


def test_failed_profile_cycle_does_not_stage_or_advance_state(monkeypatch, tmp_path, profile):
    _capture(monkeypatch, returncode=4)
    cfg = _config(tmp_path, profile)
    with pytest.raises(CopilotReplayError):
        cycle.run_sleep_cycle(
            cfg, seed_tasks=[_task(reference_kind="exact", reference="answer", split="val")],
        )
    assert latest_staging(str(tmp_path)) is None
    assert not Path(cfg.state_path).exists()
