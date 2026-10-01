"""SkillOpt-Sleep — Stage 3: replay.

Re-run mined TaskRecords offline under a given (skill, memory) and score
them, producing the (hard, soft) signal SkillOpt's gate consumes.

Single-shot text replay by default. Tasks whose rule judge requires a tool
call (gbrain's `tool_called`) use the backend's tool-aware path. Backends that
cannot enforce that execution boundary fail explicitly rather than scoring a
self-reported tool call.
"""
from __future__ import annotations

from typing import List, Tuple

from skillopt_sleep.backend import Backend
from skillopt_sleep.types import ReplayResult, TaskRecord


def _required_tools(task: TaskRecord) -> List[str]:
    """Tool names a rule judge requires (op == 'tool_called')."""
    if task.reference_kind != "rule" or not task.judge:
        return []
    tools = []
    for c in task.judge.get("checks", []) or []:
        if isinstance(c, dict) and c.get("op") == "tool_called" and c.get("arg"):
            tools.append(str(c["arg"]))
    return tools


def replay_one(backend: Backend, task: TaskRecord, skill: str, memory: str,
               sample_id: int = 0) -> ReplayResult:
    """``sample_id`` distinguishes repeated dream rollouts of the same
    (task, skill, memory) in the attempt cache — without it all K rollouts
    collapse to one cached response and the contrastive signal is always 0."""
    import time
    tools = _required_tools(task)
    tools_called: List[str] = []
    t0 = time.time()
    tok_before = backend.tokens_used()
    if tools:
        response, tools_called = backend.attempt_with_tools(task, skill, memory, tools)
    else:
        response = backend.attempt(task, skill, memory, sample_id=sample_id)
    latency_ms = (time.time() - t0) * 1000.0
    tokens = max(0, backend.tokens_used() - tok_before)
    # if the backend doesn't track tokens (e.g. mock), approximate from text length
    if tokens == 0:
        tokens = (len(skill) + len(memory) + len(task.intent) + len(response)) // 4

    # rule judges may need the detected tool calls; score locally when possible
    if task.reference_kind == "rule" and task.judge:
        from skillopt_sleep.judges import score_rule_judge_with_feedback
        hard, soft, rationale, optimizer_feedback = score_rule_judge_with_feedback(
            task.judge, response, tools_called
        )
    else:
        hard, soft, rationale = backend.judge(task, response)
        # Backend judge rationales are audit evidence and may contain provider
        # or grader implementation details. Keep the optimizer channel generic
        # for non-rule judges unless a future typed safe-feedback API exists.
        optimizer_feedback = (
            "The response did not satisfy the task's evaluation criteria."
            if hard < 1.0
            else ""
        )

    ev = getattr(backend, "evidence", None)
    if ev is not None:
        # One scored-attempt record per (phase, task): the task->score link of
        # the evidentiary chain, with the failing checks named in `why`.
        ev.log("replay", "result",
               phase=getattr(backend, "evidence_phase", ""),
               task_id=task.id, split=task.split, origin=task.origin,
               reference_kind=task.reference_kind, sample_id=sample_id,
               hard=float(hard), soft=float(soft), why=rationale or "",
               response_head=(response or "")[:400], tools_called=tools_called,
               tokens=int(tokens), latency_ms=round(latency_ms, 1))

    return ReplayResult(
        id=task.id,
        hard=float(hard),
        soft=float(soft),
        response=response,
        fail_reason="" if hard >= 1.0 else (rationale or "below threshold"),
        task_type=(task.tags[0] if task.tags else "task"),
        judge_rationale=rationale,
        tools_called=tools_called,
        tokens=int(tokens),
        latency_ms=round(latency_ms, 1),
        optimizer_feedback=(optimizer_feedback if hard < 1.0 else ""),
    )


import os
from concurrent.futures import ThreadPoolExecutor


def _collapse_samples(
    task: TaskRecord,
    samples: List[ReplayResult],
    backend: Backend,
) -> ReplayResult:
    """Reduce K samples of one task to a single result, and record the spread.

    Mean, not median. Median was the first choice here -- the observed failure
    is bimodal (the agent either produces the deliverable or stalls emitting a
    tool call), and a median looks like the robust option for bimodal data. It
    is the wrong one, for two measured reasons. Simulating a task that succeeds
    60% of the time, over 200 runs of a 4-task set:

        K      median sd    mean sd
        1        0.246        0.246
        5        0.246        0.111
        15       0.207        0.064

    The median barely converges, because with a success rate near one half each
    task's median is itself a coin flip. Worse, it is *biased*: it converges to
    1.000 and reports a skill that stalls 40% of the time as perfect. The mean
    converges as roughly 1/sqrt(K) and reports 0.600, which is the truth.

    Hiding an intermittent stall is precisely the failure this function exists
    to expose, so the robust-looking choice was the harmful one.

    The spread is logged rather than discarded. Without it a score looks precise
    when it is not: the same unmodified skill was measured at 0.538 and 1.000 on
    consecutive runs of an identical task set, and nothing in the output said so.
    """
    hards = [r.hard for r in samples]
    softs = [r.soft for r in samples]
    mean_hard = sum(hards) / len(hards)
    mean_soft = sum(softs) / len(softs)
    # Keep the transcript of the sample nearest the mean, so `response_head` and
    # `fail_reason` describe a real attempt rather than a synthetic average.
    chosen = min(samples, key=lambda r: abs(r.soft - mean_soft))

    ev = getattr(backend, "evidence", None)
    if ev is not None and len(samples) > 1:
        ev.log("replay", "samples",
               phase=getattr(backend, "evidence_phase", ""),
               task_id=task.id, split=task.split, k=len(samples),
               hard_samples=[round(h, 4) for h in hards],
               soft_samples=[round(s, 4) for s in softs],
               hard_mean=round(mean_hard, 4),
               soft_mean=round(mean_soft, 4),
               soft_spread=round(max(softs) - min(softs), 4),
               unstable=bool(max(softs) - min(softs) > 0.25))

    return ReplayResult(
        id=chosen.id,
        hard=mean_hard,
        soft=mean_soft,
        response=chosen.response,
        fail_reason=chosen.fail_reason,
        task_type=chosen.task_type,
        judge_rationale=chosen.judge_rationale,
        tools_called=chosen.tools_called,
        tokens=int(sum(r.tokens for r in samples) / len(samples)),
        latency_ms=round(sum(r.latency_ms for r in samples) / len(samples), 1),
    )


def replay_batch(
    backend: Backend,
    tasks: List[TaskRecord],
    skill: str,
    memory: str,
    *,
    workers: int = 0,
    k: int = 1,
) -> List[Tuple[TaskRecord, ReplayResult]]:
    """Replay tasks, optionally in parallel and optionally repeated.

    Real backends are network-bound, so a thread pool gives a large speedup on
    big test sets (like the research harness's --workers). ``workers`` defaults
    to env SKILLOPT_SLEEP_WORKERS or 1 (sequential). Mock stays sequential
    (deterministic) unless asked otherwise.

    ``k`` repeats every task K times and collapses each to its median. One
    sample per task is not a measurement when the backend is nondeterministic:
    an identical skill and task set scored 0.538 on one run and 1.000 on the
    next, because the agent sometimes stalls emitting a tool call instead of the
    deliverable. A gate comparing one baseline sample against one candidate
    sample is unreliable in both directions -- it can manufacture an improvement
    and it can hide one. Cost scales linearly with K.

    Note this is distinct from ``dream_rollouts``/``rollouts_k``, which repeats
    TRAIN tasks to give the optimizer contrastive signal. That path never
    touched the validation scoring, so it does not address measurement variance.
    """
    if workers <= 0:
        workers = int(os.environ.get("SKILLOPT_SLEEP_WORKERS", "1") or "1")
    k = max(1, int(k))

    # (task_index, sample_id) is the unit of work, so K samples of one task can
    # run concurrently with samples of another rather than serialising per task.
    units = [(i, s) for i in range(len(tasks)) for s in range(k)]

    if workers <= 1 or len(units) <= 1:
        raw = [(i, replay_one(backend, tasks[i], skill, memory, sample_id=s))
               for i, s in units]
    else:
        raw = []
        with ThreadPoolExecutor(max_workers=min(workers, len(units))) as ex:
            futs = {ex.submit(replay_one, backend, tasks[i], skill, memory, s): i
                    for i, s in units}
            for fut in futs:
                raw.append((futs[fut], fut.result()))

    if k == 1:
        by_index = {i: r for i, r in raw}
        return [(t, by_index[i]) for i, t in enumerate(tasks)]

    grouped: dict = {}
    for i, r in raw:
        grouped.setdefault(i, []).append(r)
    return [(t, _collapse_samples(t, grouped[i], backend))
            for i, t in enumerate(tasks)]


def aggregate_scores(pairs: List[Tuple[TaskRecord, ReplayResult]]) -> Tuple[float, float]:
    if not pairs:
        return 0.0, 0.0
    hard = sum(r.hard for _t, r in pairs) / len(pairs)
    soft = sum(r.soft for _t, r in pairs) / len(pairs)
    return hard, soft


def aggregate_cost(pairs: List[Tuple[TaskRecord, ReplayResult]]) -> Tuple[float, float]:
    """Mean (tokens, latency_ms) per task — the cost objectives."""
    if not pairs:
        return 0.0, 0.0
    tok = sum(r.tokens for _t, r in pairs) / len(pairs)
    lat = sum(r.latency_ms for _t, r in pairs) / len(pairs)
    return tok, lat


def multi_objective_reward(
    pairs: List[Tuple[TaskRecord, ReplayResult]],
    *,
    w_acc: float = 1.0,
    w_tokens: float = 0.0,
    w_latency: float = 0.0,
    token_ref: float = 2000.0,
    latency_ref_ms: float = 15000.0,
) -> float:
    """Weighted reward = accuracy↑, tokens↓, latency↓.

    Cost terms are normalized against a reference and clamped to [0,1], so a
    response at/under the reference cost contributes ~1.0 and an expensive one
    less. Weights let the user trade off (default = accuracy only, backward
    compatible).
    """
    if not pairs:
        return 0.0
    acc, _soft = aggregate_scores(pairs)
    tok, lat = aggregate_cost(pairs)
    tok_score = max(0.0, 1.0 - tok / max(1.0, token_ref)) if token_ref else 0.0
    lat_score = max(0.0, 1.0 - lat / max(1.0, latency_ref_ms)) if latency_ref_ms else 0.0
    total_w = w_acc + w_tokens + w_latency
    if total_w <= 0:
        return acc
    return (w_acc * acc + w_tokens * tok_score + w_latency * lat_score) / total_w
