"""SkillOpt-Sleep — vendored validation gate.

This is a self-contained copy of the SkillOpt validation gate so the sleep
engine has ZERO dependency on the research package (skillopt/*). The research
repo's ``skillopt.evaluation.gate`` is the reference implementation and the two
are kept behaviourally identical; vendoring keeps this open-source tool
decoupled from the paper's experiment code.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class GateResult:
    action: str            # "accept_new_best" | "accept" | "reject"
    current_skill: str
    current_score: float
    best_skill: str
    best_score: float
    best_step: int


def select_gate_score(hard: float, soft: float, metric: str = "hard",
                      mixed_weight: float = 0.5) -> float:
    """Project (hard, soft) onto a single comparison metric."""
    if metric == "hard":
        return float(hard)
    if metric == "soft":
        return float(soft)
    if metric == "mixed":
        w = max(0.0, min(1.0, float(mixed_weight)))
        return (1.0 - w) * float(hard) + w * float(soft)
    raise ValueError(f"unknown gate metric {metric!r}; expected hard/soft/mixed")


def passes_margin(
    base_score: float,
    cand_score: float,
    task_deltas: Sequence[Mapping[str, Any]] | None = None,
    *,
    min_margin: float = 0.0,
    bootstrap: int = 0,
    min_bootstrap_n: int = 8,
    seed: int = 12345,
) -> Tuple[bool, str]:
    """Is the improvement large enough to be worth believing?

    ``cand_score > base_score`` alone accepts any epsilon. An observed run
    accepted on +0.0056 -- a total delta of 0.05 across nine validation tasks,
    i.e. a single task moving one notch on a judge whose smallest observed step
    was 0.05 -- while that same candidate dropped the only task the baseline had
    answered perfectly from 1.00 to 0.85.

    Two independent hurdles, both optional so the historical behaviour is the
    default:

    * ``min_margin`` -- the mean must improve by more than this.
    * ``bootstrap`` -- resample the per-task deltas with replacement N times and
      require the 5th percentile of the mean delta to clear ``min_margin`` too.
      This is what rejects an improvement carried by one or two tasks out of
      many, which a mean cannot distinguish from a broad small gain.

    ``min_bootstrap_n`` exists because the resampling test has no power on a tiny
    holdout and produces false negatives there. Measured on a real accepted run:
    four validation tasks, two genuinely fixed (0.08 -> 1.00) and two already at
    ceiling (1.00 -> 1.00, delta 0). Drawing four samples from ``[0, 0, +0.92,
    +0.92]`` yields all-zeros 6.25% of the time, so the 5th percentile is exactly
    0.00 and an unambiguous improvement is rejected. Below the threshold the
    margin check stands alone.

    Returns ``(passed, reason)``; the reason is recorded in the evidence log so a
    rejection is never mistaken for "the candidate was simply worse".
    """
    delta = float(cand_score) - float(base_score)
    if delta <= min_margin:
        return False, (
            f"margin {delta:+.4f} <= required {min_margin:.4f}"
            if min_margin > 0 else f"no improvement ({delta:+.4f})"
        )
    if bootstrap and task_deltas:
        per_task = [
            float(r.get("candidate_score", 0.0)) - float(r.get("baseline_score", 0.0))
            for r in task_deltas
            if r.get("scores_are_finite", True)
        ]
        if len(per_task) < min_bootstrap_n:
            return True, (
                f"margin {delta:+.4f}; bootstrap skipped "
                f"(n={len(per_task)} < {min_bootstrap_n}, no power)"
            )
        rng = random.Random(seed)
        n = len(per_task)
        means = sorted(
            sum(rng.choice(per_task) for _ in range(n)) / n
            for _ in range(bootstrap)
        )
        p05 = means[max(0, int(0.05 * bootstrap) - 1)]
        if p05 <= min_margin:
            return False, (
                f"margin {delta:+.4f} holds on the mean but the bootstrap "
                f"5th percentile is {p05:+.4f} (n={n}, {bootstrap} resamples)"
            )
        return True, f"margin {delta:+.4f}, bootstrap p05 {p05:+.4f} (n={n})"
    return True, f"margin {delta:+.4f}"


def evaluate_gate(candidate_skill: str, cand_hard: float, current_skill: str,
                  current_score: float, best_skill: str, best_score: float,
                  best_step: int, global_step: int, *, cand_soft: float = 0.0,
                  metric: str = "hard", mixed_weight: float = 0.5) -> GateResult:
    """Pure gate decision: compare candidate score to current/best."""
    cand_score = select_gate_score(cand_hard, cand_soft, metric, mixed_weight)
    if cand_score > current_score:
        if cand_score > best_score:
            return GateResult("accept_new_best", candidate_skill, cand_score,
                              candidate_skill, cand_score, global_step)
        return GateResult("accept", candidate_skill, cand_score,
                          best_skill, best_score, best_step)
    return GateResult("reject", current_skill, current_score,
                      best_skill, best_score, best_step)
