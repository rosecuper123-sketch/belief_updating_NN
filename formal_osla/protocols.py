from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Iterable

import numpy as np
import torch


INPUT_FIELDS = (
    "prev_action",
    "prev_outcome",
    "prev_token_delta",
    "current_total_tokens_norm",
    "trial_position",
    "boundary_flag",
    "cue_logit",
)

FORMAL_GROUPS = ("mac1A", "mac1B", "mac2A", "mac2B")

FORMAL_CONFLICTS: dict[str, dict[str, float]] = {
    "mac1A": {"consistent_cue": 0.70, "consistent_true": 0.70, "conflict_cue": 0.30, "conflict_true": 0.60},
    "mac1B": {"consistent_cue": 0.30, "consistent_true": 0.30, "conflict_cue": 0.70, "conflict_true": 0.40},
    "mac2A": {"consistent_cue": 0.70, "consistent_true": 0.70, "conflict_cue": 0.30, "conflict_true": 0.50},
    "mac2B": {"consistent_cue": 0.30, "consistent_true": 0.30, "conflict_cue": 0.70, "conflict_true": 0.50},
}


@dataclass(frozen=True, slots=True)
class RoundSpec:
    group: str
    protocol: str
    cue_probability: float
    true_probability: float
    kind: str
    round_id: int
    feedback_trials: int = 120

    @property
    def mismatch_flag(self) -> bool:
        return abs(self.cue_probability - self.true_probability) > 1e-8


@dataclass(slots=True)
class RoundBatch:
    cue_probability: torch.Tensor
    true_probability: torch.Tensor
    group: tuple[str, ...]
    protocol: str
    kind: str
    round_id: int
    feedback_trials: int

    @property
    def mismatch_flag(self) -> torch.Tensor:
        return self.cue_probability.ne(self.true_probability)


def probability_logit(value: float | torch.Tensor, eps: float = 1e-5) -> float | torch.Tensor:
    if isinstance(value, torch.Tensor):
        p = torch.clamp(value, eps, 1.0 - eps)
        return torch.log(p / (1.0 - p))
    p = min(max(float(value), eps), 1.0 - eps)
    return log(p / (1.0 - p))


def sample_consistent_round_batches(
    *,
    batch_size: int,
    n_rounds: int,
    probability_grid: Iterable[float],
    feedback_trials: int,
    generator: torch.Generator,
    device: torch.device,
) -> list[RoundBatch]:
    grid = torch.tensor(tuple(probability_grid), dtype=torch.float32, device=device)
    if grid.ndim != 1 or grid.numel() == 0:
        raise ValueError("probability_grid must contain at least one value")
    batches: list[RoundBatch] = []
    for rid in range(1, n_rounds + 1):
        idx = torch.randint(grid.numel(), (batch_size,), generator=generator, device=device)
        prob = grid[idx]
        batches.append(
            RoundBatch(
                cue_probability=prob,
                true_probability=prob.clone(),
                group=tuple("train_consistent" for _ in range(batch_size)),
                protocol="conditionA_train",
                kind="consistent",
                round_id=rid,
                feedback_trials=feedback_trials,
            )
        )
    return batches


def formal_test_rounds(
    *,
    group: str,
    protocol: str = "human_default",
    feedback_trials: int = 120,
    custom: list[tuple[str, float, float]] | None = None,
) -> list[RoundSpec]:
    if group not in FORMAL_CONFLICTS:
        raise ValueError(f"unknown formal group {group!r}; choose from {FORMAL_GROUPS}")
    spec = FORMAL_CONFLICTS[group]
    if custom is not None:
        rounds = [
            RoundSpec(
                group=group,
                protocol="custom",
                kind=kind,
                cue_probability=float(cue),
                true_probability=float(true),
                round_id=i + 1,
                feedback_trials=feedback_trials,
            )
            for i, (kind, cue, true) in enumerate(custom)
        ]
        validate_test_rounds(rounds)
        return rounds

    if protocol == "human_default":
        layout = ("consistent", "conflict")
    elif protocol == "legacy_3plus1":
        layout = ("consistent", "consistent", "consistent", "conflict")
    elif protocol == "conflict_only":
        layout = ("conflict",)
    else:
        raise ValueError("protocol must be human_default, legacy_3plus1, conflict_only, or custom")

    rounds: list[RoundSpec] = []
    for rid, kind in enumerate(layout, start=1):
        if kind == "consistent":
            cue = spec["consistent_cue"]
            true = spec["consistent_true"]
        else:
            cue = spec["conflict_cue"]
            true = spec["conflict_true"]
        rounds.append(RoundSpec(group, protocol, cue, true, kind, rid, feedback_trials))
    validate_test_rounds(rounds)
    return rounds


def expand_round_specs_to_batch(
    rounds: list[RoundSpec],
    *,
    episodes: int,
    device: torch.device,
) -> list[RoundBatch]:
    batches: list[RoundBatch] = []
    for round_spec in rounds:
        batches.append(
            RoundBatch(
                cue_probability=torch.full((episodes,), round_spec.cue_probability, device=device),
                true_probability=torch.full((episodes,), round_spec.true_probability, device=device),
                group=tuple(round_spec.group for _ in range(episodes)),
                protocol=round_spec.protocol,
                kind=round_spec.kind,
                round_id=round_spec.round_id,
                feedback_trials=round_spec.feedback_trials,
            )
        )
    return batches


def validate_consistent_only(rounds: Iterable[RoundBatch | RoundSpec]) -> None:
    for round_spec in rounds:
        cue = round_spec.cue_probability
        true = round_spec.true_probability
        if isinstance(cue, torch.Tensor):
            if torch.any(cue.ne(true)).item():
                raise ValueError("training/validation rounds must be consistent-only")
        elif abs(float(cue) - float(true)) > 1e-8:
            raise ValueError("training/validation rounds must be consistent-only")


def validate_test_rounds(rounds: list[RoundSpec]) -> None:
    for round_spec in rounds:
        if round_spec.feedback_trials != 120:
            raise ValueError("all formal test rounds must have exactly 120 behavioral trials")
        if round_spec.kind == "consistent" and round_spec.mismatch_flag:
            raise ValueError("consistent test rounds must have cue == true")
        if round_spec.kind == "conflict" and not round_spec.mismatch_flag:
            raise ValueError("conflict test rounds must have cue != true")


def block_balanced_outcomes(
    true_probability: torch.Tensor,
    n_trials: int,
    block_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if n_trials % block_size != 0:
        raise ValueError("n_trials must be divisible by block_size")
    batch = true_probability.numel()
    device = true_probability.device
    blocks = n_trials // block_size
    outcomes = torch.empty(batch, n_trials, dtype=torch.float32, device=device)
    for b in range(batch):
        p = float(true_probability[b].item())
        wins = int(round(p * block_size))
        if not np.isclose(wins / block_size, p):
            raise ValueError(f"probability {p} is incompatible with block_size={block_size}")
        block = torch.cat(
            [
                torch.ones(wins, dtype=torch.float32, device=device),
                torch.zeros(block_size - wins, dtype=torch.float32, device=device),
            ]
        )
        for k in range(blocks):
            outcomes[b, k * block_size : (k + 1) * block_size] = block[
                torch.randperm(block_size, generator=generator, device=device)
            ]
    return outcomes
