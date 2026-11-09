from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from .rollout import RolloutResult


TRIAL_FIELDS = [
    "episode_id",
    "round_id",
    "trial_id",
    "global_trial",
    "group",
    "protocol",
    "cue_probability",
    "true_probability",
    "mismatch_flag",
    "action",
    "outcome",
    "reward",
    "prev_action",
    "prev_outcome",
    "prev_token_delta",
    "current_total_tokens",
    "current_total_tokens_norm",
    "choice_probability",
    "prediction_error",
    "network_id",
    "seed",
]

BOUNDARY_FIELDS = [
    "episode_id",
    "round_id",
    "group",
    "protocol",
    "cue_probability",
    "true_probability",
    "mismatch_flag",
    "boundary_flag",
    "cue_logit",
    "p_boundary",
    "network_id",
    "seed",
]


def _cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def write_rollout_outputs(
    result: RolloutResult,
    *,
    out_dir: Path,
    network_id: str,
    seed: int,
    groups: tuple[str, ...],
    protocol: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_trials_csv(result, out_dir / "trials.csv", network_id, seed, groups, protocol)
    _write_boundary_csv(result, out_dir / "boundary_events.csv", network_id, seed, groups, protocol)
    _write_hidden_npz(result, out_dir / "hidden.npz", groups, protocol)
    _write_boundary_npz(result, out_dir / "boundary_hidden.npz", groups, protocol)


def _write_trials_csv(
    result: RolloutResult,
    path: Path,
    network_id: str,
    seed: int,
    groups: tuple[str, ...],
    protocol: str,
) -> None:
    arrays = {name: _cpu(getattr(result, name)) for name in (
        "round_index",
        "trial_id",
        "cue_probability",
        "true_probability",
        "mismatch_flag",
        "actions",
        "outcomes",
        "rewards",
        "prev_action",
        "prev_outcome",
        "prev_token_delta",
        "current_total_tokens",
        "current_total_tokens_norm",
        "probabilities",
        "prediction_error",
    )}
    n_episodes, n_trials = arrays["actions"].shape
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        for ep in range(n_episodes):
            for t in range(n_trials):
                writer.writerow(
                    {
                        "episode_id": ep,
                        "round_id": int(arrays["round_index"][ep, t]),
                        "trial_id": int(arrays["trial_id"][ep, t]),
                        "global_trial": t + 1,
                        "group": groups[ep],
                        "protocol": protocol,
                        "cue_probability": arrays["cue_probability"][ep, t],
                        "true_probability": arrays["true_probability"][ep, t],
                        "mismatch_flag": int(arrays["mismatch_flag"][ep, t]),
                        "action": int(arrays["actions"][ep, t]),
                        "outcome": int(arrays["outcomes"][ep, t]),
                        "reward": int(arrays["rewards"][ep, t]),
                        "prev_action": int(arrays["prev_action"][ep, t]),
                        "prev_outcome": int(arrays["prev_outcome"][ep, t]),
                        "prev_token_delta": int(arrays["prev_token_delta"][ep, t]),
                        "current_total_tokens": int(arrays["current_total_tokens"][ep, t]),
                        "current_total_tokens_norm": arrays["current_total_tokens_norm"][ep, t],
                        "choice_probability": arrays["probabilities"][ep, t],
                        "prediction_error": arrays["prediction_error"][ep, t],
                        "network_id": network_id,
                        "seed": seed,
                    }
                )


def _write_boundary_csv(
    result: RolloutResult,
    path: Path,
    network_id: str,
    seed: int,
    groups: tuple[str, ...],
    protocol: str,
) -> None:
    cue = _cpu(result.boundary_cue_probability)
    true = _cpu(result.boundary_true_probability)
    mismatch = _cpu(result.boundary_mismatch_flag)
    p_boundary = _cpu(result.p_boundary)
    n_episodes, n_rounds = cue.shape
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOUNDARY_FIELDS)
        writer.writeheader()
        for ep in range(n_episodes):
            for rid in range(n_rounds):
                c = float(cue[ep, rid])
                writer.writerow(
                    {
                        "episode_id": ep,
                        "round_id": rid + 1,
                        "group": groups[ep],
                        "protocol": protocol,
                        "cue_probability": c,
                        "true_probability": float(true[ep, rid]),
                        "mismatch_flag": int(mismatch[ep, rid]),
                        "boundary_flag": 1,
                        "cue_logit": float(np.log(np.clip(c, 1e-5, 1.0 - 1e-5) / np.clip(1.0 - c, 1e-5, 1.0))),
                        "p_boundary": float(p_boundary[ep, rid]),
                        "network_id": network_id,
                        "seed": seed,
                    }
                )


def _write_hidden_npz(result: RolloutResult, path: Path, groups: tuple[str, ...], protocol: str) -> None:
    np.savez(
        path,
        hidden=_cpu(result.hidden),
        prediction_error=_cpu(result.prediction_error),
        actions=_cpu(result.actions),
        outcomes=_cpu(result.outcomes),
        rewards=_cpu(result.rewards),
        episode_ids=np.arange(result.actions.shape[0]),
        groups=np.asarray(groups),
        protocol=np.asarray(protocol),
    )


def _write_boundary_npz(result: RolloutResult, path: Path, groups: tuple[str, ...], protocol: str) -> None:
    np.savez(
        path,
        h_pre_boundary=_cpu(result.h_pre_boundary),
        h_post_boundary=_cpu(result.h_post_boundary),
        delta_h_boundary=_cpu(result.h_post_boundary - result.h_pre_boundary),
        p_boundary=_cpu(result.p_boundary),
        episode_ids=np.arange(result.p_boundary.shape[0]),
        groups=np.asarray(groups),
        protocol=np.asarray(protocol),
        cue_probability=_cpu(result.boundary_cue_probability),
        true_probability=_cpu(result.boundary_true_probability),
        mismatch_flag=_cpu(result.boundary_mismatch_flag),
    )
