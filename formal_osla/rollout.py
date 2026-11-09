from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import FormalOneShotGRU
from .protocols import RoundBatch, block_balanced_outcomes, probability_logit


@dataclass(slots=True)
class RolloutResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    actions: torch.Tensor
    outcomes: torch.Tensor
    rewards: torch.Tensor
    hidden: torch.Tensor
    trial_inputs: torch.Tensor
    cue_probability: torch.Tensor
    true_probability: torch.Tensor
    mismatch_flag: torch.Tensor
    round_index: torch.Tensor
    trial_id: torch.Tensor
    current_total_tokens: torch.Tensor
    current_total_tokens_norm: torch.Tensor
    prev_action: torch.Tensor
    prev_outcome: torch.Tensor
    prev_token_delta: torch.Tensor
    prediction_error: torch.Tensor
    h_pre_boundary: torch.Tensor
    h_post_boundary: torch.Tensor
    p_boundary: torch.Tensor
    boundary_cue_probability: torch.Tensor
    boundary_true_probability: torch.Tensor
    boundary_mismatch_flag: torch.Tensor

    @property
    def total_return(self) -> torch.Tensor:
        return self.rewards.sum(dim=1)


def rollout_episode(
    model: FormalOneShotGRU,
    rounds: list[RoundBatch],
    *,
    block_size: int,
    generator: torch.Generator,
    deterministic: bool = False,
    start_tokens: int = 100,
) -> RolloutResult:
    if not rounds:
        raise ValueError("rounds must not be empty")
    batch = rounds[0].cue_probability.shape[0]
    device = rounds[0].cue_probability.device
    dtype = rounds[0].cue_probability.dtype
    hidden = model.initial_hidden(batch, device, dtype)

    logits_list: list[torch.Tensor] = []
    probabilities_list: list[torch.Tensor] = []
    actions_list: list[torch.Tensor] = []
    outcomes_list: list[torch.Tensor] = []
    rewards_list: list[torch.Tensor] = []
    hidden_list: list[torch.Tensor] = []
    input_list: list[torch.Tensor] = []
    cue_list: list[torch.Tensor] = []
    true_list: list[torch.Tensor] = []
    mismatch_list: list[torch.Tensor] = []
    round_list: list[torch.Tensor] = []
    trial_list: list[torch.Tensor] = []
    token_list: list[torch.Tensor] = []
    token_norm_list: list[torch.Tensor] = []
    prev_action_list: list[torch.Tensor] = []
    prev_outcome_list: list[torch.Tensor] = []
    prev_delta_list: list[torch.Tensor] = []
    prediction_error_list: list[torch.Tensor] = []

    h_pre_list: list[torch.Tensor] = []
    h_post_list: list[torch.Tensor] = []
    p_boundary_list: list[torch.Tensor] = []
    boundary_cue_list: list[torch.Tensor] = []
    boundary_true_list: list[torch.Tensor] = []
    boundary_mismatch_list: list[torch.Tensor] = []

    for round_batch in rounds:
        if round_batch.cue_probability.shape != (batch,):
            raise ValueError("all rounds must share batch size")

        h_pre = hidden
        boundary_input = torch.zeros(batch, 7, dtype=dtype, device=device)
        boundary_input[:, 5] = 1.0
        boundary_input[:, 6] = probability_logit(round_batch.cue_probability)
        hidden, _boundary_logits, p_boundary = model.step(boundary_input, hidden)

        h_pre_list.append(h_pre)
        h_post_list.append(hidden)
        p_boundary_list.append(p_boundary)
        boundary_cue_list.append(round_batch.cue_probability)
        boundary_true_list.append(round_batch.true_probability)
        boundary_mismatch_list.append(round_batch.mismatch_flag.to(dtype))

        prev_action = torch.zeros(batch, dtype=dtype, device=device)
        prev_outcome = torch.zeros(batch, dtype=dtype, device=device)
        prev_token_delta = torch.zeros(batch, dtype=dtype, device=device)
        total_tokens = torch.full((batch,), float(start_tokens), dtype=dtype, device=device)
        outcomes = block_balanced_outcomes(
            round_batch.true_probability,
            round_batch.feedback_trials,
            block_size,
            generator,
        )

        for trial in range(round_batch.feedback_trials):
            current_total_tokens_norm = (total_tokens - float(start_tokens)) / float(round_batch.feedback_trials)
            trial_input = torch.zeros(batch, 7, dtype=dtype, device=device)
            trial_input[:, 0] = prev_action
            trial_input[:, 1] = prev_outcome
            trial_input[:, 2] = prev_token_delta
            trial_input[:, 3] = current_total_tokens_norm
            trial_input[:, 4] = trial / max(round_batch.feedback_trials - 1, 1)

            hidden, logits, probabilities = model.step(trial_input, hidden)
            if deterministic:
                action = (probabilities >= 0.5).to(dtype)
            else:
                action = torch.bernoulli(probabilities, generator=generator)
            current_outcome = outcomes[:, trial]
            reward = action * (2.0 * current_outcome - 1.0)
            prediction_error = current_outcome - round_batch.true_probability

            logits_list.append(logits)
            probabilities_list.append(probabilities)
            actions_list.append(action)
            outcomes_list.append(current_outcome)
            rewards_list.append(reward)
            hidden_list.append(hidden)
            input_list.append(trial_input)
            cue_list.append(round_batch.cue_probability)
            true_list.append(round_batch.true_probability)
            mismatch_list.append(round_batch.mismatch_flag.to(dtype))
            round_list.append(torch.full((batch,), float(round_batch.round_id), dtype=dtype, device=device))
            trial_list.append(torch.full((batch,), float(trial + 1), dtype=dtype, device=device))
            token_list.append(total_tokens)
            token_norm_list.append(current_total_tokens_norm)
            prev_action_list.append(prev_action)
            prev_outcome_list.append(prev_outcome)
            prev_delta_list.append(prev_token_delta)
            prediction_error_list.append(prediction_error)

            prev_action = action.detach()
            prev_outcome = current_outcome.detach()
            prev_token_delta = reward.detach()
            total_tokens = total_tokens + reward.detach()

    return RolloutResult(
        logits=torch.stack(logits_list, dim=1),
        probabilities=torch.stack(probabilities_list, dim=1),
        actions=torch.stack(actions_list, dim=1),
        outcomes=torch.stack(outcomes_list, dim=1),
        rewards=torch.stack(rewards_list, dim=1),
        hidden=torch.stack(hidden_list, dim=1),
        trial_inputs=torch.stack(input_list, dim=1),
        cue_probability=torch.stack(cue_list, dim=1),
        true_probability=torch.stack(true_list, dim=1),
        mismatch_flag=torch.stack(mismatch_list, dim=1),
        round_index=torch.stack(round_list, dim=1),
        trial_id=torch.stack(trial_list, dim=1),
        current_total_tokens=torch.stack(token_list, dim=1),
        current_total_tokens_norm=torch.stack(token_norm_list, dim=1),
        prev_action=torch.stack(prev_action_list, dim=1),
        prev_outcome=torch.stack(prev_outcome_list, dim=1),
        prev_token_delta=torch.stack(prev_delta_list, dim=1),
        prediction_error=torch.stack(prediction_error_list, dim=1),
        h_pre_boundary=torch.stack(h_pre_list, dim=1),
        h_post_boundary=torch.stack(h_post_list, dim=1),
        p_boundary=torch.stack(p_boundary_list, dim=1),
        boundary_cue_probability=torch.stack(boundary_cue_list, dim=1),
        boundary_true_probability=torch.stack(boundary_true_list, dim=1),
        boundary_mismatch_flag=torch.stack(boundary_mismatch_list, dim=1),
    )


def reinforce_loss(result: RolloutResult, *, entropy_bonus: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    log_prob = -F.binary_cross_entropy_with_logits(result.logits, result.actions.detach(), reduction="none")
    policy_loss = -(log_prob * result.rewards.detach()).mean()
    entropy = torch.distributions.Bernoulli(probs=result.probabilities).entropy().mean()
    loss = policy_loss - float(entropy_bonus) * entropy
    return loss, {
        "loss": float(loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "entropy": float(entropy.detach().cpu().item()),
        "mean_return": float(result.total_return.detach().mean().cpu().item()),
        "mean_bet_rate": float(result.actions.detach().mean().cpu().item()),
    }
