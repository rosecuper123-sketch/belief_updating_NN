from __future__ import annotations

import csv
import json
import time
from copy import deepcopy
from pathlib import Path

import torch

from .config import ExperimentConfig
from .model import FormalOneShotGRU, MODEL_VERSION
from .protocols import sample_consistent_round_batches, validate_consistent_only
from .rollout import reinforce_loss, rollout_episode


def build_model(config: ExperimentConfig) -> FormalOneShotGRU:
    return FormalOneShotGRU(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        dropout=config.model.dropout,
    )


def train_formal(config: ExperimentConfig, *, device: torch.device | None = None) -> tuple[FormalOneShotGRU, list[dict[str, float]]]:
    device = torch.device("cpu") if device is None else device
    torch.manual_seed(config.training.seed)
    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    run_dir = config.output_root / "runs" / f"formal_osla_seed{config.training.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "training_history.csv"
    train_generator = torch.Generator(device=device).manual_seed(config.training.seed)

    best_score = float("-inf")
    best_state = deepcopy(model.state_dict())
    best_update = 0
    patience_used = 0
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for update in range(1, config.training.updates + 1):
        model.train()
        rounds = sample_consistent_round_batches(
            batch_size=config.training.batch_size,
            n_rounds=config.environment.train_rounds_per_episode,
            probability_grid=config.environment.probability_grid,
            feedback_trials=config.environment.round_feedback_trials,
            generator=train_generator,
            device=device,
        )
        validate_consistent_only(rounds)
        result = rollout_episode(
            model,
            rounds,
            block_size=config.environment.block_size,
            generator=train_generator,
            deterministic=False,
            start_tokens=config.environment.start_tokens,
        )
        loss, metrics = reinforce_loss(result, entropy_bonus=config.training.entropy_bonus)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()

        row = {
            "update": float(update),
            "elapsed_seconds": float(time.perf_counter() - start_time),
            **metrics,
        }

        should_validate = config.validation.enabled and update % config.validation.interval == 0
        if should_validate:
            val = evaluate_validation(model, config, device=device)
            row.update({f"val_{key}": value for key, value in val.items()})
            score = val["score"]
            if score > best_score + config.validation.min_delta:
                best_score = score
                best_state = deepcopy(model.state_dict())
                best_update = update
                patience_used = 0
                _save_checkpoint(model, config, run_dir / "best.pt", best_update, best_score)
            else:
                patience_used += 1
            row["best_score"] = best_score
            row["best_update"] = float(best_update)
            row["patience_used"] = float(patience_used)
            if patience_used >= config.validation.patience:
                history.append(row)
                break
        elif not config.validation.enabled and update == config.training.updates:
            best_score = row["mean_return"]
            best_update = update
            best_state = deepcopy(model.state_dict())
            _save_checkpoint(model, config, run_dir / "best.pt", best_update, best_score)

        history.append(row)

    if best_update == 0:
        best_update = len(history)
        best_score = history[-1]["mean_return"] if history else float("nan")
        best_state = deepcopy(model.state_dict())
        _save_checkpoint(model, config, run_dir / "best.pt", best_update, best_score)

    model.load_state_dict(best_state)
    _save_checkpoint(model, config, run_dir / "final.pt", best_update, best_score)
    _write_history(history_path, history)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "network_id": f"formal_osla_seed{config.training.seed}",
                "model_version": MODEL_VERSION,
                "training_seed": config.training.seed,
                "best_update": best_update,
                "best_score": best_score,
                "early_stopping_enabled": config.validation.enabled,
                "lambda_anchor": config.validation.lambda_anchor,
                "regularization": {
                    "weight_decay": config.training.weight_decay,
                    "dropout": config.model.dropout,
                    "entropy_bonus": config.training.entropy_bonus,
                    "gradient_clip": config.training.gradient_clip,
                },
                "config": config.to_dict(),
                "parameter_inventory": model.parameter_inventory(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model, history


@torch.no_grad()
def evaluate_validation(model: FormalOneShotGRU, config: ExperimentConfig, *, device: torch.device) -> dict[str, float]:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.validation.seed)
    rounds = sample_consistent_round_batches(
        batch_size=config.validation.batch_size,
        n_rounds=config.validation.rounds_per_episode,
        probability_grid=config.environment.probability_grid,
        feedback_trials=config.environment.round_feedback_trials,
        generator=generator,
        device=device,
    )
    validate_consistent_only(rounds)
    result = rollout_episode(
        model,
        rounds,
        block_size=config.environment.block_size,
        generator=generator,
        deterministic=False,
        start_tokens=config.environment.start_tokens,
    )
    mean_return = float(result.total_return.mean().cpu().item())
    anchor_error = float(((result.p_boundary - result.boundary_cue_probability) ** 2).mean().cpu().item())
    score = mean_return - config.validation.lambda_anchor * anchor_error
    return {
        "mean_return": mean_return,
        "mean_bet_rate": float(result.actions.mean().cpu().item()),
        "prior_anchor_error": anchor_error,
        "score": float(score),
    }


def _save_checkpoint(
    model: FormalOneShotGRU,
    config: ExperimentConfig,
    path: Path,
    best_update: int,
    best_score: float,
) -> None:
    torch.save(
        {
            "model_version": MODEL_VERSION,
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "best_update": best_update,
            "best_score": best_score,
        },
        path,
    )


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fields: list[str] = []
    for row in history:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
