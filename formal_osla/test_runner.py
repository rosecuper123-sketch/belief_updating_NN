from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import ExperimentConfig
from .export import write_rollout_outputs
from .model import FormalOneShotGRU, MODEL_VERSION
from .protocols import FORMAL_GROUPS, expand_round_specs_to_batch, formal_test_rounds
from .rollout import rollout_episode


def load_checkpoint(path: Path, *, device: torch.device) -> tuple[FormalOneShotGRU, ExperimentConfig, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint["model_version"] != MODEL_VERSION:
        raise ValueError(f"checkpoint version {checkpoint['model_version']} != {MODEL_VERSION}")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = FormalOneShotGRU(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config, checkpoint


@torch.no_grad()
def run_frozen_tests(
    checkpoint_path: Path,
    *,
    out_root: Path,
    protocol: str = "human_default",
    groups: tuple[str, ...] = FORMAL_GROUPS,
    episodes: int = 200,
    seed: int = 20261107,
    deterministic: bool = False,
    custom_rounds: list[tuple[str, float, float]] | None = None,
    device: torch.device | None = None,
) -> list[dict[str, object]]:
    device = torch.device("cpu") if device is None else device
    model, config, checkpoint = load_checkpoint(checkpoint_path, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    network_id = checkpoint_path.parent.name
    manifest: list[dict[str, object]] = []

    for group in groups:
        round_specs = formal_test_rounds(
            group=group,
            protocol=protocol,
            feedback_trials=config.environment.round_feedback_trials,
            custom=custom_rounds,
        )
        rounds = expand_round_specs_to_batch(round_specs, episodes=episodes, device=device)
        result = rollout_episode(
            model,
            rounds,
            block_size=config.environment.block_size,
            generator=generator,
            deterministic=deterministic,
            start_tokens=config.environment.start_tokens,
        )
        target = out_root / protocol / network_id / group
        write_rollout_outputs(
            result,
            out_dir=target,
            network_id=network_id,
            seed=seed,
            groups=tuple(group for _ in range(episodes)),
            protocol=protocol,
        )
        manifest.append(
            {
                "network_id": network_id,
                "group": group,
                "protocol": protocol,
                "episodes": episodes,
                "rounds": len(round_specs),
                "behavioral_trials_per_episode": int(result.actions.shape[1]),
                "boundary_events_per_episode": int(result.p_boundary.shape[1]),
                "output_dir": str(target),
            }
        )

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_best_update": checkpoint.get("best_update"),
                "seed": seed,
                "deterministic": deterministic,
                "entries": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest
