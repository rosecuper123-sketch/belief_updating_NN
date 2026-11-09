from __future__ import annotations

import csv
import tempfile
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from formal_osla.config import ExperimentConfig  # noqa: E402
from formal_osla.export import write_rollout_outputs  # noqa: E402
from formal_osla.model import FormalOneShotGRU  # noqa: E402
from formal_osla.protocols import (  # noqa: E402
    expand_round_specs_to_batch,
    formal_test_rounds,
    sample_consistent_round_batches,
    validate_consistent_only,
)
from formal_osla.rollout import rollout_episode  # noqa: E402
from formal_osla.train import train_formal  # noqa: E402


def assert_rollout_invariants() -> None:
    device = torch.device("cpu")
    model = FormalOneShotGRU(hidden_size=8)
    gen = torch.Generator().manual_seed(11)
    rounds = formal_test_rounds(group="mac1A", protocol="human_default")
    batches = expand_round_specs_to_batch(rounds, episodes=3, device=device)
    result = rollout_episode(model, batches, block_size=20, generator=gen)

    assert result.actions.shape == (3, 240)
    assert result.hidden.shape == (3, 240, 8)
    assert result.h_pre_boundary.shape == (3, 2, 8)
    assert result.h_post_boundary.shape == (3, 2, 8)
    assert result.p_boundary.shape == (3, 2)
    assert torch.all(result.trial_id[:, 0] == 1)
    assert torch.all(result.trial_id[:, 120] == 1)
    for idx in (0, 120):
        assert torch.all(result.prev_action[:, idx] == 0)
        assert torch.all(result.prev_outcome[:, idx] == 0)
        assert torch.all(result.prev_token_delta[:, idx] == 0)
        assert torch.all(result.current_total_tokens[:, idx] == 100)
        assert torch.all(result.trial_inputs[:, idx, 5] == 0)
        assert torch.all(result.trial_inputs[:, idx, 6] == 0)
    assert torch.all(result.boundary_mismatch_flag[:, 0] == 0)
    assert torch.all(result.boundary_mismatch_flag[:, 1] == 1)


def assert_training_consistent_only() -> None:
    device = torch.device("cpu")
    gen = torch.Generator().manual_seed(12)
    rounds = sample_consistent_round_batches(
        batch_size=16,
        n_rounds=3,
        probability_grid=(0.3, 0.4, 0.5, 0.6, 0.7),
        feedback_trials=120,
        generator=gen,
        device=device,
    )
    validate_consistent_only(rounds)
    assert all(not torch.any(r.mismatch_flag).item() for r in rounds)


def assert_export_separation() -> None:
    device = torch.device("cpu")
    model = FormalOneShotGRU(hidden_size=8)
    gen = torch.Generator().manual_seed(13)
    rounds = expand_round_specs_to_batch(formal_test_rounds(group="mac1B"), episodes=2, device=device)
    result = rollout_episode(model, rounds, block_size=20, generator=gen)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write_rollout_outputs(
            result,
            out_dir=out,
            network_id="selfcheck",
            seed=13,
            groups=("mac1B", "mac1B"),
            protocol="human_default",
        )
        with (out / "trials.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with (out / "boundary_events.csv").open(newline="", encoding="utf-8") as handle:
            boundary_rows = list(csv.DictReader(handle))
        assert len(rows) == 2 * 2 * 120
        assert len(boundary_rows) == 2 * 2
        assert all("boundary_flag" not in row for row in rows)
        with np.load(out / "hidden.npz") as hidden:
            assert hidden["hidden"].shape == (2, 240, 8)
        with np.load(out / "boundary_hidden.npz") as boundary:
            assert boundary["h_pre_boundary"].shape == (2, 2, 8)


def assert_tiny_training() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = ExperimentConfig(output_root=Path(tmp))
        config.model.hidden_size = 8
        config.training.seed = 123
        config.training.updates = 2
        config.training.batch_size = 4
        config.validation.enabled = True
        config.validation.interval = 1
        config.validation.batch_size = 4
        config.validation.patience = 5
        _model, history = train_formal(config)
        assert len(history) == 2
        assert (Path(tmp) / "runs" / "formal_osla_seed123" / "best.pt").exists()
        assert (Path(tmp) / "runs" / "formal_osla_seed123" / "training_history.csv").exists()


def main() -> None:
    assert_rollout_invariants()
    assert_training_consistent_only()
    assert_export_separation()
    assert_tiny_training()
    print("selfcheck passed: boundary/trial separation, consistent-only training, exports, and tiny training are valid.")


if __name__ == "__main__":
    main()
