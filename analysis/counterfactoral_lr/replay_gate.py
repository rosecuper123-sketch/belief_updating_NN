"""Bit-exact replay gate for the counterfactual one-step simulation.

Everything downstream assumes the one-step map `h_{t+1} = GRU(x_{t+1}, h_t)` can be
re-evaluated faithfully from the stored artifacts.  That assumption is checkable, and
nothing in this repository checked it: `selfcheck.py`, `run_prerun_tests.check_outputs`
and `collect_consistent_rollout.check_cell` all assert *shapes and invariants* only,
never hidden-state values.

Why this gate and not a full rollout replay
-------------------------------------------
A full replay would additionally have to reproduce `block_balanced_outcomes`'s
`randperm` order and the 120 `torch.bernoulli` draws per round (`rollout.py:107,127`).
The counterfactual does not need any of that -- it never samples -- so this gate
checks only the single-step map, exhaustively rather than on a sample.

Two assertions, applied to every within-round transition of every cell:

* **A -- row reconstruction**: the input rebuilt from `trials.csv` for trial t+1 drives
  the stored `h_t` to exactly the stored `h_{t+1}`.
* **B -- counterfactual consistency (the stronger one)**: the two counterfactual
  branches, built by flipping the *outcome* (and, consistently, the reward and token
  totals that depend on it), reproduce the stored next state when each branch is
  selected by the outcome that actually occurred:

      where(outcome_t == 1, h(cf_outcome=1), h(cf_outcome=0)) == h_{t+1}

  This is the form that matters: it validates the counterfactual *construction*, not
  just the input layout.  It matches the old codebase's `actual_err` check
  (`formal_batch/analysis/belief_decode/lr_counterfactual/counterfactual_lr.py:120-127`,
  which reported 0.0 over 71,520 transitions).

Both must hold at `max |delta| == 0` -- exact float equality on float32 tensors, not a
tolerance, because the re-step executes the same operations in the same order on the
same single thread.  The old codebase used a 1e-5 tolerance and in fact observed
exactly 0.0; demanding equality means a silent layout drift cannot hide under a
tolerance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from formal_osla.test_runner import load_checkpoint

START_TOKENS = 100.0


@dataclass(slots=True)
class GateResult:
    cell: str
    n_transitions: int
    max_row_delta: float
    max_cf_delta: float
    n_row_mismatch: int
    n_cf_mismatch: int

    @property
    def passed(self) -> bool:
        return self.max_row_delta == 0.0 and self.max_cf_delta == 0.0


def require_single_thread() -> None:
    if int(torch.get_num_threads()) != 1:
        raise RuntimeError(
            f"torch.get_num_threads() == {torch.get_num_threads()}, expected 1. "
            f"Multi-threaded reductions change the summation order and the replay "
            f"stops being bit-exact."
        )


def _batch_inputs(
    rows: pd.DataFrame, *, device: torch.device, feedback_trials: int
) -> torch.Tensor:
    """The 7-vector at the trial described by each row, for a whole batch of episodes.

    Vectorised over episodes rather than looped per episode: the original rollout
    stepped all 200 episodes of a cell in one `GRUCell` call, and a batched matrix
    multiply reduces in a different order than 200 single-row ones.  Stepping one
    episode at a time reproduces the trajectory only to float32 epsilon (~1.8e-07
    observed), which is not good enough to call the replay faithful.
    """
    batch = len(rows)
    x = torch.zeros(batch, 7, dtype=torch.float32, device=device)
    x[:, 0] = torch.tensor(rows["prev_action"].to_numpy(), dtype=torch.float32)
    x[:, 1] = torch.tensor(rows["prev_outcome"].to_numpy(), dtype=torch.float32)
    x[:, 2] = torch.tensor(rows["prev_token_delta"].to_numpy(), dtype=torch.float32)
    x[:, 3] = (
        torch.tensor(rows["current_total_tokens"].to_numpy(), dtype=torch.float32) - START_TOKENS
    ) / float(feedback_trials)
    x[:, 4] = (
        torch.tensor(rows["trial_id"].to_numpy(), dtype=torch.float32) - 1.0
    ) / float(max(feedback_trials - 1, 1))
    return x


def _batch_counterfactual_inputs(
    rows: pd.DataFrame, *, device: torch.device, feedback_trials: int, outcome: float
) -> torch.Tensor:
    """The same batch of inputs with `outcome` forced to 0 or 1, reward kept coherent.

    The reward being recomputed belongs to the *previous* trial (that is the trial whose
    outcome is being flipped), so the action is `prev_action`, not this row's `action`.
    Using the current row's action silently attaches the wrong reward to the branch --
    which is exactly the sort of error this gate exists to catch.
    """
    batch = len(rows)
    action = torch.tensor(rows["prev_action"].to_numpy(), dtype=torch.float32)
    x = torch.zeros(batch, 7, dtype=torch.float32, device=device)
    x[:, 0] = torch.tensor(rows["prev_action"].to_numpy(), dtype=torch.float32)
    x[:, 1] = outcome
    reward = action * (2.0 * outcome - 1.0)
    x[:, 2] = reward
    total_before = (
        torch.tensor(rows["current_total_tokens"].to_numpy(), dtype=torch.float32)
        - torch.tensor(rows["prev_token_delta"].to_numpy(), dtype=torch.float32)
    )
    x[:, 3] = (total_before + reward - START_TOKENS) / float(feedback_trials)
    x[:, 4] = (
        torch.tensor(rows["trial_id"].to_numpy(), dtype=torch.float32) - 1.0
    ) / float(max(feedback_trials - 1, 1))
    return x


def check_cell(
    model,
    hidden: np.ndarray,
    trials: pd.DataFrame,
    *,
    cell: str,
    device: torch.device,
    feedback_trials: int = 120,
) -> GateResult:
    """Run both assertions over every within-round transition of one cell."""
    n_episodes, n_trials, _ = hidden.shape
    ordered = trials.sort_values(["episode_id", "round_id", "trial_id"]).reset_index(drop=True)
    if len(ordered) != n_episodes * n_trials:
        raise ValueError(f"{cell}: {len(ordered)} rows vs hidden {hidden.shape}")

    # All episodes share the round layout, so the within-round mask is the same for
    # every episode; derive it from episode 0 and reuse it as a column mask.
    round_ids = (
        ordered["round_id"].to_numpy().reshape(n_episodes, n_trials)
    )
    within_round = round_ids[:, :-1] == round_ids[:, 1:]
    if not within_round.any():
        raise ValueError(f"{cell}: no within-round transitions")
    transitions = np.flatnonzero(within_round[0])

    state = torch.tensor(hidden, dtype=torch.float32, device=device)
    max_row = max_cf = 0.0
    n_row_bad = n_cf_bad = 0

    with torch.no_grad():
        for i in transitions:
            # `i` indexes the pair (trial i, trial i+1) 0-based, so the row whose
            # `prev_*` fields describe trial i+1 is comparison_rows.
            rows = ordered[ordered["trial_id"] == int(i) + 2]
            if len(rows) != n_episodes:
                raise ValueError(f"{cell}: trial {i + 2} has {len(rows)} rows, expected {n_episodes}")
            h_t = state[:, i, :]
            target = state[:, i + 1, :]
            outcome = torch.tensor(rows["outcome"].to_numpy(), dtype=torch.float32)

            h_row, _, _ = model.step(
                _batch_inputs(rows, device=device, feedback_trials=feedback_trials), h_t
            )
            delta_row = float((h_row - target).abs().max())
            max_row = max(max_row, delta_row)
            n_row_bad += int(delta_row != 0.0)

            # B -- the two branches, each selected by the outcome that actually occurred.
            # `prev_outcome` at trial i+1 is the outcome of trial i, which is also the
            # quantity being flipped, so selection uses the trial-i outcome.
            prev_outcome = torch.tensor(rows["prev_outcome"].to_numpy(), dtype=torch.float32)
            pos_rows = rows.assign(prev_outcome=1.0)
            neg_rows = rows.assign(prev_outcome=0.0)
            h_pos, _, _ = model.step(
                _batch_counterfactual_inputs(pos_rows, device=device,
                                             feedback_trials=feedback_trials, outcome=1.0),
                h_t,
            )
            h_neg, _, _ = model.step(
                _batch_counterfactual_inputs(neg_rows, device=device,
                                             feedback_trials=feedback_trials, outcome=0.0),
                h_t,
            )
            selector = (prev_outcome == 1.0).unsqueeze(1)
            h_cf = torch.where(selector, h_pos, h_neg)
            delta_cf = float((h_cf - target).abs().max())
            max_cf = max(max_cf, delta_cf)
            n_cf_bad += int(delta_cf != 0.0)

    return GateResult(cell, int(transitions.size) * n_episodes, max_row, max_cf,
                      n_row_bad, n_cf_bad)


def run_gate(
    checkpoint_path: Path, run_dir: Path, *, device: torch.device | None = None
) -> GateResult:
    """Load one checkpoint + one cell and gate it."""
    device = torch.device("cpu") if device is None else device
    require_single_thread()
    model, _config, _checkpoint = load_checkpoint(checkpoint_path, device=device)
    with np.load(run_dir / "hidden.npz") as data:
        hidden = np.asarray(data["hidden"], dtype=np.float32)
    trials = pd.read_csv(run_dir / "trials.csv")
    # Trials per ROUND, not per episode: the input's `trial_position` is normalised by
    # the round length.  Read it from the data rather than assuming 120.
    feedback_trials = int(trials["trial_id"].max())
    return check_cell(
        model, hidden, trials, cell=run_dir.parent.name, device=device,
        feedback_trials=feedback_trials,
    )


def reverse_selfcheck(checkpoint_path: Path, run_dir: Path, *, device: torch.device | None = None) -> bool:
    """Corrupt one input field and confirm the gate rejects it.

    Without this, a gate that always passes would look identical to a correct one.  The
    corruption is deliberate and local: `prev_outcome` is flipped, which changes the
    reward the branch computes, so assertion B must fail.  Returns True if the gate
    failed as it should.
    """
    device = torch.device("cpu") if device is None else device
    model, _config, _checkpoint = load_checkpoint(checkpoint_path, device=device)
    with np.load(run_dir / "hidden.npz") as data:
        hidden = np.asarray(data["hidden"], dtype=np.float32)
    trials = pd.read_csv(run_dir / "trials.csv")
    feedback_trials = int(trials["trial_id"].max())

    corrupted = trials.copy()
    corrupted["prev_outcome"] = 1.0 - corrupted["prev_outcome"]
    result = check_cell(
        model, hidden, corrupted, cell="corrupted", device=device,
        feedback_trials=feedback_trials,
    )
    return not result.passed


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, default=None)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--protocol", default="conflict_only")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=list(range(20260804, 20260834)))
    parser.add_argument("--groups", nargs="+", default=["mac1A", "mac1B", "mac2A", "mac2B"])
    parser.add_argument("--skip-selfcheck", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    arm = f"h{args.hidden_size:02d}_b{args.budget}"
    require_single_thread()

    rows = []
    for seed in args.seeds:
        checkpoint = out_root / "arms" / arm / "runs" / f"formal_osla_seed{seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model, _config, _ck = load_checkpoint(checkpoint, device=torch.device("cpu"))
        for group in args.groups:
            run_dir = out_root / "frozen_tests" / arm / args.protocol / f"formal_osla_seed{seed}" / group
            if not (run_dir / "trials.csv").exists():
                raise FileNotFoundError(run_dir)
            with np.load(run_dir / "hidden.npz") as data:
                hidden = np.asarray(data["hidden"], dtype=np.float32)
            trials = pd.read_csv(run_dir / "trials.csv")
            result = check_cell(
                model, hidden, trials, cell=f"seed{seed}/{group}",
                device=torch.device("cpu"), feedback_trials=int(trials["trial_id"].max()),
            )
            rows.append(result)
        print(f"  seed {seed}: {len(rows)} cells gated", flush=True)

    total = sum(r.n_transitions for r in rows)
    worst_row = max(r.max_row_delta for r in rows)
    worst_cf = max(r.max_cf_delta for r in rows)
    passed = all(r.passed for r in rows)
    print(f"\ngated {len(rows)} cells, {total} transitions")
    print(f"  max |row delta| = {worst_row:.3e}")
    print(f"  max |cf  delta| = {worst_cf:.3e}")
    print(f"  cells passing   = {sum(r.passed for r in rows)}/{len(rows)}")

    if not args.skip_selfcheck:
        seed = args.seeds[0]
        ok = reverse_selfcheck(
            out_root / "arms" / arm / "runs" / f"formal_osla_seed{seed}" / "best.pt",
            out_root / "frozen_tests" / arm / args.protocol / f"formal_osla_seed{seed}" / args.groups[0],
        )
        print(f"  reverse self-check (corrupted input must FAIL): "
              f"{'OK' if ok else 'BROKEN -- the gate accepts a corrupted input'}")
        if not ok:
            raise SystemExit(1)

    if not passed:
        raise SystemExit("replay gate FAILED -- do not interpret downstream results")


if __name__ == "__main__":
    main()
