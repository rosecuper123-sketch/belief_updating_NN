"""Loaders for the new-format frozen-test artifacts (`formal_osla`).

Layout written by `formal_osla/test_runner.py` + `formal_osla/export.py`:

    <frozen_root>/<arm>/<protocol>/<network_id>/<group>/
        trials.csv          21 columns, one row per behavioral trial
        boundary_events.csv one row per boundary step (one per round)
        hidden.npz          hidden (E, rounds*120, H), prediction_error/actions/
                            outcomes/rewards (E, rounds*120), episode_ids, groups, protocol
        boundary_hidden.npz h_pre/h_post/delta_h_boundary (E, rounds, H), p_boundary, ...
        manifest.json       checkpoint path + best_update for this call

This replaces the old codebase's split of `rounds.csv` (round-level values repeated
over their trial span) + `hidden.npz`.  The new `trials.csv` already carries
`cue_probability`/`true_probability`/`mismatch_flag` per trial, so no round-span
repetition is needed -- but the round structure is still recoverable from
`(episode_id, round_id)`.

Nothing here hard-codes a hidden size: read `hidden.shape[2]`.  The old code had
literal `32` in several reshape calls, which is exactly what breaks on h=16.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

GROUPS: tuple[str, ...] = ("mac1A", "mac1B", "mac2A", "mac2B")
TRIALS_PER_ROUND = 120
TRAIN_FRACTION = 0.7  # episode-level split, matches the old codebase's 140/60 of 200

# Round-level cue/true per group for the formal paradigm (CORE_METHODOLOGY §2.1/§2.2).
# Only used for sanity assertions, never for fitting.
FORMAL_ROUNDS: dict[str, tuple[tuple[str, float, float], ...]] = {
    "mac1A": (("consistent", 0.70, 0.70), ("conflict", 0.30, 0.60)),
    "mac1B": (("consistent", 0.30, 0.30), ("conflict", 0.70, 0.40)),
    "mac2A": (("consistent", 0.70, 0.70), ("conflict", 0.30, 0.50)),
    "mac2B": (("consistent", 0.30, 0.30), ("conflict", 0.70, 0.50)),
}


@dataclass(slots=True)
class Cell:
    """One (network, protocol, group) frozen-test cell."""

    network_id: str
    group: str
    protocol: str
    run_dir: Path
    hidden: np.ndarray  # (episodes, trials, hidden_size)
    trials: pd.DataFrame  # one row per behavioral trial, episode-major

    @property
    def hidden_size(self) -> int:
        return int(self.hidden.shape[2])

    @property
    def n_episodes(self) -> int:
        return int(self.hidden.shape[0])

    @property
    def n_trials(self) -> int:
        return int(self.hidden.shape[1])

    @property
    def n_rounds(self) -> int:
        return self.n_trials // TRIALS_PER_ROUND

    def trial_target(self, column: str) -> np.ndarray:
        """A `trials.csv` column as an (episodes, trials) array, shape-checked.

        The row order of `trials.csv` is episode-major with `trial_id` restarting
        each round, so a plain reshape is only valid after sorting on
        `(episode_id, round_id, trial_id)`.
        """
        if column not in self.trials.columns:
            raise KeyError(f"{column!r} not in trials.csv: {sorted(self.trials.columns)}")
        ordered = self.trials.sort_values(["episode_id", "round_id", "trial_id"])
        values = ordered[column].to_numpy(dtype=float)
        if values.size != self.n_episodes * self.n_trials:
            raise ValueError(
                f"{column}: {values.size} values vs hidden {self.hidden.shape}"
            )
        return values.reshape(self.n_episodes, self.n_trials)

    def round_mask(self, round_id: int) -> np.ndarray:
        """Boolean (n_trials,) mask selecting one round's trial block."""
        block = np.zeros(self.n_trials, dtype=bool)
        block[(round_id - 1) * TRIALS_PER_ROUND : round_id * TRIALS_PER_ROUND] = True
        return block

    def consistent_block(self) -> np.ndarray:
        """Trial block spanning every round whose cue equals its true probability.

        In `conflict_only` this is empty -- the only round is a conflict round.
        That is exactly the constraint that forces the belief probe to read its
        training data from a separate consistent-round rollout.
        """
        ordered = self.trials.sort_values(["episode_id", "round_id", "trial_id"])
        cue = ordered["cue_probability"].to_numpy(dtype=float)
        true = ordered["true_probability"].to_numpy(dtype=float)
        same = np.isclose(cue, true)
        mask = np.zeros(self.n_trials, dtype=bool)
        # A round counts as consistent only if every one of its trials agrees.
        per_round = same.reshape(self.n_episodes, self.n_rounds, TRIALS_PER_ROUND).all(axis=(0, 2))
        for round_index, is_consistent in enumerate(per_round, start=1):
            if is_consistent:
                mask |= self.round_mask(round_index)
        return mask


def load_cell(run_dir: Path) -> Cell:
    """Load one cell directory. Raises if the structural invariants do not hold."""
    hidden_path = run_dir / "hidden.npz"
    trials_path = run_dir / "trials.csv"
    if not hidden_path.exists() or not trials_path.exists():
        raise FileNotFoundError(f"need hidden.npz and trials.csv in {run_dir}")

    with np.load(hidden_path) as data:
        hidden = np.asarray(data["hidden"], dtype=np.float32)
        episode_ids = np.asarray(data["episode_ids"])
        protocol = str(np.asarray(data["protocol"]).item())

    trials = pd.read_csv(trials_path)
    n_episodes, n_trials, _ = hidden.shape
    if len(trials) != n_episodes * n_trials:
        raise ValueError(
            f"{run_dir}: trials.csv has {len(trials)} rows, hidden has "
            f"{n_episodes}x{n_trials}"
        )
    if len(np.unique(episode_ids)) != n_episodes:
        raise ValueError(f"{run_dir}: hidden.npz episodes are not distinct")

    return Cell(
        network_id=run_dir.parents[1].name,
        group=run_dir.name,
        protocol=protocol,
        run_dir=run_dir,
        hidden=hidden,
        trials=trials,
    )


def iter_cells(
    frozen_root: Path,
    protocol: str,
    *,
    arm: str | None = None,
    groups: Sequence[str] = GROUPS,
    networks: Sequence[str] | None = None,
) -> Iterator[Cell]:
    """Walk `<frozen_root>/<arm>/<protocol>/<network_id>/<group>/`.

    Yields in a deterministic order so downstream aggregation is reproducible.
    Networks with no output directory are skipped silently; `networks`, if given,
    also asserts that every requested network was found.
    """
    roots = sorted(frozen_root.glob("*")) if arm is None else [frozen_root / arm]
    seen: set[str] = set()
    for arm_dir in roots:
        protocol_dir = arm_dir / protocol
        if not protocol_dir.is_dir():
            continue
        for network_dir in sorted(protocol_dir.glob("formal_osla_seed*")):
            if networks is not None and network_dir.name not in networks:
                continue
            for group in groups:
                run_dir = network_dir / group
                if not (run_dir / "trials.csv").exists():
                    continue
                seen.add(network_dir.name)
                yield load_cell(run_dir)
    if networks is not None:
        missing = sorted(set(networks) - seen)
        if missing:
            raise FileNotFoundError(f"{len(missing)} networks missing under {protocol}: {missing[:5]}")


def episode_split(n_episodes: int, *, train_fraction: float = TRAIN_FRACTION) -> np.ndarray:
    """Deterministic episode-level train mask: first 70% train, last 30% test.

    Episode-level, never trial-level: trials within an episode are strongly
    dependent, so a trial-level split leaks.  This mirrors the old codebase's
    fixed 140/60 split of 200 episodes and keeps networks comparable.
    """
    n_train = int(round(n_episodes * train_fraction))
    mask = np.zeros(n_episodes, dtype=bool)
    mask[:n_train] = True
    return mask
