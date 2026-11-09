"""Extract M1/M2/M3 fitting subjects from the new batch: one RANDOM episode per network.

Ported from `formal_batch/analysis/extract_m3_subjects.py` with one deliberate change.

What changed, and why
---------------------
The old extractor is **deterministic**: network k (seed 20260804+k) always uses episode k
of its group (`episode_id = f"{group}_{k}"`).  The user asked for a **random** episode per
network instead, so this version draws one with a fixed, recorded seed.

The consequence is that the pipeline is no longer reproducible from the code alone -- it
depends on which episodes were drawn.  So the draw is written to
`metrics/m3_sampling_manifest.csv`, and the whole thing is reproducible from
(seed, manifest).  Without that file a rerun would silently produce a different dataset.

What did NOT change
-------------------
The output contract, column for column, so `build_rds.R` and the Stan programs work
untouched:

    subject_index, trial_in_block, choice, reward, p0

Two mappings deserve naming because getting either wrong fails silently:

* `choice` <- `trials.csv`'s **`action`**
* `reward` <- `trials.csv`'s **`outcome`**  (0/1).  NOT the `reward` column, which in this
  codebase is the model's payoff `action * (2*outcome - 1)` and takes values in {-1,0,1}.
  Feeding that into `bernoulli_logit` would be wrong in a way nothing would flag.
* `p0` <- the group's conflict-round **cue** (the stated prior): 0.3 / 0.7 / 0.3 / 0.7.

Trial count: the new protocol `conflict_only` has a single 120-trial round, against the
old `test3_extended` round 4 of 240 trials.  So `nTrials` becomes 120 and the Stan
generated-quantity arrays halve (7 x 120 x 30 instead of 7 x 240 x 30).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

GROUPS = ("mac1A", "mac1B", "mac2A", "mac2B")
CUE = {"mac1A": 0.3, "mac1B": 0.7, "mac2A": 0.3, "mac2B": 0.7}
SEED_BASE = 20260804
N_SEEDS = 30
# Fixed so the draw is reproducible from this file plus the manifest it writes.
SAMPLING_SEED = 20260930
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = PACKAGE_ROOT / "outputs" / "output_formal" / "runs"


def cell_dir(runs: Path, arm: str, protocol: str, seed: int, group: str) -> Path:
    return runs / "frozen_tests" / arm / protocol / f"formal_osla_seed{seed}" / group


def extract_one(
    runs: Path, arm: str, protocol: str, group: str, *, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[dict]]:
    """One long-table block (30 subjects x nTrials) plus the record of which episode was drawn."""
    rows: list[dict] = []
    manifest: list[dict] = []
    cue = CUE[group]
    n_trials: int | None = None

    # Draw the group's 30 episodes WITHOUT replacement, then assign one per network.
    # Two networks drawing the same episode would face an identical outcome sequence --
    # every network's frozen test seeds its generator from the same TEST_SEED, so episode
    # index fully determines the rewards.  The old extractor asserted 30 distinct sequences
    # for exactly this reason; sampling with replacement breaks that invariant.
    first = cell_dir(runs, arm, protocol, SEED_BASE, group) / "trials.csv"
    if not first.exists():
        raise FileNotFoundError(first)
    available = np.sort(pd.read_csv(first, usecols=["episode_id"])["episode_id"].unique())
    if len(available) < N_SEEDS:
        raise SystemExit(f"{group}: only {len(available)} episodes, need {N_SEEDS}")
    drawn = rng.choice(available, size=N_SEEDS, replace=False)

    for k in range(N_SEEDS):
        seed = SEED_BASE + k
        run_dir = cell_dir(runs, arm, protocol, seed, group)
        trials_path = run_dir / "trials.csv"
        if not trials_path.exists():
            raise FileNotFoundError(trials_path)
        trials = pd.read_csv(
            trials_path,
            usecols=["episode_id", "trial_id", "cue_probability", "true_probability",
                     "action", "outcome"],
        )

        episodes = np.sort(trials["episode_id"].unique())
        episode = int(drawn[k])
        block = trials[trials["episode_id"] == episode].sort_values("trial_id")

        if n_trials is None:
            n_trials = len(block)
        elif len(block) != n_trials:
            raise SystemExit(f"seed {seed}/{group}: {len(block)} trials, expected {n_trials}")

        cue_seen = block["cue_probability"].to_numpy(dtype=float)
        if not np.allclose(cue_seen, cue):
            raise SystemExit(
                f"seed {seed}/{group}: cue is {np.unique(cue_seen)}, expected {cue}"
            )

        action = block["action"].to_numpy(dtype=np.int64)
        outcome = block["outcome"].to_numpy(dtype=np.int64)
        if not np.isin(action, (0, 1)).all():
            raise SystemExit(f"seed {seed}/{group}: action outside {{0,1}}")
        if not np.isin(outcome, (0, 1)).all():
            raise SystemExit(f"seed {seed}/{group}: outcome outside {{0,1}}")

        manifest.append({
            "group": group,
            "subject_index": k,
            "network_seed": seed,
            "episode_id": episode,
            "n_episodes_available": len(episodes),
            "n_trials": len(block),
            "bet_rate": float(action.mean()),
            "true_probability": float(block["true_probability"].iloc[0]),
            "_outcome_bytes": outcome.tobytes(),
        })

        for t in range(len(block)):
            rows.append({
                "subject_index": k,
                "trial_in_block": t + 1,
                "choice": int(action[t]),
                "reward": int(outcome[t]),  # == outcome, NOT the +-1 reward column
                "p0": cue,
            })

    frame = pd.DataFrame(rows)
    if frame["subject_index"].nunique() != N_SEEDS:
        raise SystemExit(f"{group}: {frame['subject_index'].nunique()} subjects, expected {N_SEEDS}")
    # Same invariant the old extractor enforced: the 30 subjects must face 30 DIFFERENT
    # outcome sequences, or the hierarchical fit is being told it has 30 independent
    # subjects when it does not.
    distinct = {m["_outcome_bytes"] for m in manifest}
    if len(distinct) != N_SEEDS:
        raise SystemExit(
            f"{group}: only {len(distinct)} distinct outcome sequences for {N_SEEDS} subjects"
        )
    return frame, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS,
                        help="batch runs root (holding frozen_tests/)")
    parser.add_argument("--arm", default="h16_b200")
    parser.add_argument("--protocol", default="conflict_only")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "metrics")
    parser.add_argument("--sampling-seed", type=int, default=SAMPLING_SEED)
    args = parser.parse_args()

    runs = args.runs.resolve()
    if not runs.is_dir():
        raise SystemExit(f"runs root not found: {runs}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.sampling_seed)
    all_manifest: list[dict] = []
    for group in GROUPS:
        frame, manifest = extract_one(runs, args.arm, args.protocol, group, rng=rng)
        out = args.out_dir / f"formal_osla__{args.arm}__{group}_m3_subjects.csv"
        frame.to_csv(out, index=False)
        all_manifest.extend(manifest)
        print(f"{group}: {frame['subject_index'].nunique()} subjects x "
              f"{frame['trial_in_block'].max()} trials -> {out.name} "
              f"| choice mean {frame['choice'].mean():.3f} reward mean {frame['reward'].mean():.3f}")

    manifest_path = args.out_dir / "m3_sampling_manifest.csv"
    pd.DataFrame(all_manifest).drop(columns=["_outcome_bytes"]).to_csv(manifest_path, index=False)
    print(f"\nsampling manifest (seed {args.sampling_seed}) -> {manifest_path}")
    drawn = pd.DataFrame(all_manifest)
    print(f"  episodes drawn per group: {drawn.groupby('group')['episode_id'].nunique().to_dict()}"
          f"  (of {drawn['n_episodes_available'].iloc[0]} available)")


if __name__ == "__main__":
    main()
