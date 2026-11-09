"""Consistent-round rollouts, collected purely as belief-decoder calibration data.

Why this exists
---------------
The belief decoder is a linear readout `h_t -> P(true)`.  It can only be *fit*
where the label is unambiguous, i.e. in **consistent rounds** (cue == true), where
"what the network ought to believe" is exactly the stated/true probability.  The
formal batch's behavioral protocol is `conflict_only`, whose single round is a
conflict round with a per-group constant truth -- so it carries no usable label and
the decoder cannot be fit on it (see the batch README §11.2).

This script therefore generates the calibration set **separately from the behavioral
test**, so that `conflict_only` remains the only behavioral protocol:

* one consistent round per cell, `custom_rounds=[("consistent", cue, cue)]`;
* cues 0.1 .. 0.9, i.e. the training grid {0.3..0.7} plus four out-of-distribution
  values, matching the orthogonal-grid precedent;
* every cell starts from a **blank hidden state**, so cells are independent;
* the environment is held fixed across networks -- all networks see the same
  `TEST_SEED`, and each cue is its own cell, so no network's data is contaminated by
  another's.

Feeding the decoder: pool all cue cells of one network, using
`(cue_index, episode_index)` as the grouping key.  Grouping by episode alone would
leak, because episode k of cue 0.3 and episode k of cue 0.7 are different draws but
would land on the same side of a naive split.

This writes only new outputs under `--out-root`; nothing in `formal_osla/` changes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Pin threads before numpy/torch are imported (see prerun_batch.py).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from formal_osla.test_runner import run_frozen_tests  # noqa: E402

CUES = tuple(round(0.1 * i, 1) for i in range(1, 10))
TRIALS_PER_ROUND = 120
TEST_SEED = 20261107
# In `custom` mode a group's preset values are ignored; the key only has to be a valid
# group name, exactly as `test_orthogonal_prior_grid.py` does.
LABEL_GROUP = "mac1A"
DEFAULT_EPISODES = 50
DEFAULT_SEEDS = tuple(range(20260804, 20260834))


def require_single_thread() -> None:
    n = int(torch.get_num_threads())
    if n != 1:
        raise RuntimeError(f"torch.get_num_threads() == {n}, expected 1")


def cell_dir(out_root: Path, cue: float, hidden: int, budget: int, seed: int) -> Path:
    return (
        out_root / "consistent_rollout" / f"h{hidden:02d}_b{budget}" / f"cue{cue:.1f}" / f"h{hidden:02d}_seed{seed}"
        / "custom" / f"formal_osla_seed{seed}" / LABEL_GROUP
    )


def checkpoint_path(out_root: Path, hidden: int, budget: int, seed: int) -> Path:
    return out_root / "arms" / f"h{hidden:02d}_b{budget}" / "runs" / f"formal_osla_seed{seed}" / "best.pt"


def run_cell(out_root: Path, cue: float, hidden: int, budget: int, seed: int, episodes: int) -> Path:
    target = cell_dir(out_root, cue, hidden, budget, seed)
    if (target / "trials.csv").exists():
        return target
    checkpoint = checkpoint_path(out_root, hidden, budget, seed)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    run_frozen_tests(
        checkpoint,
        # target = <out_root_arg>/custom/<network_id>/<group>, so parents[2] is <out_root_arg>
        out_root=target.parents[2],
        protocol="custom",
        groups=(LABEL_GROUP,),
        episodes=episodes,
        seed=TEST_SEED,
        custom_rounds=[("consistent", float(cue), float(cue))],
    )
    return target


def check_cell(target: Path, *, episodes: int, cue: float, hidden: int) -> dict:
    """Structural + consistency invariants for one consistent-round cell."""
    trials = pd.read_csv(target / "trials.csv")
    if len(trials) != episodes * TRIALS_PER_ROUND:
        raise SystemExit(f"{target}: {len(trials)} rows, expected {episodes * TRIALS_PER_ROUND}")
    if trials["round_id"].nunique() != 1:
        raise SystemExit(f"{target}: expected exactly one round")
    # A consistent round is exactly one where cue == true on every trial.
    if not np.allclose(trials["cue_probability"], trials["true_probability"]):
        raise SystemExit(f"{target}: cue != true on some trial -- not a consistent round")
    if not np.allclose(trials["cue_probability"], cue):
        raise SystemExit(f"{target}: cue {trials['cue_probability'].unique()} != requested {cue}")

    with np.load(target / "hidden.npz") as data:
        shape = data["hidden"].shape
    expected = (episodes, TRIALS_PER_ROUND, hidden)
    if tuple(shape) != expected:
        raise SystemExit(f"{target}: hidden {tuple(shape)} != {expected}")

    # Trial 1 must depend on the cue alone: identical across episodes up to action noise,
    # and the state must start from blank (the boundary step is the only pre-trial input).
    first = trials[trials["trial_id"] == 1]
    clean = (
        (first["prev_action"] == 0).all()
        and (first["prev_outcome"] == 0).all()
        and (first["prev_token_delta"] == 0).all()
        and (first["current_total_tokens"] == 100).all()
    )
    if not clean:
        raise SystemExit(f"{target}: trial 1 did not start from a clean slate")
    return {"rows": len(trials), "hidden_shape": tuple(int(v) for v in shape)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True,
                        help="arm root holding arms/h{hidden}_b{budget}/runs/...")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    require_single_thread()
    out_root = args.out_root.resolve()
    total = len(args.seeds) * len(CUES)
    print(f"consistent rollout: {len(args.seeds)} networks x {len(CUES)} cues = {total} cells")
    print(f"hidden={args.hidden_size} budget={args.budget} episodes={args.episodes} "
          f"TEST_SEED={TEST_SEED}\nout_root: {out_root}\n")

    started = time.perf_counter()
    done = 0
    for seed in args.seeds:
        for cue in CUES:
            target = cell_dir(out_root, cue, args.hidden_size, args.budget, seed)
            if args.skip_existing and (target / "trials.csv").exists():
                done += 1
                continue
            run_cell(out_root, cue, args.hidden_size, args.budget, seed, args.episodes)
            check_cell(target, episodes=args.episodes, cue=cue, hidden=args.hidden_size)
            done += 1
            if done % len(CUES) == 0:
                elapsed = time.perf_counter() - started
                print(f"  {done}/{total} cells ({elapsed:.0f}s, "
                      f"{elapsed / done:.1f}s/cell)", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\ndone: {done}/{total} cells in {elapsed:.0f}s")
    if done != total:
        raise SystemExit(f"only {done} of {total} cells present")

    # Provenance: which checkpoint produced each cell, with the file hash.
    import hashlib
    records = []
    for seed in args.seeds:
        checkpoint = checkpoint_path(out_root, args.hidden_size, args.budget, seed)
        records.append({
            "seed": seed,
            "network_id": checkpoint.parent.name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        })
    manifest_path = out_root / "consistent_rollout" / "manifest_provenance.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(
        {"test_seed": TEST_SEED, "cues": list(CUES), "episodes": args.episodes,
         "cells_per_network": len(CUES), "entries": records}, indent=2), encoding="utf-8")
    print(f"provenance -> {manifest_path}")


if __name__ == "__main__":
    main()
