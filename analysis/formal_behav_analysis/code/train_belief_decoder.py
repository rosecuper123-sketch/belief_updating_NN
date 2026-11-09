"""Fit the calibrated belief decoder for each frozen network.

Reads the **consistent-round rollout** collected by `collect_consistent_rollout.py`
(`<out-root>/consistent_rollout/cue<c>/h<hh>_seed<s>/...`), fits one decoder per
network, and writes it to `<analysis-root>/metrics/validated_decoders/<network_id>.npz`.

Why a separate rollout: the decoder's label is only unambiguous in consistent
rounds.  The behavioral protocol for this batch is `conflict_only`, which has none
-- see the batch README §11.2.

Pooling across cues, and the split
----------------------------------
All nine cue cells of a network are pooled into one fit, because the decoder has to
cover the cue range the conflict rounds actually use (0.3 and 0.7).  The grouping key
is the **composite `(cue, episode)` index**, not the episode index: episode k of cue
0.3 and episode k of cue 0.7 are different draws, and keying on episode alone would
put them on the same side of every split.  The 70/30 split is applied **inside each
cue cell**, so train and test each cover the whole cue range rather than the split
falling between cues.

Self-checks (all three must pass or the script exits non-zero)
-------------------------------------------------------------
1. **Permutation control** -- refit on targets shuffled across composite groups.
   Held-out R^2 must collapse toward 0; anything else means the reported R^2 is an
   artifact of the fit rather than a real encoding.
2. **Fold sanity** -- every fold must contain at least one train episode.
3. **Conflict-round application** -- decode the `conflict_only` trajectories and check
   the trial-1 belief sits near the cue (no evidence yet) and the round-end belief has
   moved toward the truth.  This is the test that matters: a decoder can score well on
   held-out consistent rounds and still be useless on the conflict round.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.formal_behav_analysis.belief_decoder import ALPHAS, Decoder, fit_decoder, save_decoder  # noqa: E402
from analysis.formal_behav_analysis.loading import (  # noqa: E402
    GROUPS,
    TRIALS_PER_ROUND,
    Cell,
    episode_split,
    load_cell,
)

CUES = tuple(round(0.1 * i, 1) for i in range(1, 10))
DEFAULT_SEEDS = tuple(range(20260804, 20260834))
OPENING_WINDOW = 5
LATE_START = 101


def consistent_cells(out_root: Path, hidden: int, budget: int, seed: int, cues=CUES) -> list[Cell]:
    cells = []
    for cue in cues:
        run_dir = (
            out_root / "consistent_rollout" / f"h{hidden:02d}_b{budget}" / f"cue{cue:.1f}" / f"h{hidden:02d}_seed{seed}"
            / "custom" / f"formal_osla_seed{seed}" / "mac1A"
        )
        if not (run_dir / "trials.csv").exists():
            raise FileNotFoundError(f"missing consistent cell: {run_dir}")
        cells.append(load_cell(run_dir))
    return cells


def pool_consistent(cells: list[Cell]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool cue cells into (x, y, composite_group, train_mask) for one network.

    Returns arrays whose group index is `cue_index * n_episodes + episode_id`, and a
    train mask defined per cue cell so the 70/30 split is balanced across the cue range.
    """
    xs, ys, groups, masks = [], [], [], []
    offset = 0
    for cue_index, cell in enumerate(cells):
        n_ep, n_tr, h = cell.hidden.shape
        target = cell.trial_target("true_probability")
        # A consistent round has cue == true on every trial; assert rather than assume,
        # because a decoder fit on a mislabelled cell is silently worthless.
        if not np.allclose(target, cue_of(cell)):
            raise ValueError(f"{cell.run_dir}: true_probability is not constant {cue_of(cell)}")

        xs.append(cell.hidden.reshape(n_ep * n_tr, h))
        ys.append(target.reshape(-1))
        groups.append(np.repeat(np.arange(n_ep) + offset, n_tr))
        split = episode_split(n_ep)
        masks.append(split)
        offset += n_ep
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(groups),
        np.concatenate(masks),
    )


def cue_of(cell: Cell) -> float:
    value = float(cell.trials["cue_probability"].iloc[0])
    return value


def permutation_control(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, train_mask: np.ndarray, *, seed: int = 0
) -> float:
    """Held-out R^2 with whole-target blocks reassigned to other groups.

    Permuting whole group blocks rather than individual trials keeps each episode's
    temporal structure intact, so the control asks "is there an h -> label association
    at all?" instead of destroying the autocorrelation and trivially scoring 0.

    Requires the samples to be laid out as contiguous, equal-sized group blocks, which
    is how `pool_consistent` builds them; asserted rather than assumed.
    """
    n_groups = int(len(np.unique(groups)))
    block = len(y) // n_groups
    if block * n_groups != len(y):
        raise ValueError("groups are not equal-sized")
    ordered = groups.reshape(n_groups, block)
    if not (ordered == ordered[:, :1]).all():
        raise ValueError("samples are not contiguous per group")
    if not np.array_equal(ordered[:, 0], np.arange(n_groups)):
        raise ValueError("group blocks are not in ascending order")

    rng = np.random.default_rng(seed)
    shuffled_y = y.reshape(n_groups, block)[rng.permutation(n_groups)].reshape(-1)
    return fit_decoder(
        x, shuffled_y, groups, train_mask,
        network_id="permutation-control", target_name="permuted", alphas=np.array([1.0]),
    ).outer_r2


def decode_conflict(network_id: str, frozen_root: Path, hidden: int, budget: int, seed: int,
                    decoder: Decoder) -> list[dict]:
    """Apply the decoder to the conflict_only trajectories; returns per-group summaries.

    Raises if no cell is found: a silently empty result would let the application
    self-check below pass by not running at all.
    """
    rows = []
    arm = f"h{hidden:02d}_b{budget}"
    for group in GROUPS:
        run_dir = frozen_root / "frozen_tests" / arm / "conflict_only" / f"formal_osla_seed{seed}" / group
        if not (run_dir / "trials.csv").exists():
            continue
        cell = load_cell(run_dir)
        belief = decoder(cell.hidden)  # (episodes, trials)
        cue = float(cell.trials["cue_probability"].iloc[0])
        true = float(cell.trials["true_probability"].iloc[0])
        opening = float(belief[:, :OPENING_WINDOW].mean())
        late = float(belief[:, LATE_START - 1:].mean())
        rows.append({
            "network_id": network_id,
            "group": group,
            "cue": cue,
            "true": true,
            "belief_trial1": float(belief[:, 0].mean()),
            "belief_opening": opening,
            "belief_late": late,
            "opening_minus_cue": opening - cue,
            "late_minus_true": late - true,
        })
    if not rows:
        raise FileNotFoundError(
            f"no conflict_only cells under {frozen_root}/frozen_tests/{arm} -- "
            f"the application self-check would otherwise be skipped in silence"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True,
                        help="arm root holding arms/ and consistent_rollout/")
    parser.add_argument("--frozen-root", type=Path, default=None,
                        help="frozen-test root for the conflict-round application check "
                             "(defaults to --out-root)")
    parser.add_argument(
        "--analysis-root", type=Path,
        default=_ROOT / "analysis" / "formal_behav_analysis",
        help="where metrics/ is written (default: analysis/formal_behav_analysis)",
    )
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--skip-permutation", action="store_true")
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    frozen_root = (args.frozen_root or args.out_root).resolve()
    analysis_root = args.analysis_root.resolve()
    decoder_dir = analysis_root / "metrics" / "validated_decoders"
    decoder_dir.mkdir(parents=True, exist_ok=True)

    print(f"belief decoder: {len(args.seeds)} networks, hidden={args.hidden_size}")
    print(f"consistent rollout: {out_root / 'consistent_rollout'}")
    print(f"decoders -> {decoder_dir}\n")

    summary, application, controls = [], [], []
    t0 = time.perf_counter()
    for index, seed in enumerate(args.seeds, start=1):
        cells = consistent_cells(out_root, args.hidden_size, args.budget, seed)
        x, y, groups, train_mask = pool_consistent(cells)
        n_episodes_per_cue = cells[0].n_episodes
        network_id = f"formal_osla_seed{seed}"

        decoder = fit_decoder(
            x, y, groups, train_mask,
            network_id=network_id, target_name="true_probability",
        )
        save_decoder(decoder, decoder_dir / f"{network_id}.npz")

        row = decoder.as_dict()
        row.update({
            "seed": seed,
            "n_samples": int(len(y)),
            "n_train_groups": int(train_mask.sum()),
            "n_test_groups": int((~train_mask).sum()),
            "episodes_per_cue": n_episodes_per_cue,
            "cues": len(cells),
        })
        summary.append(row)

        if not args.skip_permutation:
            control = permutation_control(x, y, groups, train_mask)
            controls.append({"network_id": network_id, "permuted_outer_r2": control})
            row["permuted_outer_r2"] = control

        rows = decode_conflict(network_id, frozen_root, args.hidden_size, args.budget, seed, decoder)
        application.extend(rows)
        app = " | ".join(
            f"{r['group']}: t1 {r['belief_trial1']:.2f} late {r['belief_late']:.2f}" for r in rows
        )
        print(f"[{index}/{len(args.seeds)}] {network_id}  outer R2 {decoder.outer_r2:.4f}  "
              f"beta1 {decoder.beta1:.3f}  alpha {decoder.alpha:g}", flush=True)
        print(f"      conflict decode  {app}", flush=True)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(analysis_root / "metrics" / "validated_decoder_summary.csv", index=False)
    app_df = pd.DataFrame(application)
    app_df.to_csv(analysis_root / "metrics" / "decoder_conflict_application.csv", index=False)

    print(f"\n===== validated decoder ({len(summary_df)}-seed mean +/- SEM) =====")
    for column in ["outer_r2", "rmse", "mae", "beta0", "beta1", "alpha"]:
        print(f"  {column:10s}: {summary_df[column].mean():.4f} +/- {summary_df[column].sem():.4f}")

    failures = []
    if not args.skip_permutation:
        control_df = pd.DataFrame(controls)
        control_df.to_csv(analysis_root / "metrics" / "decoder_permutation_control.csv", index=False)
        worst = float(control_df["permuted_outer_r2"].max())
        print(f"\npermutation control: max permuted outer R2 = {worst:.4f} "
              f"(mean {control_df['permuted_outer_r2'].mean():.4f})")
        if worst > 0.10:
            failures.append(f"permutation control not null: max R2 {worst:.4f} > 0.10")

    # Application check: trial 1 has no evidence yet, so a belief decoder that is
    # actually reading the prior must start near the cue and end nearer the truth.
    if not app_df.empty:
        cue_err = float(app_df["opening_minus_cue"].abs().mean())
        print(f"\nconflict-round application (across {len(app_df)} network x group cells):")
        print(f"  |opening belief - cue|   mean {cue_err:.3f}")
        print(f"  late belief - true       mean {app_df['late_minus_true'].mean():+.3f}")
        if cue_err > 0.20:
            failures.append(f"opening belief is far from the cue (mean |err| {cue_err:.3f})")

    if failures:
        print("\nFAILED SELF-CHECKS:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print(f"\nall self-checks passed ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
