"""Decoded belief trajectories on the conflict round, 30 networks.

This is the figure that shows the decoder works on the data it will actually be
used on.  The decoder is *fit* on held-out consistent rounds; here it is *applied*
to the `conflict_only` trajectories, where it never saw a label.  A belief readout
that is doing its job should start at the cue (no evidence has arrived yet) and move
toward the environment's true probability as outcomes accumulate.

Reads:
    <out-root>/frozen_tests/h16_b200/conflict_only/formal_osla_seed<seed>/<group>/
    <analysis-root>/metrics/validated_decoders/formal_osla_seed<seed>.npz

Writes:
    <analysis-root>/figures/decoded_belief_conflict.png (+ .pdf, + the CSV twin)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from scripts.shared.prerun_style import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    SERIES,
    apply_style,
    finish,
    panel_label,
    save,
)
from analysis.formal_behav_analysis.belief_decoder import load_decoder  # noqa: E402
from analysis.formal_behav_analysis.loading import GROUPS, load_cell  # noqa: E402

DEFAULT_SEEDS = tuple(range(20260804, 20260834))
OPENING_WINDOW = 5
LATE_START = 101
# One categorical slot per group; the four groups are identities, never an ordering.
GROUP_COLOUR = dict(zip(GROUPS, (SERIES[0], SERIES[1], SERIES[2], "#7b5bb8")))


def collect(out_root: Path, analysis_root: Path, hidden: int, seeds) -> pd.DataFrame:
    """Per (seed, group) decoded trajectory plus its cue/true context."""
    records = []
    arm = f"h{hidden:02d}_b200"
    for seed in seeds:
        decoder_path = analysis_root / "metrics" / "validated_decoders" / f"formal_osla_seed{seed}.npz"
        if not decoder_path.exists():
            raise FileNotFoundError(f"missing decoder: {decoder_path} -- run train_belief_decoder.py first")
        decoder = load_decoder(decoder_path)
        for group in GROUPS:
            run_dir = out_root / "frozen_tests" / arm / "conflict_only" / f"formal_osla_seed{seed}" / group
            if not (run_dir / "trials.csv").exists():
                continue
            cell = load_cell(run_dir)
            belief = decoder(cell.hidden)  # (episodes, trials)
            trajectory = belief.mean(axis=0)
            if len(trajectory) != cell.n_trials:
                raise ValueError(f"{run_dir}: trajectory {trajectory.shape} vs {cell.n_trials} trials")
            record = {
                "seed": seed,
                "group": group,
                "cue": float(cell.trials["cue_probability"].iloc[0]),
                "true": float(cell.trials["true_probability"].iloc[0]),
                "episodes": cell.n_episodes,
                "belief_trial1": float(belief[:, 0].mean()),
                "opening": float(belief[:, :OPENING_WINDOW].mean()),
                "late": float(belief[:, LATE_START - 1:].mean()),
            }
            for trial, value in enumerate(trajectory, start=1):
                record[f"t{trial}"] = float(value)
            records.append(record)
    if not records:
        raise FileNotFoundError(f"no conflict cells under {out_root}/frozen_tests/{arm}")
    return pd.DataFrame(records)


def plot(df: pd.DataFrame, analysis_root: Path, hidden: int, n_seeds: int) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.4), sharex=True, sharey=True)
    trial_columns = [f"t{t}" for t in range(1, 121)]

    for ax, group in zip(axes.ravel(), GROUPS):
        block = df[df["group"] == group]
        if block.empty:
            ax.set_visible(False)
            continue
        curves = block[trial_columns].to_numpy(dtype=float)
        mean, sd = curves.mean(axis=0), curves.std(axis=0)
        trials = np.arange(1, len(mean) + 1)
        colour = GROUP_COLOUR[group]

        cue = float(block["cue"].iloc[0])
        true = float(block["true"].iloc[0])
        ax.axhline(cue, color=INK_MUTED, lw=0.9, ls="--", zorder=1)
        ax.axhline(true, color=INK_MUTED, lw=0.9, ls="-.", zorder=1)
        ax.fill_between(trials, mean - sd, mean + sd, color=colour, alpha=0.16, linewidth=0)
        ax.plot(trials, mean, color=colour, lw=1.9, zorder=3)

        ax.set_title(f"{group}   cue {cue:.2f} → true {true:.2f}", fontsize=9, color=INK_PRIMARY)
        ax.set_xlim(1, len(mean))
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([1, 60, 120])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        panel_label(
            ax,
            f"opens {block['opening'].mean():.2f}  →  ends {block['late'].mean():.2f}",
            loc="lower right", box=True,
        )

    for ax in axes[-1]:
        ax.set_xlabel("trial within the conflict round")
    for ax in axes[:, 0]:
        ax.set_ylabel("decoded belief  D(h) = w·h + c")

    handles = [
        plt.Line2D([], [], color=INK_MUTED, lw=0.9, ls="--", label="stated prior (cue)"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.9, ls="-.", label="true probability"),
    ]
    episodes = sorted(df["episodes"].unique())
    finish(
        fig,
        title="Decoded belief on the conflict round — the decoder applied where it has no labels\n"
              f"mean of {n_seeds} networks (h = {hidden}), shaded band = ±1 s.d. across networks; "
              + (f"{episodes[0]} episodes per network per group" if len(episodes) == 1
                 else f"episodes per group: {episodes}"),
        legend_handles=handles, ncol=2,
    )
    save(fig, analysis_root / "figures" / "decoded_belief_conflict.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path,
                        default=_ROOT / "analysis" / "formal_behav_analysis")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    # `n_episodes` in the title is read from the data rather than assumed.
    out_root = args.out_root.resolve()
    analysis_root = args.analysis_root.resolve()

    df = collect(out_root, analysis_root, args.hidden_size, args.seeds)
    metrics_dir = analysis_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "decoded_belief_conflict_trajectories.csv", index=False)

    print(f"groups: {sorted(df['group'].unique())}  networks: {df['seed'].nunique()}")
    print(f"{'group':6s} {'cue':>5s} {'true':>5s} {'opens':>7s} {'ends':>7s} {'end-true':>9s}")
    for group, block in df.groupby("group", sort=True):
        print(f"{group:6s} {block['cue'].iloc[0]:5.2f} {block['true'].iloc[0]:5.2f} "
              f"{block['opening'].mean():7.3f} {block['late'].mean():7.3f} "
              f"{block['late'].mean() - block['true'].iloc[0]:+9.3f}")

    plot(df, analysis_root, args.hidden_size, df["seed"].nunique())


if __name__ == "__main__":
    main()
