"""Reversal learning on the `conflict_only` round, plus the non-circular decoder check.

Two halves, and the split matters:

* **Behavioural reversal** for the two groups with a *directional* target, `mac1A`
  (cue 0.30 -> true 0.60) and `mac1B` (cue 0.70 -> true 0.40): the reversal must take
  the bet rate from the cue side across to the truth side.
* **Settling** for `mac2A`/`mac2B`, whose true probability is **0.50**.  There is no
  direction to reverse toward, so a midpoint-crossing metric is *undefined* there by
  construction -- not a bug, and not something to force.  Those two are scored on how
  close they settle to neutral instead.

Crossing definition: the first trial at which `sign(P(bet) - 0.5)` matches
`sign(true - 0.5)` -- i.e. crossing the midpoint *toward the truth*.  The pre-run has
a second, different definition (`test_prior_injection.round_metrics:82` crosses the
midpoint between the opening and late levels); the two are not interchangeable and
only the truth-directed one is used here.

Return score is `plot_orthogonal_by_capacity.return_score` (imported, not rewritten):
1.0 = the oracle policy, 0.0 = the worst policy, for betting and for withholding.
`regret` is also reported, on the pre-run's convention, for comparability.

The decoder check is the one place decoded belief is used non-circularly: the decoder
never saw a conflict round.  If decoded belief tracks behaviour at the round opening
and at the round end, the decoder is reading something behaviourally real; a
systematic offset would instead point at decoder bias under distribution shift.
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
for _path in (_ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from scripts.shared.prerun_style import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    apply_style,
    finish,
    panel_label,
    save,
)
from scripts.plotting.plot_orthogonal_by_capacity import return_score  # noqa: E402  (reused, not rewritten)
from analysis.formal_behav_analysis.belief_decoder import load_decoder  # noqa: E402
from analysis.formal_behav_analysis.loading import GROUPS, TRIALS_PER_ROUND, load_cell  # noqa: E402

DEFAULT_SEEDS = tuple(range(20260804, 20260834))
OPENING_WINDOW = 5
LATE_START = 101
DIRECTIONAL = ("mac1A", "mac1B")  # true != 0.5, so a reversal direction exists
NEUTRAL = ("mac2A", "mac2B")  # true == 0.5, no direction


def cell_dir(out_root: Path, hidden: int, budget: int, seed: int, group: str) -> Path:
    return out_root / "frozen_tests" / f"h{hidden:02d}_b{budget}" / "conflict_only" / f"formal_osla_seed{seed}" / group


def collect(out_root: Path, analysis_root: Path, hidden: int, budget: int, seeds) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, trajectories = [], {}
    for seed in seeds:
        decoder_path = analysis_root / "metrics" / "validated_decoders" / f"formal_osla_seed{seed}.npz"
        if not decoder_path.exists():
            raise FileNotFoundError(f"missing decoder: {decoder_path}")
        decoder = load_decoder(decoder_path)
        for group in GROUPS:
            run_dir = cell_dir(out_root, hidden, budget, seed, group)
            if not (run_dir / "trials.csv").exists():
                raise FileNotFoundError(run_dir)
            cell = load_cell(run_dir)
            trials = cell.trials
            if trials["round_id"].nunique() != 1:
                raise SystemExit(f"{run_dir}: expected exactly one round")
            trajectory = (
                trials.groupby("trial_id")["choice_probability"].mean()
                .reindex(range(1, TRIALS_PER_ROUND + 1)).to_numpy(dtype=float)
            )
            belief = decoder(cell.hidden).mean(axis=0)
            true_probability = float(trials["true_probability"].iloc[0])
            cue_probability = float(trials["cue_probability"].iloc[0])

            if true_probability == 0.5:
                crossing = np.nan
            else:
                reached = np.sign(trajectory - 0.5) == np.sign(true_probability - 0.5)
                crossing = float(np.argmax(reached) + 1) if reached.any() else np.nan

            rows.append({
                "seed": seed, "group": group, "network_id": cell.network_id,
                "cue_probability": cue_probability,
                "true_probability": true_probability,
                "opening_pbet": float(trajectory[:OPENING_WINDOW].mean()),
                "late_pbet": float(trajectory[LATE_START - 1:].mean()),
                "mean_pbet": float(trajectory.mean()),
                "crossing_trial": crossing,
                # At p=0.50 the oracle and worst policies have the same expected
                # return, so the normalized score has a zero denominator.  Mark
                # it undefined explicitly instead of asking return_score() to
                # take means of empty positive/negative-return sets.
                "return_score": (
                    return_score(float(trajectory.sum()), true_probability)
                    if true_probability != 0.5 else np.nan
                ),
                "belief_opening": float(belief[:OPENING_WINDOW].mean()),
                "belief_late": float(belief[LATE_START - 1:].mean()),
            })
            trajectories[(seed, group)] = {"pbet": trajectory, "belief": belief}

    expected = len(seeds) * len(GROUPS)
    if len(rows) != expected:
        raise SystemExit(f"collected {len(rows)} cells, expected {expected}")
    return pd.DataFrame(rows), trajectories



def plot_reversal_metrics(df: pd.DataFrame, analysis_root: Path, hidden: int) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.8))
    positions = np.arange(len(GROUPS))

    # (a) crossing trial, defined only where a direction exists
    ax = axes[0]
    for position, group in enumerate(GROUPS):
        block = df[df["group"] == group]
        values = block["crossing_trial"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            ax.text(position, 60, "undefined\n(true = 0.50)", ha="center", va="center",
                    fontsize=7.5, color=INK_MUTED)
            continue
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(position + jitter, values, s=24, color=SERIES[0], alpha=0.72, linewidths=0, zorder=3)
        ax.scatter([position], [np.median(values)], marker="_", s=420, color=INK_PRIMARY,
                   zorder=4, linewidths=2.2)
        never = int(block["crossing_trial"].isna().sum())
        if never:
            # Sit just above the slowest dot: a fixed y would collide with the
            # panel label in the top-left corner for whichever group comes first.
            ax.text(position, values.max() + 5, f"{never} never crosses", ha="center", va="bottom",
                    fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS)
    ax.set_ylim(0, 125)
    ax.set_ylabel("trial crossing the midpoint")
    ax.set_title("Reversal speed", fontsize=9, color=INK_PRIMARY)
    panel_label(ax, "bar = median; crossing = sign(P-0.5) matches sign(true-0.5)",
                loc="upper left", box=True)

    # (b) oracle-anchored return score, defined for every group
    ax = axes[1]
    for position, group in enumerate(GROUPS):
        values = df.loc[df["group"] == group, "return_score"].dropna().to_numpy(dtype=float)
        if len(values) == 0:
            ax.text(position, 0.5, "undefined\n(true = 0.50)", ha="center", va="center",
                    fontsize=7.5, color=INK_MUTED)
            continue
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(position + jitter, values, s=24, color=SERIES[0], alpha=0.72, linewidths=0, zorder=3)
        ax.scatter([position], [np.median(values)], marker="_", s=420, color=INK_PRIMARY,
                   zorder=4, linewidths=2.2)
    ax.axhline(1.0, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("return score (1 = oracle, 0 = worst)")
    ax.set_title("How well the round is played overall", fontsize=9, color=INK_PRIMARY)

    # (c) opening -> late displacement
    ax = axes[2]
    for position, group in enumerate(GROUPS):
        block = df[df["group"] == group]
        for _, row in block.iterrows():
            ax.plot([position - 0.16, position + 0.16], [row["opening_pbet"], row["late_pbet"]],
                    color=INK_SECONDARY, lw=0.7, alpha=0.35, zorder=2)
        ax.scatter([position - 0.16] * len(block), block["opening_pbet"], s=16, color=SERIES[0],
                   alpha=0.6, linewidths=0, zorder=3)
        ax.scatter([position + 0.16] * len(block), block["late_pbet"], s=16, color=SERIES[1],
                   alpha=0.6, linewidths=0, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("P(bet)")
    ax.set_title("Opening → late shift", fontsize=9, color=INK_PRIMARY)
    handles = [
        plt.Line2D([], [], color=SERIES[0], marker="o", ls="none", ms=4, label="opening (t1-5)"),
        plt.Line2D([], [], color=SERIES[1], marker="o", ls="none", ms=4, label="late (t101-120)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7.5)

    # (d) where the round ends, against the truth and against neutral
    ax = axes[3]
    for position, group in enumerate(GROUPS):
        block = df[df["group"] == group]
        jitter = np.linspace(-0.12, 0.12, len(block))
        ax.scatter(position + jitter, block["late_pbet"], s=24, color=SERIES[0], alpha=0.72,
                   linewidths=0, zorder=3)
        ax.scatter([position], [block["late_pbet"].median()], marker="_", s=420,
                   color=INK_PRIMARY, zorder=4, linewidths=2.2)
        ax.scatter([position + 0.34], [block["true_probability"].iloc[0]], marker="v", s=30,
                   color=INK_MUTED, zorder=4, linewidths=0)
    ax.axhline(0.5, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(GROUPS)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("late P(bet), trials 101-120")
    ax.set_title("Where the round ends", fontsize=9, color=INK_PRIMARY)
    panel_label(ax, "triangle = the true probability", loc="upper left", box=True)

    finish(
        fig,
        title=f"Reversal and settling on conflict_only — {df['seed'].nunique()} networks, h = {hidden}, "
              f"one round from a blank state\n"
              f"mac1A/mac1B have a direction to reverse toward; mac2A/mac2B have true = 0.50, "
              f"so reversal latency is undefined there",
        legend_handles=None,
    )
    save(fig, analysis_root / "figures" / f"formal_reversal_metrics_h{hidden:02d}.png")


def plot_decoder_check(df: pd.DataFrame, analysis_root: Path, hidden: int) -> None:
    """Decoded belief vs behaviour -- the decoder's only non-circular use."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    panels = (
        ("belief_opening", "opening_pbet", "Round opening (trials 1-5)"),
        ("belief_late", "late_pbet", "Round end (trials 101-120)"),
    )
    for ax, (belief_column, behaviour_column, title) in zip(axes, panels):
        x = df[belief_column].to_numpy(dtype=float)
        y = df[behaviour_column].to_numpy(dtype=float)
        for position, group in enumerate(GROUPS):
            mask = (df["group"] == group).to_numpy()
            ax.scatter(x[mask], y[mask], s=26, color=SERIES[0], alpha=0.6, linewidths=0,
                       zorder=3, label="a network x group cell" if position == 0 else None)
        low = min(x.min(), y.min()) - 0.05
        high = max(x.max(), y.max()) + 0.05
        ax.plot([low, high], [low, high], color=INK_MUTED, lw=0.9, ls="--", zorder=1)
        residual = y - x
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xlabel("decoded belief  D(h) = w·h + c")
        ax.set_ylabel("behavioural P(bet)")
        ax.set_title(title, fontsize=9, color=INK_PRIMARY)
        panel_label(ax, f"mean offset {residual.mean():+.3f}\nsd {residual.std():.3f}\n"
                        f"r = {np.corrcoef(x, y)[0, 1]:.3f}",
                    loc="upper left", box=True)
    axes[0].legend(loc="lower right")
    finish(
        fig,
        title="Decoder check on data it never saw — the conflict round\n"
              "no group colour: the four groups are the four clusters; dashed = identity",
        legend_handles=None,
    )
    save(fig, analysis_root / "figures" / f"formal_decoder_vs_behavior_h{hidden:02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path,
                        default=_ROOT / "analysis" / "formal_behav_analysis")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    analysis_root = args.analysis_root.resolve()
    metrics_dir = analysis_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df, _ = collect(out_root, analysis_root, args.hidden_size, args.budget, args.seeds)
    df.to_csv(metrics_dir / f"reversal_metrics_h{args.hidden_size:02d}.csv", index=False)

    print(f"cells: {len(df)}  networks: {df['seed'].nunique()}\n")
    print(f"{'group':6s} {'cue':>5s} {'true':>5s} {'opening':>8s} {'late':>7s} {'cross med':>10s} "
          f"{'never':>6s} {'score':>7s}")
    for group in GROUPS:
        b = df[df["group"] == group]
        crossing = b["crossing_trial"].dropna()
        median = f"{crossing.median():.0f}" if len(crossing) else "undefined"
        # true == 0.5 makes the oracle and the worst policy both earn zero, so the
        # return score has no scale there.  np.nanmean on an all-NaN column would warn.
        score = b["return_score"]
        score_text = f"{score.mean():7.3f}" if score.notna().any() else "  n/a  "
        print(f"{group:6s} {b['cue_probability'].iloc[0]:5.2f} {b['true_probability'].iloc[0]:5.2f} "
              f"{b['opening_pbet'].mean():8.3f} {b['late_pbet'].mean():7.3f} {median:>10s} "
              f"{int(b['crossing_trial'].isna().sum()):6d} {score_text}")

    print("\ndirectional groups only (mac2A/mac2B have no reversal direction):")
    for group in DIRECTIONAL:
        b = df[df["group"] == group]
        crossing = b["crossing_trial"].dropna()
        print(f"  {group}: crossing median {crossing.median():.0f} "
              f"range {crossing.min():.0f}-{crossing.max():.0f}, "
              f"{int(b['crossing_trial'].isna().sum())}/{len(b)} never cross; "
              f"shift {b['opening_pbet'].mean():.3f} -> {b['late_pbet'].mean():.3f} "
              f"(true {b['true_probability'].iloc[0]:.2f}); "
              f"score {b['return_score'].mean():.3f} +/- {b['return_score'].std():.3f}")

    print("\ndecoder check (conflict round -- never seen by the decoder):")
    for belief_column, behaviour_column, label in (
        ("belief_opening", "opening_pbet", "opening"),
        ("belief_late", "late_pbet", "late"),
    ):
        offset = (df[behaviour_column] - df[belief_column])
        print(f"  {label:8s} mean offset {offset.mean():+.3f}  sd {offset.std():.3f}  "
              f"r {np.corrcoef(df[belief_column], df[behaviour_column])[0, 1]:.3f}")

    plot_reversal_metrics(df, analysis_root, args.hidden_size)
    plot_decoder_check(df, analysis_root, args.hidden_size)


if __name__ == "__main__":
    main()
