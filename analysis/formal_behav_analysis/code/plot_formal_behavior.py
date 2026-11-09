"""Behavioural distributions on the formal batch's `conflict_only` test round.

Reads `frozen_tests/h<hh>_b<budget>/conflict_only/formal_osla_seed*/<group>/trials.csv`
for every network x group cell, and reports both the **trajectory** and the
**spread across networks**.

The spread is the point, not an extra: 30 networks in one arm still differ, and a
mean-only figure would hide exactly the between-network variation that decides
whether a downstream gate is meaningful.

Presentation follows `prerun_style` and the dataviz rules: `SERIES` has three
categorical slots and slot order must stay fixed, so 30 networks are drawn in one
neutral family (thin, low alpha) with the mean bolded -- identity comes from
position, not hue.  Distribution panels use one categorical slot per *group*
(there are exactly four), which is what the slots are for.

`regret` is lifted from the pre-run's `plot_pbet_structure.summarize()`
(`scripts/plot_pbet_structure.py:81-112`) so this batch's numbers stay comparable
with the pre-run's, including its `oracle` convention.  That script itself cannot
run here (it requires all three protocols and a 3-seed lookup table), so the
arithmetic is copied rather than imported.
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

_ROOT = Path(__file__).resolve().parents[3]
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
from analysis.formal_behav_analysis.loading import GROUPS, TRIALS_PER_ROUND, load_cell  # noqa: E402

DEFAULT_SEEDS = tuple(range(20260804, 20260834))
OPENING_WINDOW = 5
LATE_START = 101
# One validated categorical slot for all four groups, not four.
# `prerun_style.SERIES` holds three validated slots and the dataviz rules forbid
# cycling them or inventing an unvalidated fourth.  Here a fourth hue would be
# redundant anyway: every group gets its own panel or its own labelled x position, so
# the group identity is already carried by position and label.  Redundant colour is
# only acceptable when the colour itself is validated.
GROUP_COLOUR = dict.fromkeys(GROUPS, SERIES[0])


def cell_dir(out_root: Path, hidden: int, budget: int, protocol: str, seed: int, group: str) -> Path:
    return out_root / "frozen_tests" / f"h{hidden:02d}_b{budget}" / protocol / f"formal_osla_seed{seed}" / group


def behaviour_metrics(cell) -> dict:
    """Per-cell summary. `regret`/`oracle` follow plot_pbet_structure.summarize()."""
    trials = cell.trials
    if trials["round_id"].nunique() != 1:
        raise SystemExit(f"{cell.run_dir}: expected exactly one round for conflict_only")
    trajectory = (
        trials.groupby("trial_id")["choice_probability"].mean()
        .reindex(range(1, TRIALS_PER_ROUND + 1)).to_numpy(dtype=float)
    )
    if not np.isfinite(trajectory).all():
        raise SystemExit(f"{cell.run_dir}: trajectory has gaps")
    action_rate = float(trials["action"].mean())
    # Binary entropy of the action stream: 0 = fully committed, 1 nat = indifferent.
    p = min(max(action_rate, 1e-9), 1 - 1e-9)
    true_probability = float(trials["true_probability"].iloc[0])
    oracle = float(np.mean(np.where(trials["true_probability"] > 0.5, trials["true_probability"], 0.0)))
    spread = float(np.nanmax(trajectory) - np.nanmin(trajectory))
    return {
        "start_pbet": float(trajectory[0]),
        "opening_pbet": float(np.nanmean(trajectory[:OPENING_WINDOW])),
        "late_pbet": float(np.nanmean(trajectory[LATE_START - 1:])),
        "mean_pbet": float(np.nanmean(trajectory)),
        "action_rate": action_rate,
        "action_entropy": float(-(p * np.log(p) + (1 - p) * np.log(1 - p))),
        "regret": oracle - float(trials["reward"].mean()),
        "traj_range": spread,
        "flat_flag": int(spread < 0.1),
        "_traj": trajectory,
    }


def collect(out_root: Path, hidden: int, budget: int, protocol: str, seeds) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for group in GROUPS:
            run_dir = cell_dir(out_root, hidden, budget, protocol, seed, group)
            if not (run_dir / "trials.csv").exists():
                raise FileNotFoundError(run_dir)
            cell = load_cell(run_dir)
            record = {
                "seed": seed, "group": group, "network_id": cell.network_id,
                "episodes": cell.n_episodes,
                "cue_probability": float(cell.trials["cue_probability"].iloc[0]),
                "true_probability": float(cell.trials["true_probability"].iloc[0]),
                **behaviour_metrics(cell),
            }
            rows.append(record)
    expected = len(seeds) * len(GROUPS)
    if len(rows) != expected:
        raise SystemExit(f"collected {len(rows)} cells, expected {expected}")
    return pd.DataFrame(rows)


def plot_trajectories(df: pd.DataFrame, analysis_root: Path, hidden: int, budget: int,
                      protocol: str) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6), sharey=True)
    for ax, group in zip(axes, GROUPS):
        block = df[df["group"] == group]
        curves = np.vstack(block["_traj"].to_numpy())
        trials = np.arange(1, curves.shape[1] + 1)
        cue = float(block["cue_probability"].iloc[0])
        true = float(block["true_probability"].iloc[0])
        colour = GROUP_COLOUR[group]

        for curve in curves:
            ax.plot(trials, curve, color=INK_SECONDARY, lw=0.5, alpha=0.18, zorder=2)
        ax.axhline(cue, color=INK_MUTED, lw=0.9, ls="--", zorder=1)
        ax.axhline(true, color=INK_MUTED, lw=0.9, ls="-.", zorder=1)
        mean, sd = curves.mean(axis=0), curves.std(axis=0)
        ax.fill_between(trials, mean - sd, mean + sd, color=colour, alpha=0.18, linewidth=0)
        ax.plot(trials, mean, color=colour, lw=1.9, zorder=4)

        ax.set_title(f"{group}   cue {cue:.2f} → true {true:.2f}", fontsize=9, color=INK_PRIMARY)
        ax.set_xlim(1, curves.shape[1])
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([1, 60, 120])
        ax.set_xlabel("trial within the round")
        panel_label(ax, f"opens {block['opening_pbet'].mean():.2f}\nends {block['late_pbet'].mean():.2f}",
                    loc="lower right", box=True)
    axes[0].set_ylabel("P(bet)")

    handles = [
        plt.Line2D([], [], color=INK_SECONDARY, lw=0.8, alpha=0.5, label="each network (n = 30)"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.9, ls="--", label="stated prior (cue)"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.9, ls="-.", label="true probability"),
    ]
    finish(
        fig,
        title=f"Formal batch test behaviour — {protocol}, one round from a blank state\n"
              f"bold = mean of {df['seed'].nunique()} networks (h = {hidden}), band = ±1 s.d. across networks, "
              f"thin grey = each network individually",
        legend_handles=handles, ncol=3,
    )
    save(fig, analysis_root / "figures" / f"formal_behavior_trajectories_h{hidden:02d}_b{budget}.png")


def plot_distributions(df: pd.DataFrame, analysis_root: Path, hidden: int, budget: int,
                       protocol: str) -> None:
    apply_style()
    panels = (
        ("opening_pbet", "opening P(bet), trials 1-5"),
        ("late_pbet", "late P(bet), trials 101-120"),
        ("mean_pbet", "whole-round P(bet)"),
        ("action_entropy", "action entropy (nats)"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.9), sharex=True)
    positions = np.arange(len(GROUPS))

    for ax, (column, title) in zip(axes, panels):
        for position, group in enumerate(GROUPS):
            values = df.loc[df["group"] == group, column].to_numpy(dtype=float)
            jitter = np.linspace(-0.12, 0.12, len(values))
            ax.scatter(position + jitter, values, s=24, color=GROUP_COLOUR[group],
                       alpha=0.72, linewidths=0, zorder=3)
            ax.scatter([position], [np.median(values)], marker="_", s=420,
                       color=INK_PRIMARY, zorder=4, linewidths=2.2)
            ax.text(position, ax.get_ylim()[0], "", ha="center")
        ax.set_title(title, fontsize=9, color=INK_PRIMARY)
        ax.set_xticks(positions)
        ax.set_xticklabels(GROUPS)
        panel_label(ax, "bar = median", loc="upper left", box=True)

    # Each panel gets its own natural range; only the x axis is shared.
    axes[0].set_ylabel("value")
    finish(
        fig,
        title=f"Between-network spread on {protocol} — one dot per network (n = {df['seed'].nunique()} per group)\n"
              f"h = {hidden}, {int(df['episodes'].iloc[0])} episodes per network per group",
        legend_handles=None,
    )
    save(fig, analysis_root / "figures" / f"formal_behavior_distributions_h{hidden:02d}_b{budget}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path,
                        default=_ROOT / "analysis" / "formal_behav_analysis")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--protocol", default="conflict_only")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    analysis_root = args.analysis_root.resolve()
    (analysis_root / "metrics").mkdir(parents=True, exist_ok=True)

    df = collect(out_root, args.hidden_size, args.budget, args.protocol, args.seeds)
    metrics_path = analysis_root / "metrics" / f"formal_behavior_{args.protocol}_h{args.hidden_size:02d}_b{args.budget}.csv"
    df.drop(columns=["_traj"]).to_csv(metrics_path, index=False)

    print(f"cells: {len(df)}  networks: {df['seed'].nunique()}  episodes/cell: {df['episodes'].iloc[0]}")
    print(f"\n{'group':6s} {'cue':>5s} {'true':>5s} {'opening':>8s} {'late':>7s} {'mean':>6s} "
          f"{'entropy':>8s} {'regret':>7s} {'flat':>4s}")
    for group in GROUPS:
        b = df[df["group"] == group]
        print(f"{group:6s} {b['cue_probability'].iloc[0]:5.2f} {b['true_probability'].iloc[0]:5.2f} "
              f"{b['opening_pbet'].mean():8.3f} {b['late_pbet'].mean():7.3f} {b['mean_pbet'].mean():6.3f} "
              f"{b['action_entropy'].mean():8.3f} {b['regret'].mean():7.3f} {b['flat_flag'].sum():4d}")

    print("\nbetween-network s.d. (the spread a mean-only figure would hide):")
    for column in ("opening_pbet", "late_pbet", "mean_pbet", "action_entropy"):
        spreads = "  ".join(f"{g}:{df.loc[df['group'] == g, column].std():.3f}" for g in GROUPS)
        print(f"  {column:15s} {spreads}")

    plot_trajectories(df, analysis_root, args.hidden_size, args.budget, args.protocol)
    plot_distributions(df, analysis_root, args.hidden_size, args.budget, args.protocol)
    print(f"\nwrote {metrics_path}")


if __name__ == "__main__":
    main()
