"""Figure B: betting-rate curves under different test round structures, by seed.

The conflict round is the last round of each episode.  In the new code every
round is 120 behavioral trials and `trial_id` runs 1..120 correctly (it never
wraps), so `trial_id` can be used directly -- the old `real_round_trial` patch
from the 240-trial `test3_extended` era is not needed.

Why the three protocols sit side by side: `FORMAL_CONFLICTS` gives each group a
consistent-round cue that is the *complement* of its conflict-round cue
(mac1A/mac2A: 0.70 -> 0.30; mac1B/mac2B: 0.30 -> 0.70), and the hidden state is
never reset between rounds.  So in `human_default` / `legacy_3plus1` the
conflict-round opening P(bet) is dominated by the *carried* state from the
previous round, not by the one-shot cue.  Only `conflict_only` (no preceding
round) gives a clean read of the cue's influence.  The old baseline's
`cue_lock_corr = 0.945` was computed with *random* preceding-round cues, so it is
not the same quantity as this column.

`regret` here is `mean(where(true > 0.5, true, 0)) - mean(reward)`.  Its floor is
`1 - true` for `true > 0.5`, i.e. 0.4 for mac1A, so mac1A regret near 0.4 means a
policy that simply keeps betting -- not a policy that used the prior well.  Read
it next to `late_pbet`, the trajectory and `flat_flag`, never alone.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# `formal_osla/__init__.py` imports torch, so pin the thread count before that
# happens -- the protocol constants are the only thing this script needs from the
# package, but the import still has to be deterministic, and `scripts/` must be
# on the path (with the repo root) when run directly.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.shared.prerun_style import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    SERIES,
    apply_style,
    finish,
    panel_label,
    reference_line,
    save,
)
import matplotlib.pyplot as plt  # noqa: E402

from formal_osla.protocols import FORMAL_CONFLICTS, FORMAL_GROUPS  # noqa: E402

PROTOCOLS = ("conflict_only", "human_default", "legacy_3plus1")
PROTOCOL_NOTE = {
    "conflict_only": "0 preceding rounds",
    "human_default": "1 preceding round, opposed cue",
    "legacy_3plus1": "3 preceding rounds, opposed cue",
}
SEED_SHORT = {20260928: "s28", 20260929: "s29", 20260930: "s30"}
TRIALS_PER_ROUND = 120
OPENING_WINDOW = 5
LATE_START = 101


def trial_dir(output_root: Path, hidden: int, budget: int, protocol: str, seed: int, group: str) -> Path:
    return (
        output_root / "frozen_tests" / f"h{hidden:02d}_b{budget}" / protocol
        / f"formal_osla_seed{seed}" / group
    )


def summarize(trials: pd.DataFrame) -> dict:
    """Metrics for the conflict round (the last round) of one network x group."""
    conflict_round = int(trials["round_id"].max())
    conflict = trials[trials["round_id"] == conflict_round]
    traj = (
        conflict.groupby("trial_id")["choice_probability"].mean()
        .reindex(range(1, TRIALS_PER_ROUND + 1)).to_numpy(dtype=float)
    )
    action_traj = (
        conflict.groupby("trial_id")["action"].mean()
        .reindex(range(1, TRIALS_PER_ROUND + 1)).to_numpy(dtype=float)
    )
    true_probability = float(conflict["true_probability"].iloc[0])
    cue_probability = float(conflict["cue_probability"].iloc[0])
    oracle = float(np.mean(np.where(conflict["true_probability"] > 0.5, conflict["true_probability"], 0.0)))
    spread = float(np.nanmax(traj) - np.nanmin(traj))
    return {
        "conflict_round": conflict_round,
        "cue_probability": cue_probability,
        "true_probability": true_probability,
        "start_pbet": float(traj[0]),
        "opening_pbet": float(np.nanmean(traj[:OPENING_WINDOW])),
        "late_pbet": float(np.nanmean(traj[LATE_START - 1:])),
        "mean_pbet": float(np.nanmean(traj)),
        "action_rate": float(conflict["action"].mean()),
        "regret": oracle - float(conflict["reward"].mean()),
        "regret_floor": max(1.0 - true_probability, 0.0) if true_probability > 0.5 else 0.0,
        "traj_range": spread,
        "flat_flag": int(spread < 0.1),
        "_traj": traj,
        "_action_traj": action_traj,
    }


def collect(output_root: Path, hidden: int, budget: int, seeds) -> tuple[dict, pd.DataFrame]:
    data: dict = {}
    rows: list[dict] = []
    for protocol in PROTOCOLS:
        for group in FORMAL_GROUPS:
            for seed in seeds:
                path = trial_dir(output_root, hidden, budget, protocol, seed, group) / "trials.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                summary = summarize(pd.read_csv(path))
                data[(protocol, group, seed)] = summary
                rows.append(
                    {
                        "hidden_size": hidden,
                        "protocol": protocol,
                        "group": group,
                        "seed": seed,
                        **{k: v for k, v in summary.items() if not k.startswith("_")},
                    }
                )
    return data, pd.DataFrame(rows)


def cue_lock_corr(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pearson r between the conflict cue and start_pbet across the four groups.

    Only meaningful for `conflict_only` -- see the module docstring.  Reported for
    the other protocols too, but there it measures carried-state inertia.
    """
    rows: list[dict] = []
    for (hidden, protocol, seed), frame in metrics.groupby(["hidden_size", "protocol", "seed"]):
        frame = frame.sort_values("group")
        cues = frame["cue_probability"].to_numpy(dtype=float)
        starts = frame["start_pbet"].to_numpy(dtype=float)
        r = float("nan") if np.std(cues) == 0 or np.std(starts) == 0 else float(np.corrcoef(cues, starts)[0, 1])
        rows.append({"hidden_size": hidden, "protocol": protocol, "seed": seed,
                     "cue_lock_corr": r, "n_flat": int(frame["flat_flag"].sum()),
                     "n_groups": len(frame),
                     "mean_regret": float(frame["regret"].mean())})
    return pd.DataFrame(rows)


def figB1(data: dict, hidden: int, budget: int, seeds, output_root: Path) -> None:
    fig, axes = plt.subplots(
        len(PROTOCOLS), len(FORMAL_GROUPS), figsize=(11.0, 7.8), sharex=True, sharey=True
    )
    for row, protocol in enumerate(PROTOCOLS):
        for col, group in enumerate(FORMAL_GROUPS):
            ax = axes[row, col]
            n_flat = 0
            for s_index, seed in enumerate(seeds):
                summary = data[(protocol, group, seed)]
                x = np.arange(1, TRIALS_PER_ROUND + 1)
                ax.plot(x, summary["_traj"], color=SERIES[s_index], lw=1.6)
                ax.plot(x, summary["_action_traj"], color=SERIES[s_index], lw=1.0,
                        alpha=0.45, ls=(0, (3, 2)))
                # Direct labels satisfy the contrast-relief rule for the aqua slot;
                # stagger them so converging seeds do not overprint each other.
                ax.annotate(SEED_SHORT[seed], xy=(TRIALS_PER_ROUND, summary["late_pbet"]),
                            xytext=(4, (s_index - 1) * 6), textcoords="offset points",
                            fontsize=6.5, color=SERIES[s_index], va="center")
                n_flat += summary["flat_flag"]
            cue = FORMAL_CONFLICTS[group]["conflict_cue"]
            true = FORMAL_CONFLICTS[group]["conflict_true"]
            reference_line(ax, cue)
            reference_line(ax, true, style="-.")
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, TRIALS_PER_ROUND + 16)
            if row == 0:
                ax.set_title(group, color=INK_PRIMARY, fontsize=9.5)
            if col == 0:
                ax.set_ylabel(f"{protocol}\n{PROTOCOL_NOTE[protocol]}", fontsize=7.5)
            if row == len(PROTOCOLS) - 1:
                ax.set_xlabel("conflict-round trial")
            panel_label(ax, f"cue {cue:.2f}  true {true:.2f}\nflat {n_flat}/{len(seeds)}",
                        loc="lower left", box=True)

    handles = [plt.Line2D([], [], color=SERIES[i], lw=1.6, label=f"seed {seed}")
               for i, seed in enumerate(seeds)]
    handles += [
        plt.Line2D([], [], color=INK_MUTED, lw=1.0, ls=(0, (3, 2)), label="realised action rate"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.8, ls="--", label="conflict cue"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.8, ls="-.", label="true probability"),
    ]
    try:
        fig.supylabel("P(bet)", fontsize=8.5, color=INK_SECONDARY)
    except AttributeError:  # older matplotlib
        pass
    finish(
        fig,
        title=(
            f"B1  Conflict-round betting rate by test round structure  "
            f"(hidden={hidden}, {budget}-update checkpoints)\n"
            "solid = P(bet), light dashed = realised action rate. In the multi-round protocols the opening "
            "value is carried state, not the one-shot cue."
        ),
        legend_handles=handles,
        ncol=5,
    )
    save(fig, output_root / "figures" / f"figB1_pbet_by_round_structure_h{hidden}_b{budget}.png")


def figB2(metrics: pd.DataFrame, hidden: int, budget: int, seeds, output_root: Path) -> None:
    fig, axes = plt.subplots(1, len(FORMAL_GROUPS), figsize=(10.4, 3.8), sharey=True)
    offset = 0.22
    positions = {protocol: i for i, protocol in enumerate(PROTOCOLS)}
    for col, group in enumerate(FORMAL_GROUPS):
        ax = axes[col]
        frame = metrics[(metrics["group"] == group) & (metrics["hidden_size"] == hidden)]
        for s_index, seed in enumerate(seeds):
            sub = frame[frame["seed"] == seed]
            for _, row in sub.iterrows():
                x = positions[row["protocol"]] + (s_index - 1) * offset
                ax.scatter(x, row["start_pbet"], color=SERIES[s_index], marker="o", s=30, zorder=5)
                ax.scatter(x, row["late_pbet"], color=SERIES[s_index], marker="s", s=30, zorder=5)
        reference_line(ax, FORMAL_CONFLICTS[group]["conflict_cue"])
        reference_line(ax, FORMAL_CONFLICTS[group]["conflict_true"], style="-.")
        ax.set_xticks(range(len(PROTOCOLS)))
        ax.set_xticklabels([p.replace("_", "\n") for p in PROTOCOLS], fontsize=6.8)
        ax.set_title(group, color=INK_PRIMARY, fontsize=9.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, len(PROTOCOLS) - 0.5)
        if col == 0:
            ax.set_ylabel("P(bet)")
    handles = [plt.Line2D([], [], color=SERIES[i], marker="o", ls="none", ms=5, label=f"seed {seed}")
               for i, seed in enumerate(seeds)]
    handles += [
        plt.Line2D([], [], color=INK_SECONDARY, marker="o", ls="none", ms=5, label="start (trial 1)"),
        plt.Line2D([], [], color=INK_SECONDARY, marker="s", ls="none", ms=5, label="late (trials 101-120)"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.8, ls="--", label="conflict cue"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.8, ls="-.", label="true probability"),
    ]
    finish(
        fig,
        title=(
            f"B2  Opening vs late P(bet) by round structure  (hidden={hidden}, {budget}-update checkpoints)\n"
            "the same numbers are in metrics/pbet_by_structure.csv"
        ),
        legend_handles=handles,
        ncol=7,
    )
    save(fig, output_root / "figures" / f"figB2_pbet_summary_h{hidden}_b{budget}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figure B (betting rate by round structure).")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "outputs_prerun")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260928, 20260929, 20260930])
    args = parser.parse_args()

    apply_style()
    output_root = args.output_root.resolve()
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[pd.DataFrame] = []
    all_corr: list[pd.DataFrame] = []
    for hidden in args.hidden_sizes:
        data, metrics = collect(output_root, hidden, args.budget, args.seeds)
        all_metrics.append(metrics)
        figB1(data, hidden, args.budget, args.seeds, output_root)
        figB2(metrics, hidden, args.budget, args.seeds, output_root)
        all_corr.append(cue_lock_corr(metrics))

    corr = pd.concat(all_corr, ignore_index=True)
    corr.to_csv(metrics_dir / f"cue_lock_corr_b{args.budget}.csv", index=False)
    print(f"\ncue_lock_corr at {args.budget} updates "
          f"(only conflict_only is a clean cue read; the others measure carried-state inertia):")
    print(corr.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    pd.concat(all_metrics, ignore_index=True).to_csv(
        metrics_dir / f"pbet_by_structure_b{args.budget}.csv", index=False
    )
    print(f"\nwrote metrics/pbet_by_structure_b{args.budget}.csv, metrics/cue_lock_corr_b{args.budget}.csv")


if __name__ == "__main__":
    main()
