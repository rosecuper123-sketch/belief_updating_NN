"""Prior-injection robustness on the frozen networks, single reversal round only.

Question: with exactly ONE round (so nothing is carried in from a previous round),
does the network (a) accept the stated prior and open by betting at roughly the
prior's probability, and (b) then reverse according to the evidence?

Design: cue x true = {0.30, 0.70} x {prior, neutral, against}.  Every cell is a
*blank-state* single round, so no carried state can confound the reading:

| cue  | true | what it asks of the network                    | where it lives            |
|------|------|------------------------------------------------|---------------------------|
| 0.30 | 0.30 | accept the prior, keep betting ~0.30           | human_default round 1     |
| 0.30 | 0.50 | prior says 0.30, truth is indifferent          | conflict_only (mac2A)     |
| 0.30 | 0.60 | prior says 0.30, truth says bet: reverse UP    | conflict_only (mac1A)     |
| 0.70 | 0.70 | accept the prior, keep betting ~0.70           | human_default round 1     |
| 0.70 | 0.50 | prior says 0.70, truth is indifferent          | conflict_only (mac2B)     |
| 0.70 | 0.40 | prior says 0.70, truth says don't: reverse DOWN| conflict_only (mac1B)     |

`human_default` round 1 is the episode's first round, i.e. it starts from a zero
hidden state exactly like a `conflict_only` round; that was verified bit-exactly
against `legacy_3plus1` round 1 earlier, so the two sources are interchangeable
for a blank-state reading.

This script is deliberately **torch-free** (pandas + numpy + matplotlib only) so
it can run while the training-side environment is unavailable; it reads only
already-collected frozen-test CSVs.  Reuse note: `plot_pbet_structure.summarize`
does the same metric arithmetic but needs `formal_osla` (and therefore torch), so
the small metric function is repeated here instead of imported.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.shared.prerun_style import (
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ORDINAL,
    ROOT,
    apply_style,
    finish,
    reference_line,
    save,
)
import matplotlib.pyplot as plt

TRIALS_PER_ROUND = 120
EPISODES = 200
OPENING_WINDOW = 5
LATE_START = 101
# (cue, true, scenario, protocol, group, round_id)
CELLS = (
    (0.30, 0.30, "prior only (cue == true)", "human_default", "mac1B", 1),
    (0.30, 0.50, "neutral truth", "conflict_only", "mac2A", 1),
    (0.30, 0.60, "reverse UP", "conflict_only", "mac1A", 1),
    (0.70, 0.70, "prior only (cue == true)", "human_default", "mac1A", 1),
    (0.70, 0.50, "neutral truth", "conflict_only", "mac2B", 1),
    (0.70, 0.40, "reverse DOWN", "conflict_only", "mac1B", 1),
)
HIDDEN_SIZES = (16, 32)


def round_metrics(trials: pd.DataFrame, round_id: int) -> dict:
    frame = trials[trials["round_id"] == round_id]
    if len(frame) != EPISODES * TRIALS_PER_ROUND:
        raise SystemExit(f"round {round_id}: expected {EPISODES * TRIALS_PER_ROUND} rows, got {len(frame)}")
    traj = (
        frame.groupby("trial_id")["choice_probability"].mean()
        .reindex(range(1, TRIALS_PER_ROUND + 1)).to_numpy(dtype=float)
    )
    if not np.isfinite(traj).all():
        raise SystemExit(f"round {round_id}: trajectory has gaps")
    opening = float(np.nanmean(traj[:OPENING_WINDOW]))
    late = float(np.nanmean(traj[LATE_START - 1:]))
    # How long does it hold the prior?  First trial at which the trajectory has
    # travelled half the way from its opening level to its late level.
    mid = (opening + late) / 2.0
    crossed = np.where((traj > mid) if late > opening else (traj < mid))[0]
    return {
        "opening_pbet": opening,
        "pbet_t1": float(traj[0]),
        "late_pbet": late,
        "midpoint_trial": float(crossed[0] + 1) if len(crossed) else float("nan"),
        "dev_opening_from_cue": opening - float(frame["cue_probability"].iloc[0]),
        "reversal_size": late - opening,
        "traj": traj,
    }


def collect(output_root: Path, hidden: int) -> pd.DataFrame:
    rows = []
    for cue, true, scenario, protocol, group, round_id in CELLS:
        for seed in (20260928, 20260929, 20260930):
            base = output_root / "frozen_tests" / f"h{hidden:02d}_b200" / protocol
            path = base / f"formal_osla_seed{seed}" / group / "trials.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            trials = pd.read_csv(path)
            metrics = round_metrics(trials, round_id)
            rows.append(
                {
                    "hidden_size": hidden,
                    "seed": seed,
                    "group": group,
                    "protocol": protocol,
                    "round_id": round_id,
                    "cue": cue,
                    "true": true,
                    "scenario": scenario,
                    **{k: v for k, v in metrics.items() if k != "traj"},
                    **{f"traj_{i+1}": v for i, v in enumerate(metrics["traj"])},
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prior-injection robustness, single reversal round.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "outputs_prerun")
    args = parser.parse_args()

    apply_style()
    output_root = args.output_root.resolve()
    frames = [collect(output_root, h) for h in HIDDEN_SIZES]
    all_rows = pd.concat(frames, ignore_index=True)
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    all_rows.to_csv(metrics_dir / "prior_injection_single_round.csv", index=False)

    summary = (
        all_rows.groupby(["cue", "true", "scenario"])[
            ["opening_pbet", "pbet_t1", "late_pbet", "midpoint_trial", "dev_opening_from_cue", "reversal_size"]
        ]
        .mean()
        .reset_index()
    )
    print(f"single-round prior-injection test, {len(HIDDEN_SIZES)} hidden sizes x 3 seeds "
          f"= {len(HIDDEN_SIZES) * 3} networks, {EPISODES} episodes per cell\n")
    print(summary.round(4).to_string(index=False))

    def row(cue: float, true: float) -> pd.Series:
        hit = summary[(summary["cue"] == cue) & (summary["true"] == true)]
        if len(hit) != 1:
            raise SystemExit(f"expected exactly one row for cue {cue}/true {true}, got {len(hit)}")
        return hit.iloc[0]

    print("\n1) does it OPEN at the prior's probability?  (opening_pbet vs the cue)")
    for cue in (0.30, 0.70):
        sub = row(cue, cue)
        print(f"   cue {cue:.2f} (no conflict): opening {sub['opening_pbet']:.3f}  "
              f"deviation from the cue {sub['dev_opening_from_cue']:+.3f}")
    print("\n2) does it REVERSE toward the evidence?  (opening -> late)")
    for cue, true in ((0.30, 0.60), (0.70, 0.40)):
        sub = row(cue, true)
        print(f"   cue {cue:.2f} / true {true:.2f}: {sub['opening_pbet']:.3f} -> {sub['late_pbet']:.3f} "
              f"(size {sub['reversal_size']:+.3f}), midpoint crossed at trial {sub['midpoint_trial']:.0f}")

    plot(all_rows, summary, output_root)
    print(f"\nwrote {metrics_dir / 'prior_injection_single_round.csv'}")


def plot(rows: pd.DataFrame, summary: pd.DataFrame, output_root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), sharey=True)
    for ax, cue in zip(axes, (0.30, 0.70)):
        block = rows[rows["cue"] == cue]
        cells = sorted(block["true"].unique())
        for i, true in enumerate(cells):
            cell = block[block["true"] == true]
            traj = cell[[c for c in cell.columns if c.startswith("traj_")]].to_numpy()
            mean = traj.mean(axis=0)
            for line in traj:
                ax.plot(range(1, TRIALS_PER_ROUND + 1), line, color=ORDINAL[i], lw=0.7, alpha=0.30)
            scenario = str(cell["scenario"].iloc[0])
            ax.plot(range(1, TRIALS_PER_ROUND + 1), mean, color=ORDINAL[i], lw=2.0,
                    label=f"true {true:.2f} — {scenario}")
            ax.annotate(f"true {true:.2f}", xy=(TRIALS_PER_ROUND, mean[-20:].mean()),
                        xytext=(4, (i - 1) * 7), textcoords="offset points", fontsize=6.5,
                        color=ORDINAL[i], va="center")
        reference_line(ax, cue)
        ax.annotate(f"cue (prior) {cue:.2f}", xy=(0.02, cue), xytext=(0, 3), textcoords="offset points",
                    fontsize=6.8, color=INK_MUTED)
        ax.set_xticks([1, 20, 40, 60, 80, 100, 120])
        ax.set_xlim(1, TRIALS_PER_ROUND + 12)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("trial within the single round")
        ax.set_title(f"stated prior = {cue:.2f}", fontsize=9, color=INK_PRIMARY)
    axes[0].set_ylabel("P(bet)")
    handles = [
        plt.Line2D([], [], color=ORDINAL[0], lw=2.0, label="true = prior (no conflict)"),
        plt.Line2D([], [], color=ORDINAL[1], lw=2.0, label="true = 0.50 (neutral)"),
        plt.Line2D([], [], color=ORDINAL[2], lw=2.0, label="true = opposite side (reversal)"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.8, ls="--", label="stated prior (cue)"),
    ]
    finish(
        fig,
        title="Prior-injection robustness, ONE round only (blank state): does it open at the prior, then reverse?\n"
              "thin lines = the 6 networks (2 hidden sizes x 3 seeds), thick = their mean",
        legend_handles=handles,
        ncol=4,
    )
    save(fig, output_root / "figures" / "prior_injection_single_round.png")


if __name__ == "__main__":
    main()
