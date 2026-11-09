"""Training curves for the formal batch (30 seeds, one arm).

Reads `arms/h<hh>_b<budget>/runs/formal_osla_seed*/training_history.csv`.

The pre-run's `plot_learning_curves.py` is NOT reusable here and must not be bent to
fit: it hard-binds `--long-budget 600` and loads that arm unconditionally, its
consistency gate compares two budgets, figure A2 exits unless >=2 budgets exist, and
it indexes `prerun_style.SERIES` per seed -- 3 slots, so 30 seeds raise IndexError.
This batch has one arm and 30 seeds, so it gets its own script.  The pre-run script
stays frozen, recording the pre-run's own cut.

Two things the figure must not imply:

* **Validation points are 20 samples, not a curve.**  Validation runs every 10
  updates, so those panels get markers at the sampled updates and no interpolation
  between them.
* **There is no validation entropy.**  `evaluate_validation` returns only
  `mean_return`, `mean_bet_rate`, `prior_anchor_error`, `score`.  The entropy panel
  is *training* entropy and is labelled as such.
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
    ORDINAL,
    apply_style,
    finish,
    panel_label,
    save,
)

DEFAULT_SEEDS = tuple(range(20260804, 20260834))
UPDATES = 200
VALIDATION_EVERY = 10

# (column, panel title, y label, is per-update or sampled at validation points)
PANELS = (
    ("loss", "training loss", "REINFORCE loss", "per_update"),
    ("entropy", "training entropy", "policy entropy (nats)", "per_update"),
    ("mean_return", "training return", "mean return per episode", "per_update"),
    ("mean_bet_rate", "training bet rate", "P(bet)", "per_update"),
    ("val_mean_return", "validation return", "mean return per episode", "sampled"),
    ("val_prior_anchor_error", "validation prior-anchor error", "anchor error", "sampled"),
)


def history_path(out_root: Path, hidden: int, budget: int, seed: int) -> Path:
    return (
        out_root / "arms" / f"h{hidden:02d}_b{budget}" / "runs"
        / f"formal_osla_seed{seed}" / "training_history.csv"
    )


def load_histories(out_root: Path, hidden: int, budget: int, seeds) -> dict[int, pd.DataFrame]:
    histories = {}
    for seed in seeds:
        path = history_path(out_root, hidden, budget, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if len(frame) != UPDATES:
            raise SystemExit(f"{path}: {len(frame)} rows, expected {UPDATES}")
        histories[seed] = frame
    if len(histories) != len(seeds):
        raise SystemExit(f"loaded {len(histories)} of {len(seeds)} training histories")
    return histories


def plot_curves(histories: dict[int, pd.DataFrame], analysis_root: Path, hidden: int, budget: int) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.2))
    seeds = sorted(histories)

    for ax, (column, title, ylabel, kind) in zip(axes.ravel(), PANELS):
        stacked = []
        for seed in seeds:
            frame = histories[seed]
            values = frame[column].to_numpy(dtype=float)
            updates = frame["update"].to_numpy(dtype=float)
            ok = np.isfinite(values)
            # 30 networks cannot take 30 categorical colours (SERIES has 3 slots and
            # the dataviz rule forbids cycling), so identity is carried by position,
            # not by hue: one neutral family, one bold mean.
            ax.plot(updates[ok], values[ok], color=INK_SECONDARY, lw=0.55, alpha=0.22, zorder=2)
            filled = np.full(len(frame), np.nan)
            filled[ok] = values[ok]
            stacked.append(filled)
        stack = np.vstack(stacked)
        updates = next(iter(histories.values()))["update"].to_numpy(dtype=float)
        # Reduce only over updates where at least one network recorded a value, so a
        # sampled panel does not average 30 columns of pure NaN (which warns and is
        # discarded anyway).
        observed = np.isfinite(stack).any(axis=0)
        mean = np.full(len(updates), np.nan)
        sd = np.full(len(updates), np.nan)
        mean[observed] = np.nanmean(stack[:, observed], axis=0)
        sd[observed] = np.nanstd(stack[:, observed], axis=0)

        if kind == "sampled":
            sampled = np.isfinite(mean)
            ax.fill_between(updates, mean - sd, mean + sd, color=ORDINAL[1], alpha=0.16, linewidth=0)
            ax.plot(updates[sampled], mean[sampled], color=ORDINAL[1], lw=1.8, marker="o", ms=3.0,
                    zorder=4, label=f"mean of {len(seeds)} networks")
            panel_label(ax, f"validation every {VALIDATION_EVERY} updates\n"
                            f"({int(sampled.sum())} sampled points, no interpolation)",
                        loc="lower right", box=True)
        else:
            ax.fill_between(updates, mean - sd, mean + sd, color=ORDINAL[1], alpha=0.16, linewidth=0)
            ax.plot(updates, mean, color=ORDINAL[1], lw=1.8, zorder=4,
                    label=f"mean of {len(seeds)} networks")

        ax.set_title(title, fontsize=9, color=INK_PRIMARY)
        ax.set_xlabel("update")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, UPDATES)

    axes[0, 0].legend(loc="upper right")
    finish(
        fig,
        title=f"Formal batch training curves — {len(seeds)} networks, h = {hidden}, {budget} updates\n"
              f"thin grey = each network, bold = mean, band = ±1 s.d. across networks; "
              f"entropy is TRAINING entropy (no validation entropy is recorded)",
        legend_handles=None,
    )
    save(fig, analysis_root / "figures" / f"formal_training_curves_h{hidden:02d}_b{budget}.png")


def plot_outcome_summary(histories: dict[int, pd.DataFrame], analysis_root: Path,
                         hidden: int, budget: int) -> pd.DataFrame:
    rows = []
    for seed, frame in histories.items():
        validation = frame.dropna(subset=["val_mean_return"])
        best_row = validation.loc[validation["val_mean_return"].idxmax()]
        rows.append({
            "seed": seed,
            "best_update": int(best_row["update"]),
            "best_val_return": float(best_row["val_mean_return"]),
            "final_val_return": float(validation["val_mean_return"].iloc[-1]),
            "final_train_entropy": float(frame["entropy"].dropna().iloc[-1]),
            "final_mean_return": float(frame["mean_return"].dropna().iloc[-1]),
        })
    summary = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6))

    ax = axes[0]
    counts = summary["best_update"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=ORDINAL[1], width=0.55)
    for position, (value, count) in enumerate(counts.items()):
        ax.text(position, count + 0.4, str(count), ha="center", va="bottom",
                fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel("update at the best validation return")
    ax.set_ylabel("networks")
    ax.set_title("Checkpoint actually used", fontsize=9, color=INK_PRIMARY)
    panel_label(ax, "best.pt is the argmax over the first 200 updates,\nnot the final update",
                loc="upper left", box=True)

    ax = axes[1]
    values = summary["best_val_return"].to_numpy()
    ax.scatter(np.zeros_like(values) + np.linspace(-0.08, 0.08, len(values)), values,
               s=26, color=ORDINAL[1], alpha=0.75, linewidths=0)
    ax.scatter([0], [np.median(values)], marker="_", s=260, color=INK_PRIMARY, zorder=4, linewidths=2.2)
    outlier = summary.loc[summary["best_val_return"].idxmin()]
    ax.annotate(f"seed {int(outlier['seed'])}\n{outlier['best_val_return']:.1f}",
                xy=(0.06, outlier["best_val_return"]), xytext=(0.28, outlier["best_val_return"] + 1.0),
                fontsize=7.5, color=INK_SECONDARY,
                arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 0.8})
    ax.set_xticks([])
    ax.set_xlim(-0.35, 0.75)
    ax.set_ylabel("best validation return")
    ax.set_title(f"Validation quality (n = {len(summary)})", fontsize=9, color=INK_PRIMARY)

    ax = axes[2]
    ax.scatter(summary["final_train_entropy"], summary["best_val_return"],
               s=26, color=ORDINAL[1], alpha=0.75, linewidths=0)
    ax.set_xlabel("final training entropy (nats)")
    ax.set_ylabel("best validation return")
    ax.set_title("Return vs. policy collapse", fontsize=9, color=INK_PRIMARY)

    finish(
        fig,
        title="Formal batch training outcome — 30 networks, h = 16, 200 updates",
        legend_handles=None,
    )
    save(fig, analysis_root / "figures" / f"formal_training_outcome_h{hidden:02d}_b{budget}.png")
    return summary


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
    (analysis_root / "metrics").mkdir(parents=True, exist_ok=True)

    histories = load_histories(out_root, args.hidden_size, args.budget, args.seeds)
    print(f"training histories: {len(histories)} networks x {UPDATES} updates")

    summary = plot_outcome_summary(histories, analysis_root, args.hidden_size, args.budget)
    summary.to_csv(
        analysis_root / "metrics" / f"formal_training_summary_h{args.hidden_size:02d}_b{args.budget}.csv",
        index=False,
    )
    print(f"\nbest_update distribution: "
          f"{summary['best_update'].value_counts().sort_index().to_dict()}")
    print(f"best validation return: mean {summary['best_val_return'].mean():.2f} "
          f"sd {summary['best_val_return'].std():.2f} "
          f"min {summary['best_val_return'].min():.2f} (seed "
          f"{int(summary.loc[summary['best_val_return'].idxmin(), 'seed'])}) "
          f"max {summary['best_val_return'].max():.2f}")
    print(f"final training entropy: mean {summary['final_train_entropy'].mean():.3f} "
          f"sd {summary['final_train_entropy'].std():.3f}")

    plot_curves(histories, analysis_root, args.hidden_size, args.budget)


if __name__ == "__main__":
    main()
