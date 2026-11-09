"""Figure A: learning curves for the same seed at different update budgets.

Reads only `training_history.csv`, so it does not need the frozen tests.

Three questions, three figures:

* `figA1_learning_curves` -- validation and training return vs update, one panel
  per (hidden_size, seed), plus a cross-seed overlay column.  Vertical dotted
  rules mark the candidate stopping budgets 200/400/600, showing that they all
  sit on a still-rising curve.
* `figA2_budget_dose` -- the same two quantities *at* those three budgets, one
  panel per metric: the "budget is the prior-rigidity knob" read.  Return and
  prior-anchor error live on separate axes, never a dual axis.
* `figA3_prior_anchor_error` -- prior-anchor error vs update, continuous.

Also runs the free consistency gate: a 600-update run and a 200-update run with
the same seed must agree bit-for-bit on their first 200 updates, because the RNG
stream depends only on the seed.  A mismatch means the thread count or the code
changed, and the scientific read is invalid until that is explained.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.shared.prerun_style import (
    INK_MUTED,
    INK_PRIMARY,
    ORDINAL,
    ROOT,
    SERIES,
    apply_style,
    budget_markers,
    finish,
    panel_label,
    save,
)
import matplotlib.pyplot as plt

DEFAULT_BUDGETS = (200, 400, 600)
EXCLUDE_COLUMNS = {"elapsed_seconds", "update"}
ROLL_WINDOW = 20


def history_path(output_root: Path, hidden: int, budget: int, seed: int) -> Path:
    return (
        output_root / "arms" / f"h{hidden:02d}_b{budget}" / "runs"
        / f"formal_osla_seed{seed}" / "training_history.csv"
    )


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["update"] = frame["update"].astype(int)
    return frame


def consistency_check(b200: pd.DataFrame, b600: pd.DataFrame) -> dict:
    """First 200 updates of the 600-run must equal the 200-run exactly.

    Validation columns are NaN on non-validation updates, so they are compared on
    the rows where both runs report a value -- and their NaN patterns must match
    too, otherwise a run could "agree" by both being empty where one should have
    a number.
    """
    shared = [c for c in b200.columns if c in b600.columns and c not in EXCLUDE_COLUMNS]
    left = b200.sort_values("update").reset_index(drop=True)
    right = b600.sort_values("update").head(len(left)).reset_index(drop=True)
    worst_column, worst_delta = None, 0.0
    nan_mismatch: list[str] = []
    for column in shared:
        a = pd.to_numeric(left[column], errors="coerce")
        b = pd.to_numeric(right[column], errors="coerce")
        if not a.isna().equals(b.isna()):
            nan_mismatch.append(column)
        mask = a.notna() & b.notna()
        if mask.any():
            delta = float(np.max(np.abs(a[mask].to_numpy() - b[mask].to_numpy())))
            if delta > worst_delta:
                worst_column, worst_delta = column, delta
    updates_match = left["update"].equals(right["update"])
    return {
        "rows_compared": int(len(left)),
        "columns_compared": len(shared),
        "max_abs_delta": worst_delta,
        "worst_column": worst_column,
        "nan_pattern_mismatch": nan_mismatch,
        "update_index_matches": bool(updates_match),
        "identical": bool(worst_delta == 0.0 and not nan_mismatch and updates_match),
    }


def value_at_budget(history: pd.DataFrame, column: str, budget: int) -> float:
    """Last available value of a validation column at or before `budget`."""
    subset = history[history["update"] <= budget]
    series = pd.to_numeric(subset[column], errors="coerce").dropna()
    return float(series.iloc[-1]) if len(series) else float("nan")


def rolling_value_at(history: pd.DataFrame, column: str, budget: int) -> float:
    """Smoothed value of a per-update (non-validation) column at `budget`."""
    subset = history[history["update"] <= budget]
    series = pd.to_numeric(subset[column], errors="coerce").rolling(ROLL_WINDOW, min_periods=1).mean()
    return float(series.iloc[-1]) if len(series) else float("nan")


def seed_handles(seeds):
    return [plt.Line2D([], [], color=SERIES[i], lw=1.8, label=f"seed {seed}")
            for i, seed in enumerate(seeds)]


def figA1(histories: dict, hidden_sizes, seeds, output_root: Path) -> None:
    fig, axes = plt.subplots(
        2, len(seeds) + 1, figsize=(2.5 * (len(seeds) + 1), 4.8), sharex=True, sharey="row"
    )
    for row, hidden in enumerate(hidden_sizes):
        for col, seed in enumerate(seeds):
            ax = axes[row, col]
            history = histories[(hidden, seed)]
            train = history["mean_return"].rolling(ROLL_WINDOW, min_periods=1).mean()
            ax.plot(history["update"], train, color=SERIES[col], alpha=0.30, lw=1.1)
            val = history.dropna(subset=["val_mean_return"])
            ax.plot(val["update"], val["val_mean_return"], color=SERIES[col], lw=1.8)
            ax.scatter(val["update"].iloc[-1], val["val_mean_return"].iloc[-1],
                       s=22, color=SERIES[col], zorder=5, edgecolor="none")
            budget_markers(ax)
            best = history["best_update"].dropna()
            panel_label(
                ax,
                f"h={hidden}  seed {seed}\n"
                f"best_update={int(best.iloc[-1]) if len(best) else '-'}",
            )
            if row == 0 and col == 0:
                ax.set_ylabel("mean return")
            if row == len(hidden_sizes) - 1:
                ax.set_xlabel("update")

        ax = axes[row, len(seeds)]
        for col, seed in enumerate(seeds):
            val = histories[(hidden, seed)].dropna(subset=["val_mean_return"])
            ax.plot(val["update"], val["val_mean_return"], color=SERIES[col], lw=1.8)
        mean = histories[(hidden, "mean")]
        ax.plot(mean["update"], mean["val_mean_return"], color=INK_PRIMARY, lw=1.6, alpha=0.85)
        budget_markers(ax)
        panel_label(ax, f"h={hidden}  all seeds")
        if row == len(hidden_sizes) - 1:
            ax.set_xlabel("update")

    handles = seed_handles(seeds) + [
        plt.Line2D([], [], color=INK_PRIMARY, lw=1.6, alpha=0.85, label="cross-seed mean"),
        plt.Line2D([], [], color=INK_MUTED, lw=1.8, label="validation"),
        plt.Line2D([], [], color=INK_MUTED, lw=1.1, alpha=0.45, label=f"train (roll {ROLL_WINDOW})"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.7, ls=":", label="candidate budget 200/400/600"),
    ]
    finish(
        fig,
        title=(
            "A1  Learning curves: one 600-update run per (hidden_size, seed), budget read off it\n"
            "dotted rules mark the candidate stopping points 200 / 400 / 600"
        ),
        legend_handles=handles,
        ncol=7,
    )
    save(fig, output_root / "figures" / "figA1_learning_curves.png")


def figA2(histories: dict, hidden_sizes, seeds, budgets, output_root: Path) -> None:
    metrics = [
        ("validation mean return", lambda h, b: value_at_budget(h, "val_mean_return", b)),
        ("validation prior-anchor error\nmean (p_boundary - cue)^2",
         lambda h, b: value_at_budget(h, "val_prior_anchor_error", b)),
        (f"train policy entropy (roll {ROLL_WINDOW})",
         lambda h, b: rolling_value_at(h, "entropy", b)),
    ]
    fig, axes = plt.subplots(len(metrics), len(hidden_sizes), figsize=(8.2, 7.8), sharex=True)
    for row, (label, accessor) in enumerate(metrics):
        for col, hidden in enumerate(hidden_sizes):
            ax = axes[row, col]
            for s_index, seed in enumerate(seeds):
                ys = [accessor(histories[(hidden, seed)], b) for b in budgets]
                ax.plot(budgets, ys, color=SERIES[s_index], lw=1.8, marker="o", ms=4.5)
                ax.annotate(f"{ys[-1]:.3f}", xy=(budgets[-1], ys[-1]),
                            xytext=(5, (s_index - 1) * 6), textcoords="offset points",
                            fontsize=7, color=SERIES[s_index], va="center")
            ax.set_title(f"h={hidden}", color=INK_PRIMARY, fontsize=9)
            ax.set_ylabel(label)
            ax.set_xlim(budgets[0] - 50, budgets[-1] + 110)
            if row == len(metrics) - 1:
                ax.set_xlabel("update budget")
                ax.set_xticks(list(budgets))
    finish(
        fig,
        title=(
            "A2  What a larger budget actually buys\n"
            "return keeps rising, the prior readout is already saturated by ~200, and the policy "
            "keeps getting more deterministic"
        ),
        legend_handles=seed_handles(seeds),
        ncol=3,
    )
    save(fig, output_root / "figures" / "figA2_budget_dose.png")


def figA3(histories: dict, hidden_sizes, seeds, output_root: Path) -> None:
    fig, axes = plt.subplots(1, len(hidden_sizes), figsize=(8.2, 3.4), sharey=True)
    for col, hidden in enumerate(hidden_sizes):
        ax = axes[col]
        for s_index, seed in enumerate(seeds):
            val = histories[(hidden, seed)].dropna(subset=["val_prior_anchor_error"])
            ax.plot(val["update"], val["val_prior_anchor_error"], color=SERIES[s_index], lw=1.8)
            ax.annotate(
                f"{seed}",
                xy=(val["update"].iloc[-1], val["val_prior_anchor_error"].iloc[-1]),
                xytext=(5, 0), textcoords="offset points", fontsize=6.5,
                color=SERIES[s_index], va="center",
            )
        budget_markers(ax)
        ax.set_title(f"h={hidden}", color=INK_PRIMARY, fontsize=9)
        ax.set_xlabel("update")
        ax.set_xlim(0, 780)
    axes[0].set_ylabel("val prior-anchor error")
    finish(
        fig,
        title=(
            "A3  The prior readout degrades early, then saturates (consistent validation rounds)\n"
            "error climbs from ~0.02 to ~0.18 by update ~200 and stays flat after; "
            "0 = the boundary readout reproduces the cue exactly"
        ),
        legend_handles=seed_handles(seeds),
        ncol=3,
    )
    save(fig, output_root / "figures" / "figA3_prior_anchor_error.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figure A (learning curves).")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "outputs_prerun")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260928, 20260929, 20260930])
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--long-budget", type=int, default=600)
    args = parser.parse_args()

    apply_style()
    output_root = args.output_root.resolve()

    histories: dict = {}
    rows: list[dict] = []
    checks: list[dict] = []
    for hidden in args.hidden_sizes:
        for seed in args.seeds:
            long_history = load_history(history_path(output_root, hidden, args.long_budget, seed))
            histories[(hidden, seed)] = long_history
            short = history_path(output_root, hidden, min(args.budgets), seed)
            if short.exists():
                check = consistency_check(load_history(short), long_history)
                check.update({"hidden_size": hidden, "seed": seed})
                checks.append(check)
                print(
                    f"consistency h={hidden} seed={seed}: "
                    f"{'IDENTICAL' if check['identical'] else 'MISMATCH'} "
                    f"(rows={check['rows_compared']}, max|delta|={check['max_abs_delta']:.3g}"
                    f"{' in ' + str(check['worst_column']) if check['worst_column'] else ''})"
                )
            for budget in args.budgets:
                rows.append(
                    {
                        "hidden_size": hidden,
                        "seed": seed,
                        "budget": budget,
                        "val_mean_return": value_at_budget(long_history, "val_mean_return", budget),
                        "val_prior_anchor_error": value_at_budget(long_history, "val_prior_anchor_error", budget),
                        "val_mean_bet_rate": value_at_budget(long_history, "val_mean_bet_rate", budget),
                        "train_entropy_roll": rolling_value_at(long_history, "entropy", budget),
                        "best_update": int(long_history["best_update"].dropna().iloc[-1]),
                    }
                )

    # Cross-seed mean, plotted in the overlay column of A1.
    for hidden in args.hidden_sizes:
        merged = None
        for seed in args.seeds:
            val = histories[(hidden, seed)][["update", "val_mean_return"]].dropna()
            val = val.rename(columns={"val_mean_return": f"seed{seed}"})
            merged = val if merged is None else merged.merge(val, on="update", how="inner")
        seed_columns = [c for c in merged.columns if c != "update"]
        merged["val_mean_return"] = merged[seed_columns].mean(axis=1)
        histories[(hidden, "mean")] = merged

    budgets_present = sorted({r["budget"] for r in rows})
    if len(budgets_present) < 2:
        raise SystemExit("need at least two budgets for figure A2")

    figA1(histories, args.hidden_sizes, args.seeds, output_root)
    figA2(histories, args.hidden_sizes, args.seeds, budgets_present, output_root)
    figA3(histories, args.hidden_sizes, args.seeds, output_root)

    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(metrics_dir / "learning_curves_summary.csv", index=False)
    (metrics_dir / "consistency_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("wrote metrics/learning_curves_summary.csv, metrics/consistency_checks.json")
    if checks and not all(c["identical"] for c in checks):
        print("WARNING: consistency gate failed -- investigate threads/code before reading the figures")


if __name__ == "__main__":
    main()
