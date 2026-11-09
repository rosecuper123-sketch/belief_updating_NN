"""Split the orthogonal cue x true grid by hidden size, and score the two capacities.

Read-only re-aggregation of `metrics/orthogonal_grid_trajectories.csv` written by
`test_orthogonal_prior_grid.py` -- no new runs, no new checkpoints.

Why this exists: the pooled 9x9 figure in that script averages all 6 networks into
one line per cell.  That is not harmless, because the dominant variance is the
systematic h16-vs-h32 difference and not the seed noise (the three seeds within a
capacity agree far more closely than the two capacities do).  So this script draws
the grid twice -- once per capacity -- keeping the three seed trajectories visible
underneath the capacity mean, and then scores the two capacities on the three things
the task is actually about:

* **prior absorption**  -- how close the opening bet rate (trials 1-5) is to the cue;
* **reversal speed**    -- trials to cross the midpoint when cue and truth disagree;
* **return efficiency** -- cumulative expected return as a fraction of the oracle's.

Expected return is used rather than realised return because the outcome sequence is
held fixed across cues by construction (the matched-feedback check in the generator
script), so realised differences would just be action noise on one shared draw.  Per
trial, E[reward] = P(bet) * (2*true - 1); the return score is normalised between the
oracle policy and the worst policy, so 1 is optimal and 0 is the worst on both sides
of true = 0.5 (see `return_score`).

Unit of analysis is the **network seed** (n = 3 per capacity), so the capacity
comparison is descriptive; the per-cell counts are reported as a robustness read, not
as independent replicates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

from scripts.shared.prerun_style import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    ROOT,
    SERIES,
    SERIES_NAMES,
    apply_style,
    finish,
    panel_label,
    save,
)

GRID = tuple(round(0.1 * i, 1) for i in range(1, 10))
IN_DISTRIBUTION = (0.3, 0.4, 0.5, 0.6, 0.7)
TRIALS_PER_ROUND = 120
OPENING_WINDOW = 5
LATE_START = 101
HIDDEN_SIZES = (16, 32)
SEEDS = (20260928, 20260929, 20260930)

# h16 vs h32 is a comparison of two capacities: two categorical slots, never cycled.
CAPACITY_COLOUR = {16: "#2a78d6", 32: "#eb6834"}
CAPACITY_LABEL = {16: "h = 16", 32: "h = 32"}


# --------------------------------------------------------------------------- metrics


def return_score(pbet_sum: np.ndarray | float, true: float) -> np.ndarray | float:
    """Expected return relative to the oracle, on a [0, 1] scale that is fair on both sides.

    Per trial E[reward] = P(bet) * (2*true - 1).  The oracle bets only when
    ``true > 0.5`` and therefore earns ``max(0, 2*true - 1)`` per trial; the worst
    policy bets always (``true < 0.5``) or never (``true > 0.5``).  Normalising
    against *both* anchors makes the same number mean the same thing above and below
    0.5 -- dividing by ``|2*true - 1|`` on both sides, which is tempting, silently
    scores a below-0.5 cell on a scale it can never reach.

        score = (R - R_worst) / (R_oracle - R_worst),  R = sum_t P(bet)_t * (2*true - 1)
    """
    spread = 2.0 * true - 1.0
    oracle = TRIALS_PER_ROUND * max(0.0, spread)
    worst = TRIALS_PER_ROUND * min(0.0, spread)
    if oracle == worst:  # true == 0.5: any policy earns 0 in expectation
        return float("nan") if np.isscalar(pbet_sum) else np.full_like(
            np.asarray(pbet_sum, dtype=float), np.nan
        )
    return (pbet_sum * spread - worst) / (oracle - worst)


def per_network_cells(traj: pd.DataFrame) -> pd.DataFrame:
    """One row per (hidden_size, seed, cue, true) with the four task quantities."""
    rows = []
    for (hidden, seed, cue, true), block in traj.groupby(
        ["hidden_size", "seed", "cue", "true"], sort=True
    ):
        series = block.sort_values("trial_id")["pbet"].to_numpy(dtype=float)
        if len(series) != TRIALS_PER_ROUND:
            raise SystemExit(f"h{hidden}/seed {seed}/cue {cue}/true {true}: {len(series)} trials")

        opening = float(series[:OPENING_WINDOW].mean())
        late = float(series[LATE_START - 1:].mean())
        true_sign = 1.0 if true > 0.5 else (-1.0 if true < 0.5 else 0.0)

        if true_sign == 0.0:
            crossing = np.nan
            efficiency = np.nan
        else:
            reached = np.sign(series - 0.5) == true_sign
            crossing = float(np.argmax(reached) + 1) if reached.any() else np.nan
            efficiency = return_score(series.sum(), true)

        # "Reversal needed": the cue points one way, the truth the other.
        conflict = (
            abs(cue - 0.5) > 0.05 and abs(true - 0.5) > 0.05
            and np.sign(cue - 0.5) != np.sign(true - 0.5)
        )
        rows.append(
            {
                "hidden_size": int(hidden), "seed": int(seed), "cue": cue, "true": true,
                "opening_pbet": opening, "late_pbet": late, "crossing_trial": crossing,
                "return_efficiency": efficiency, "conflict": bool(conflict),
                "in_distribution": cue in IN_DISTRIBUTION,
                "true_sign": true_sign,
            }
        )
    return pd.DataFrame(rows)


def efficiency_curve(traj: pd.DataFrame, cells: pd.DataFrame, *, conflict_only: bool) -> pd.DataFrame:
    """Cumulative-return score (see `return_score`) per trial, per capacity.

    Each seed gets equal weight: average over its cells first, then over the three seeds.
    """
    keep = cells[np.abs(cells["true"] - 0.5) > 0.05]
    if conflict_only:
        keep = keep[keep["conflict"]]
    merged = traj.merge(keep[["hidden_size", "seed", "cue", "true"]],
                        on=["hidden_size", "seed", "cue", "true"], how="inner")
    merged = merged.sort_values(["hidden_size", "seed", "cue", "true", "trial_id"])
    merged["cum"] = merged.groupby(["hidden_size", "seed", "cue", "true"])["pbet"].cumsum()
    merged["efficiency"] = [
        return_score(total, true)
        for total, true in zip(merged["cum"].to_numpy(), merged["true"].to_numpy())
    ]
    per_seed = merged.groupby(["hidden_size", "seed", "trial_id"])["efficiency"].mean().reset_index()
    return per_seed.groupby(["hidden_size", "trial_id"])["efficiency"].agg(
        ["mean", "std", "count"]
    ).reset_index()


def absorption_slope(cells: pd.DataFrame) -> pd.DataFrame:
    """Least-squares slope of opening P(bet) on cue, per network (ideal = 1.0)."""
    rows = []
    for (hidden, seed), block in cells.groupby(["hidden_size", "seed"]):
        cue = block["cue"].to_numpy(dtype=float)
        opening = block["opening_pbet"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(cue, opening, 1)
        rows.append(
            {
                "hidden_size": int(hidden), "seed": int(seed), "slope": float(slope),
                "intercept": float(intercept),
                "signed_dev": float((opening - cue).mean()),
                "mean_abs_dev": float(np.abs(opening - cue).mean()),
                "max_abs_dev": float(np.abs(opening - cue).max()),
            }
        )
    return pd.DataFrame(rows)


def reversal_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Crossing trial over the conflict cells, per network."""
    conflict = cells[cells["conflict"]]
    rows = []
    for (hidden, seed), block in conflict.groupby(["hidden_size", "seed"]):
        crossing = block["crossing_trial"].to_numpy(dtype=float)
        rows.append(
            {
                "hidden_size": int(hidden), "seed": int(seed),
                "n_conflict_cells": int(len(block)),
                "never_crossed": int(np.isnan(crossing).sum()),
                "median_crossing": float(np.nanmedian(crossing)) if np.isfinite(crossing).any() else np.nan,
                "p90_crossing": float(np.nanpercentile(crossing, 90)) if np.isfinite(crossing).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def oracle_deviation(cells: pd.DataFrame) -> pd.DataFrame:
    """How far the late bet rate ends from the oracle target (1 / 0)."""
    defined = cells[cells["true_sign"] != 0.0].copy()
    defined["target"] = (defined["true_sign"] > 0).astype(float)
    defined["late_dev"] = (defined["late_pbet"] - defined["target"]).abs()
    rows = []
    for (hidden, seed), block in defined.groupby(["hidden_size", "seed"]):
        rows.append(
            {
                "hidden_size": int(hidden), "seed": int(seed),
                "late_abs_dev": float(block["late_dev"].mean()),
                "late_worst_dev": float(block["late_dev"].max()),
                "return_efficiency": float(block["return_efficiency"].mean()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures


def plot_grid_by_capacity(traj: pd.DataFrame, hidden: int, output_root: Path) -> None:
    apply_style()
    n = len(GRID)
    fig, axes = plt.subplots(n, n, figsize=(1.36 * n, 1.30 * n), sharex=True, sharey=True)
    subset = traj[traj["hidden_size"] == hidden]
    for i, cue in enumerate(GRID):
        for j, true in enumerate(GRID):
            ax = axes[i, j]
            cell = subset[(subset["cue"] == cue) & (subset["true"] == true)]
            colour = INK_PRIMARY if cue == true else "#2a78d6"
            for k, seed in enumerate(SEEDS):
                line = cell[cell["seed"] == seed].sort_values("trial_id")
                ax.plot(line["trial_id"], line["pbet"], color=SERIES[k], lw=0.5, alpha=0.55,
                        zorder=2)
            mean = cell.groupby("trial_id")["pbet"].mean()
            ax.plot(mean.index, mean.to_numpy(), color=colour, lw=1.4, zorder=3)
            ax.axhline(cue, color=INK_MUTED, lw=0.6, ls="--", zorder=1)
            if cue != true:
                ax.axhline(true, color=INK_MUTED, lw=0.6, ls="-.", zorder=1)
            ax.set_ylim(-0.04, 1.04)
            ax.set_xlim(0, TRIALS_PER_ROUND)
            ax.set_xticks([1, 60, 120])
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(labelsize=5.2, length=2, pad=1)
            if i == 0:
                ax.set_title(f"true {true:.1f}", fontsize=6.4, color=INK_PRIMARY, pad=3)
            if j == 0:
                ax.set_ylabel(f"cue {cue:.1f}", fontsize=6.4, color=INK_PRIMARY, labelpad=2)
            if i == n - 1:
                ax.set_xlabel("trial", fontsize=5.6, labelpad=1)
            if cue not in IN_DISTRIBUTION or true not in IN_DISTRIBUTION:
                ax.set_facecolor("#f4f3ef")

    handles = [
        plt.Line2D([], [], color=INK_PRIMARY, lw=1.6, label="cue == true"),
        plt.Line2D([], [], color="#2a78d6", lw=1.6, label="cue != true"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.6, ls="--", label="stated prior (cue)"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.6, ls="-.", label="true probability"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#f4f3ef", edgecolor="none",
                      label="cue outside the training grid"),
        *[plt.Line2D([], [], color=SERIES[k], lw=0.9, alpha=0.7, label=SERIES_NAMES[k])
          for k in range(len(SEEDS))],
    ]
    finish(
        fig,
        title=f"Orthogonal cue x true grid, one round from a blank state — {CAPACITY_LABEL[hidden]}\n"
              f"bold = capacity mean of the 3 seeds, thin = each seed; 9 x 9 cues/truths, "
              f"50 episodes per cell",
        legend_handles=handles, ncol=8,
    )
    save(fig, output_root / "figures" / f"orthogonal_grid_9x9_h{hidden}.png")


def plot_assessment(cells: pd.DataFrame, curves: pd.DataFrame, output_root: Path) -> None:
    """Four panels: absorption, convergence, reversal cost, reversal speed."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.2))

    # --- (A) prior absorption: opening P(bet) vs cue --------------------------
    ax = axes[0, 0]
    grid = np.array(GRID)
    ax.plot(grid, grid, color=INK_MUTED, lw=0.9, ls=":", zorder=1, label="faithful reading (y = cue)")
    for hidden in HIDDEN_SIZES:
        block = cells[cells["hidden_size"] == hidden]
        for k, seed in enumerate(SEEDS):
            seed_block = block[block["seed"] == seed].groupby("cue")["opening_pbet"].mean()
            ax.scatter(seed_block.index, seed_block.to_numpy(), s=7, color=CAPACITY_COLOUR[hidden],
                       alpha=0.30, zorder=2, linewidths=0)
        mean = block.groupby("cue")["opening_pbet"].mean()
        sem = block.groupby("cue")["opening_pbet"].sem()
        ax.errorbar(mean.index, mean.to_numpy(), yerr=sem.to_numpy(), color=CAPACITY_COLOUR[hidden],
                    lw=1.8, marker="o", ms=3.6, capsize=2, zorder=3, label=CAPACITY_LABEL[hidden])
    ax.axvspan(0.25, 0.75, color="#f4f3ef", zorder=0)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("stated prior (cue)")
    ax.set_ylabel("opening P(bet), trials 1-5")
    ax.set_title("Prior absorption", fontsize=9, color=INK_PRIMARY)
    panel_label(ax, "grey band = seen in training", loc="upper left", box=True)
    ax.legend(loc="lower right")

    # --- (B) end-of-round convergence to the oracle --------------------------
    ax = axes[0, 1]
    defined = cells[cells["true_sign"] != 0.0]
    trues = sorted(defined["true"].unique())
    oracle = [1.0 if t > 0.5 else 0.0 for t in trues]
    ax.plot(trues, oracle, color=INK_MUTED, lw=1.6, ls="--", zorder=1,
            label="oracle (bet iff true > 0.5)")
    for hidden in HIDDEN_SIZES:
        block = defined[defined["hidden_size"] == hidden]
        mean = block.groupby("true")["late_pbet"].mean()
        sem = block.groupby("true")["late_pbet"].sem()
        ax.errorbar(mean.index, mean.to_numpy(), yerr=sem.to_numpy(), color=CAPACITY_COLOUR[hidden],
                    lw=1.8, marker="o", ms=3.6, capsize=2, zorder=3, label=CAPACITY_LABEL[hidden])
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("true probability")
    ax.set_ylabel("late P(bet), trials 101-120")
    ax.set_title("End-of-round convergence", fontsize=9, color=INK_PRIMARY)
    ax.legend(loc="lower right")

    # --- (C) expected-return time course, conflict cells only -----------------
    ax = axes[1, 0]
    for hidden in HIDDEN_SIZES:
        agg = curves[curves["hidden_size"] == hidden].sort_values("trial_id")
        ax.plot(agg["trial_id"], agg["mean"], color=CAPACITY_COLOUR[hidden], lw=1.8,
                label=CAPACITY_LABEL[hidden])
        ax.fill_between(agg["trial_id"], agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=CAPACITY_COLOUR[hidden], alpha=0.14, linewidth=0)
    ax.axhline(1.0, color=INK_MUTED, lw=0.9, ls="--", zorder=1)
    ax.axhline(0.0, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax.set_xlim(1, TRIALS_PER_ROUND)
    ax.set_xlabel("trial within the round")
    ax.set_ylabel("cumulative return score")
    ax.set_title("Reversal cost: return recovered within the round", fontsize=9, color=INK_PRIMARY)
    panel_label(ax, "0 = worst policy, 1 = oracle\n32 conflict cells, cumulative to trial t",
                loc="lower right", box=True)
    ax.legend(loc="upper left")

    # --- (D) reversal speed ---------------------------------------------------
    ax = axes[1, 1]
    conflict = cells[cells["conflict"] & np.isfinite(cells["crossing_trial"])]
    groups = sorted(conflict["true"].unique())
    for offset, hidden in (( -0.16, 16), (0.16, 32)):
        block = conflict[conflict["hidden_size"] == hidden]
        for gi, true in enumerate(groups):
            values = block[block["true"] == true]["crossing_trial"].to_numpy(dtype=float)
            jitter = np.linspace(-0.05, 0.05, len(values)) if len(values) else values
            ax.scatter(np.full(len(values), gi + offset) + jitter, values, s=22,
                       color=CAPACITY_COLOUR[hidden], alpha=0.55, linewidths=0, zorder=3)
            ax.scatter([gi + offset], [np.median(values)], s=54, marker="_",
                       color=CAPACITY_COLOUR[hidden], zorder=4, linewidths=2.2,
                       label=CAPACITY_LABEL[hidden] if gi == 0 else None)
    never = cells[cells["conflict"] & ~np.isfinite(cells["crossing_trial"])]
    if len(never):
        panel_label(ax, f"{len(never)} of {int(cells['conflict'].sum())} network-cells never cross",
                    loc="lower left", box=True)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{t:.1f}" for t in groups])
    ax.set_xlabel("true probability")
    ax.set_ylabel("trial crossing the midpoint")
    ax.set_ylim(0, TRIALS_PER_ROUND + 6)
    ax.set_title("Reversal speed (conflict cells)", fontsize=9, color=INK_PRIMARY)
    panel_label(ax, "4 cues x 3 seeds per group\nbar = median", loc="upper left", box=True)
    ax.legend(loc="upper right")

    finish(
        fig,
        title="Do the two capacities differ on the orthogonal task?  h16 vs h32, "
              "3 seeds each, 50 episodes per cell\n"
              "n = 3 network seeds per capacity — descriptive; error bars are s.e.m. across the "
              "3 seeds, shaded band is ±1 s.d. across seeds",
        legend_handles=None,
    )
    save(fig, output_root / "figures" / "orthogonal_capacity_assessment.png")


def plot_prior_footprint(cells: pd.DataFrame, output_root: Path) -> None:
    """The true = 0.5 column per network -- the one place the prior survives the round.

    Drawn per network rather than per capacity because this is exactly where the
    capacity framing breaks down: the within-capacity spread across the three seeds
    is much wider than the gap between the two capacities.
    """
    apply_style()
    neutral = cells[cells["true"] == 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9), sharey=True)
    for ax, hidden in zip(axes, HIDDEN_SIZES):
        block = neutral[neutral["hidden_size"] == hidden]
        for k, seed in enumerate(SEEDS):
            line = block[block["seed"] == seed].sort_values("cue")
            opening = line["opening_pbet"].to_numpy()
            late = line["late_pbet"].to_numpy()
            retention = (late[-1] - late[0]) / (opening[-1] - opening[0])
            ax.plot(line["cue"], late, color=SERIES[k], lw=1.8, marker="o", ms=3.4,
                    label=f"{SERIES_NAMES[k]}  ({retention:.0%} kept)")
            ax.plot(line["cue"], opening, color=SERIES[k], lw=1.0, ls=":", alpha=0.75)
        ax.plot(GRID, GRID, color=INK_MUTED, lw=0.9, ls="--", zorder=1)
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("stated prior (cue)")
        ax.set_title(f"{CAPACITY_LABEL[hidden]} — late P(bet) at true = 0.5",
                     fontsize=9, color=INK_PRIMARY)
        ax.legend(loc="upper left", fontsize=7.5)
    axes[0].set_ylabel("P(bet), trials 101-120")
    finish(
        fig,
        title="Prior footprint at true = 0.5, per network: solid = end of round, dotted = opening\n"
              "The three seeds inside a capacity differ more than the two capacities do — the "
              "poorest seed is 20260929 in both",
        legend_handles=None,
    )
    save(fig, output_root / "figures" / "orthogonal_prior_footprint_by_network.png")


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the orthogonal grid by hidden size.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "outputs_prerun")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    metrics_dir = output_root / "metrics"

    traj = pd.read_csv(metrics_dir / "orthogonal_grid_trajectories.csv")
    if sorted(traj["hidden_size"].unique()) != list(HIDDEN_SIZES):
        raise SystemExit(f"unexpected hidden sizes: {sorted(traj['hidden_size'].unique())}")
    if sorted(traj["seed"].unique()) != list(SEEDS):
        raise SystemExit(f"unexpected seeds: {sorted(traj['seed'].unique())}")
    if len(traj) != len(GRID) ** 2 * len(HIDDEN_SIZES) * len(SEEDS) * TRIALS_PER_ROUND:
        raise SystemExit(f"{len(traj)} rows; the grid is not complete")

    cells = per_network_cells(traj)
    cells.to_csv(metrics_dir / "orthogonal_capacity_per_network.csv", index=False)
    curves = efficiency_curve(traj, cells, conflict_only=True)
    curves.to_csv(metrics_dir / "orthogonal_capacity_return_curve.csv", index=False)

    absorption = absorption_slope(cells)
    reversal = reversal_summary(cells)
    convergence = oracle_deviation(cells)
    summary = absorption.merge(reversal, on=["hidden_size", "seed"]).merge(
        convergence, on=["hidden_size", "seed"]
    )
    summary.to_csv(metrics_dir / "orthogonal_capacity_summary.csv", index=False)

    print("per-network scores (unit of analysis = one network seed)\n")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(summary.to_string(index=False))

    print("\ncapacity means (n = 3 seeds each)")
    means = summary.groupby("hidden_size")[
        ["slope", "signed_dev", "mean_abs_dev", "max_abs_dev", "median_crossing", "p90_crossing",
         "late_abs_dev", "return_efficiency"]
    ].mean().round(4)
    means["never_crossed"] = summary.groupby("hidden_size")["never_crossed"].sum()
    print(means.to_string())

    print("\npaired across the 81 cue x true cells (capacity mean per cell, both capacities):")
    wide = cells.pivot_table(index=["cue", "true"], columns="hidden_size",
                             values=["opening_pbet", "late_pbet", "return_efficiency"],
                             aggfunc="mean")
    for quantity in ("opening_pbet", "late_pbet", "return_efficiency"):
        diff = wide[(quantity, 16)] - wide[(quantity, 32)]
        print(f"  {quantity:18s} mean h16-h32 = {diff.mean():+.4f}   "
              f"h16 larger in {int((diff > 0).sum())}/{len(diff)} cells")

    plot_grid_by_capacity(traj, 16, output_root)
    plot_grid_by_capacity(traj, 32, output_root)
    plot_assessment(cells, curves, output_root)
    plot_prior_footprint(cells, output_root)

    print("\nretention of the true = 0.5 footprint (late span / opening span), per network:")
    neutral = cells[cells["true"] == 0.5]
    for (hidden, seed), block in neutral.groupby(["hidden_size", "seed"]):
        line = block.sort_values("cue")
        opening = line["opening_pbet"].to_numpy()
        late = line["late_pbet"].to_numpy()
        print(f"  h{hidden} seed {seed}: opening span {opening[-1] - opening[0]:.3f} -> "
              f"late span {late[-1] - late[0]:.3f}   kept "
              f"{(late[-1] - late[0]) / (opening[-1] - opening[0]):.0%}")

    print(f"\nwrote {metrics_dir / 'orthogonal_capacity_summary.csv'}")


if __name__ == "__main__":
    main()
