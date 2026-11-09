"""Orthogonal cue x true grid (0.1-0.9) on the frozen networks, single round, trajectories.

Extends the earlier single-round prior-injection test to the full orthogonal design
the project's cue-evidence diagnostic calls for
(`docs/superpowers/specs/2026-09-08-cue-evidence-orthogonal-diagnostic-design.md`):
every stated prior (cue) 0.1..0.9 is crossed with every true probability 0.1..0.9,
one round only, so each cell starts from a **blank hidden state** and nothing is
carried in.  The deliverable is the betting-rate *trajectory* (120 trials) per cell,
not just its endpoints.

Two design points worth stating because they are what makes the grid readable:

* **Matched feedback along the cue axis.** `run_frozen_tests` seeds its generator
  from the `seed` argument and the number of RNG draws per round depends only on
  `true_probability` (the block-balanced outcome count) and on the fixed
  batch x trials action draws -- not on the cue.  Calling it once per cell with the
  same seed therefore replays the *same outcome sequence* for every cue at a given
  truth, so a row of the grid differs only in the stated prior.  The script asserts
  this rather than assuming it.
* **Cues 0.1/0.2/0.8/0.9 are out of distribution.** Training drew cues only from
  {0.3,0.4,0.5,0.6,0.7}; those four rows/columns are extrapolation and are marked
  as such on the figures.

The cue==true diagonal is entered as a `consistent` round and everything else as a
`conflict` round, because `validate_test_rounds` distinguishes them by exactly that.
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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from scripts.shared.prerun_style import INK_MUTED, INK_PRIMARY, INK_SECONDARY, ROOT, apply_style, finish, save  # noqa: E402

from formal_osla.test_runner import run_frozen_tests  # noqa: E402

GRID = tuple(round(0.1 * i, 1) for i in range(1, 10))  # 0.1 .. 0.9
IN_DISTRIBUTION = (0.3, 0.4, 0.5, 0.6, 0.7)  # the training grid
TRIALS_PER_ROUND = 120
EPISODES = 50  # matches the project's earlier cue x evidence diagnostic
TEST_SEED = 20261107
HIDDEN_SIZES = (16, 32)
SEEDS = (20260928, 20260929, 20260930)
OPENING_WINDOW = 5
LATE_START = 101
# Any valid group key works: in `custom` mode the group's preset values are ignored,
# they are only used for the FORMAL_CONFLICTS membership check.
LABEL_GROUP = "mac1A"
# Documented sequential ramp (single hue, light -> dark).
BLUES = LinearSegmentedColormap.from_list(
    "prerun_blues",
    ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
     "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
)


def cell_dir(output_root: Path, cue: float, true: float, hidden: int, seed: int) -> Path:
    return (
        output_root / "diagnostics" / "orthogonal_grid"
        / f"cue{cue:.1f}_true{true:.1f}" / f"h{hidden:02d}_seed{seed}" / "custom"
        / f"formal_osla_seed{seed}" / LABEL_GROUP
    )


def run_cell(output_root: Path, cue: float, true: float, hidden: int, seed: int,
             episodes: int) -> Path:
    target = cell_dir(output_root, cue, true, hidden, seed)
    if (target / "trials.csv").exists():
        return target
    checkpoint = (
        output_root / "arms" / f"h{hidden:02d}_b200" / "runs" / f"formal_osla_seed{seed}" / "best.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    kind = "consistent" if cue == true else "conflict"
    run_frozen_tests(
        checkpoint,
        # target = <out_root>/custom/<network_id>/<group>, so parents[2] is <out_root>
        out_root=target.parents[2],
        protocol="custom",
        groups=(LABEL_GROUP,),
        episodes=episodes,
        seed=TEST_SEED,
        custom_rounds=[(kind, float(cue), float(true))],
    )
    return target


def verify_matched_feedback(output_root: Path) -> None:
    """For each truth, the outcome sequence must be identical across every cue.

    That is what makes a *row* of the grid a clean read of the stated prior: the
    environment is held fixed, only the cue moves.  It holds because the number of
    RNG draws per round depends on `true_probability` and on the fixed
    batch x trials action draws, never on the cue, and every cell re-seeds the
    generator from `TEST_SEED`.

    Must run *after* every cell exists, not per cue.
    """
    for true in GRID:
        anchors = []
        for cue in GRID:
            path = cell_dir(output_root, cue, true, HIDDEN_SIZES[0], SEEDS[0]) / "trials.csv"
            if not path.exists():
                raise SystemExit(f"matched-feedback check needs {path} — run the full grid first")
            anchors.append(pd.read_csv(path)["outcome"].to_numpy()[:TRIALS_PER_ROUND])
        for cue, arr in zip(GRID[1:], anchors[1:]):
            if not np.array_equal(arr, anchors[0]):
                raise SystemExit(
                    f"matched-feedback check FAILED: true {true}, cue {cue} has a different "
                    f"outcome sequence than cue {GRID[0]}"
                )
    print("\nmatched-feedback check passed: for every truth the outcomes are identical across cues")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orthogonal cue x true grid on frozen networks.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "outputs_prerun")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    args = parser.parse_args()

    if int(torch.get_num_threads()) != 1:
        raise SystemExit(f"torch.get_num_threads() == {torch.get_num_threads()}, expected 1")

    output_root = args.output_root.resolve()
    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    networks = [(h, s) for h in HIDDEN_SIZES for s in SEEDS]
    total = len(GRID) ** 2 * len(networks)
    print(f"orthogonal grid: {len(GRID)} cues x {len(GRID)} truths x {len(networks)} networks = {total} cells")
    print(f"one round per cell (blank state), {args.episodes} episodes, TEST_SEED={TEST_SEED}\n")

    long_rows: list[dict] = []
    summary_rows: list[dict] = []

    for cue in GRID:
        for true in GRID:
            for hidden, seed in networks:
                target = run_cell(output_root, cue, true, hidden, seed, args.episodes)
                trials = pd.read_csv(target / "trials.csv")
                if len(trials) != args.episodes * TRIALS_PER_ROUND:
                    raise SystemExit(
                        f"cue {cue}/true {true}/h{hidden}/seed {seed}: {len(trials)} rows, "
                        f"expected {args.episodes * TRIALS_PER_ROUND}"
                    )
                if trials["round_id"].nunique() != 1:
                    raise SystemExit("each cell must be exactly one round")
                traj = (
                    trials.groupby("trial_id")["choice_probability"].mean()
                    .reindex(range(1, TRIALS_PER_ROUND + 1)).to_numpy(dtype=float)
                )
                if not np.isfinite(traj).all():
                    raise SystemExit(f"cue {cue}/true {true}: trajectory has gaps")
                boundary = pd.read_csv(target / "boundary_events.csv")
                opening = float(np.nanmean(traj[:OPENING_WINDOW]))
                late = float(np.nanmean(traj[LATE_START - 1:]))
                summary_rows.append(
                    {
                        "cue": cue, "true": true, "hidden_size": hidden, "seed": seed,
                        "opening_pbet": opening, "pbet_t1": float(traj[0]), "late_pbet": late,
                        "p_boundary": float(boundary["p_boundary"].mean()),
                        "round_mean_pbet": float(np.nanmean(traj)),
                        "in_distribution": cue in IN_DISTRIBUTION,
                    }
                )
                for trial, value in enumerate(traj, start=1):
                    long_rows.append(
                        {"cue": cue, "true": true, "hidden_size": hidden, "seed": seed,
                         "trial_id": trial, "pbet": float(value)}
                    )
        print(f"  cue {cue:.1f}: {len(GRID)} truths done", flush=True)

    verify_matched_feedback(output_root)

    long_df = pd.DataFrame(long_rows)
    summary = pd.DataFrame(summary_rows)
    long_df.to_csv(metrics_dir / "orthogonal_grid_trajectories.csv", index=False)
    network_mean = (
        summary.groupby(["cue", "true"])[
            ["opening_pbet", "pbet_t1", "late_pbet", "p_boundary", "round_mean_pbet"]
        ].mean().round(4)
    )
    network_mean.to_csv(metrics_dir / "orthogonal_grid_summary.csv")
    summary.to_csv(metrics_dir / "orthogonal_grid_per_network.csv", index=False)

    print("\nopening P(bet), trials 1-5 (rows = cue, columns = true):")
    print(network_mean["opening_pbet"].unstack().to_string())
    print("\nlate P(bet), trials 101-120:")
    print(network_mean["late_pbet"].unstack().to_string())

    plot_grid(long_df, network_mean, output_root)
    plot_heatmaps(network_mean, output_root)
    print(f"\nwrote {metrics_dir / 'orthogonal_grid_trajectories.csv'} ({len(long_df)} rows)")


def plot_grid(long_df: pd.DataFrame, network_mean: pd.DataFrame, output_root: Path) -> None:
    apply_style()
    n = len(GRID)
    fig, axes = plt.subplots(n, n, figsize=(1.36 * n, 1.30 * n), sharex=True, sharey=True)
    trajectories = long_df.groupby(["cue", "true", "trial_id"])["pbet"].mean().reset_index()
    for i, cue in enumerate(GRID):
        for j, true in enumerate(GRID):
            ax = axes[i, j]
            block = trajectories[(trajectories["cue"] == cue) & (trajectories["true"] == true)]
            colour = INK_PRIMARY if cue == true else "#2a78d6"
            ax.plot(block["trial_id"], block["pbet"], color=colour, lw=1.3)
            ax.axhline(cue, color=INK_MUTED, lw=0.6, ls="--")
            if cue != true:
                ax.axhline(true, color=INK_MUTED, lw=0.6, ls="-.")
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
                ax.set_facecolor("#f4f3ef")  # out-of-distribution cell
    handles = [
        plt.Line2D([], [], color=INK_PRIMARY, lw=1.6, label="cue == true"),
        plt.Line2D([], [], color="#2a78d6", lw=1.6, label="cue != true"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.6, ls="--", label="stated prior (cue)"),
        plt.Line2D([], [], color=INK_MUTED, lw=0.6, ls="-.", label="true probability"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#f4f3ef", edgecolor="none",
                      label="cue outside the training grid"),
    ]
    finish(
        fig,
        title="Orthogonal cue x true grid, ONE round from a blank state: betting-rate trajectory per cell\n"
              "9 cues x 9 true probabilities x 6 frozen networks (2 hidden sizes x 3 seeds), 50 episodes per cell",
        legend_handles=handles, ncol=5,
    )
    save(fig, output_root / "figures" / "orthogonal_grid_9x9.png")


def plot_heatmaps(network_mean: pd.DataFrame, output_root: Path) -> None:
    apply_style()
    panels = [
        ("opening_pbet", "opening P(bet), trials 1-5"),
        ("late_pbet", "late P(bet), trials 101-120"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5))
    for ax, (column, label) in zip(axes, panels):
        matrix = network_mean[column].unstack()
        image = ax.imshow(matrix.values, cmap=BLUES, vmin=0.0, vmax=1.0, origin="lower")
        for i, cue in enumerate(GRID):
            for j, true in enumerate(GRID):
                value = matrix.loc[cue, true]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.6,
                        color="#ffffff" if value > 0.55 else INK_PRIMARY)
        for k, value in enumerate(GRID):
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor=INK_PRIMARY, lw=1.1))
            if value not in IN_DISTRIBUTION:
                ax.add_patch(plt.Rectangle((-0.5, k - 0.5), len(GRID), 1, fill=False,
                                           edgecolor="#eb6834", lw=0.9, ls=":"))
                ax.add_patch(plt.Rectangle((k - 0.5, -0.5), 1, len(GRID), fill=False,
                                           edgecolor="#eb6834", lw=0.9, ls=":"))
        ax.set_xticks(range(len(GRID)))
        ax.set_xticklabels([f"{g:.1f}" for g in GRID], fontsize=6.5)
        ax.set_yticks(range(len(GRID)))
        ax.set_yticklabels([f"{g:.1f}" for g in GRID], fontsize=6.5)
        ax.set_xlabel("true probability", fontsize=7.5)
        ax.set_ylabel("stated prior (cue)", fontsize=7.5)
        ax.set_title(label, fontsize=8.5, color=INK_PRIMARY)
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=6)
    finish(
        fig,
        title="Orthogonal cue x true grid: where the betting rate ends up\n"
              "solid square = cue == true diagonal; orange dotted = cue outside the training grid {0.3..0.7}",
        legend_handles=None,
    )
    save(fig, output_root / "figures" / "orthogonal_grid_heatmaps.png")


if __name__ == "__main__":
    main()
