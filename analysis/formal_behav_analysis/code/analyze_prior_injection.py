"""Prior injection: does the stated prior drive the bet rate, and by how much?

Reads the **consistent-round rollout** (`consistent_rollout/cue<c>/h<hh>_seed<s>/...`,
one consistent round per cell, blank start, cue 0.1-0.9).  That tree was collected as
belief-decoder calibration data; it is reused here because it is also the cleanest
prior-injection design available -- nine cue values, each its own independent cell,
200 episodes x 30 networks.

Why behaviour and not the decoder
---------------------------------
Prior injection is measured on **P(bet)** (`choice_probability`), never on decoded
belief.  The decoder is *fit* on exactly this data, so reading belief-vs-behaviour
back off it here would be circular.  The non-circular use of the decoder is the
`conflict_only` side, which it never saw (see `plot_decoded_belief.py`).

What is measured
----------------
* **opening** P(bet), trials 1-5 -- the prior read, before any outcome is visible;
* **trial 1** alone -- the purest read, taken before any feedback exists;
* **late** P(bet), trials 101-120 -- whether evidence has overwritten the prior;
  here cue == true, so a faithful reader should end at the cue, and a
  near-oracle step rule should instead end at 0 for cue <= 0.5 and 1 above it.
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
    ORDINAL,
    SERIES,
    apply_style,
    finish,
    panel_label,
    save,
)
from analysis.formal_behav_analysis.loading import TRIALS_PER_ROUND, load_cell  # noqa: E402

CUES = tuple(round(0.1 * i, 1) for i in range(1, 10))
DEFAULT_SEEDS = tuple(range(20260804, 20260834))
OPENING_WINDOW = 5
LATE_START = 101


def cell_dir(out_root: Path, cue: float, hidden: int, seed: int) -> Path:
    return (
        out_root / "consistent_rollout" / f"cue{cue:.1f}" / f"h{hidden:02d}_seed{seed}"
        / "custom" / f"formal_osla_seed{seed}" / "mac1A"
    )


def collect(out_root: Path, hidden: int, seeds, cues=CUES) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for cue in cues:
            run_dir = cell_dir(out_root, cue, hidden, seed)
            if not (run_dir / "trials.csv").exists():
                raise FileNotFoundError(f"missing consistent cell: {run_dir}")
            cell = load_cell(run_dir)
            trials = cell.trials
            if trials["round_id"].nunique() != 1:
                raise SystemExit(f"{run_dir}: expected exactly one round")
            # A consistent round is one where every trial has cue == true.  If that
            # fails the cell is mislabelled and the whole readout is meaningless.
            if not np.allclose(trials["cue_probability"], trials["true_probability"]):
                raise SystemExit(f"{run_dir}: cue != true on some trial")
            if not np.allclose(trials["cue_probability"], cue):
                raise SystemExit(f"{run_dir}: cue is not {cue}")
            trajectory = (
                trials.groupby("trial_id")["choice_probability"].mean()
                .reindex(range(1, TRIALS_PER_ROUND + 1)).to_numpy(dtype=float)
            )
            if not np.isfinite(trajectory).all():
                raise SystemExit(f"{run_dir}: trajectory has gaps")
            rows.append({
                "seed": seed, "cue": cue, "episodes": cell.n_episodes,
                "pbet_trial1": float(trajectory[0]),
                "opening": float(np.nanmean(trajectory[:OPENING_WINDOW])),
                "late": float(np.nanmean(trajectory[LATE_START - 1:])),
                "round_mean": float(np.nanmean(trajectory)),
                "_traj": trajectory,
            })
    expected = len(seeds) * len(cues)
    if len(rows) != expected:
        raise SystemExit(f"collected {len(rows)} cells, expected {expected}")
    return pd.DataFrame(rows)


def per_network_slopes(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, block in df.groupby("seed"):
        cue = block["cue"].to_numpy(dtype=float)
        record = {"seed": seed}
        for column in ("pbet_trial1", "opening", "late"):
            slope, intercept = np.polyfit(cue, block[column].to_numpy(dtype=float), 1)
            record[f"{column}_slope"] = float(slope)
            record[f"{column}_intercept"] = float(intercept)
            record[f"{column}_dev"] = float((block[column].to_numpy(dtype=float) - cue).mean())
            record[f"{column}_abs_dev"] = float(np.abs(block[column].to_numpy(dtype=float) - cue).mean())
        rows.append(record)
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def plot(df: pd.DataFrame, slopes: pd.DataFrame, analysis_root: Path, hidden: int) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    grid = np.array(CUES)
    seeds = sorted(df["seed"].unique())

    for ax, (column, title, ylabel) in zip(
        axes[:2],
        (
            ("opening", "Prior read at round opening", "P(bet), trials 1-5"),
            ("late", "After a full round of evidence", "P(bet), trials 101-120"),
        ),
    ):
        ax.plot(grid, grid, color=INK_MUTED, lw=0.9, ls=":", zorder=1,
                label="faithful reading (y = cue)")
        for seed in seeds:
            block = df[df["seed"] == seed].sort_values("cue")
            ax.plot(block["cue"], block[column], color=INK_SECONDARY, lw=0.55, alpha=0.22, zorder=2)
        mean = df.groupby("cue")[column].mean()
        sem = df.groupby("cue")[column].sem()
        ax.errorbar(mean.index, mean.to_numpy(), yerr=sem.to_numpy(), color=SERIES[0], lw=1.9,
                    marker="o", ms=3.6, capsize=2.4, zorder=4, label=f"mean of {len(seeds)} networks")
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("stated prior (cue)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, color=INK_PRIMARY)
        ax.legend(loc="upper left", fontsize=7.5)

    axes[0].axvspan(0.25, 0.75, color="#f4f3ef", zorder=0)
    panel_label(axes[0], "grey band = the cue values seen in training",
                loc="lower right", box=True)

    ax = axes[2]
    for position, (column, label) in enumerate(
        (("opening_slope", "opening (t1-5)"), ("pbet_trial1_slope", "trial 1"), ("late_slope", "late (t101-120)"))
    ):
        values = slopes[column].to_numpy(dtype=float)
        jitter = np.linspace(-0.10, 0.10, len(values))
        ax.scatter(position + jitter, values, s=24, color=SERIES[0], alpha=0.72, linewidths=0, zorder=3)
        ax.scatter([position], [np.median(values)], marker="_", s=420, color=INK_PRIMARY,
                   zorder=4, linewidths=2.2)
        ax.text(position + 0.30, np.median(values), f"median {np.median(values):.2f}",
                fontsize=7.5, color=INK_SECONDARY, va="center")
    ax.axhline(1.0, color=INK_MUTED, lw=0.9, ls=":", zorder=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["opening\n(t1-5)", "trial 1", "late\n(t101-120)"])
    ax.set_ylabel("slope of P(bet) on cue  (ideal 1.0)")
    ax.set_title("How steeply each network follows the cue", fontsize=9, color=INK_PRIMARY)
    panel_label(ax, "one dot per network", loc="upper left", box=True)

    finish(
        fig,
        title=f"Prior injection on consistent rounds — {len(seeds)} networks, h = {hidden}, "
              f"{int(df['episodes'].iloc[0])} episodes per network per cue\n"
              f"behaviour (P(bet)), not decoded belief: the decoder is fit on this very data, so reading belief back here would be circular",
        legend_handles=None,
    )
    save(fig, analysis_root / "figures" / f"formal_prior_injection_h{hidden:02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path,
                        default=_ROOT / "analysis" / "formal_behav_analysis")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    analysis_root = args.analysis_root.resolve()
    metrics_dir = analysis_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = collect(out_root, args.hidden_size, args.seeds)
    slopes = per_network_slopes(df)
    df.drop(columns=["_traj"]).to_csv(metrics_dir / f"prior_injection_h{args.hidden_size:02d}.csv", index=False)
    slopes.to_csv(metrics_dir / f"prior_injection_slopes_h{args.hidden_size:02d}.csv", index=False)

    print(f"cells: {len(df)}  networks: {df['seed'].nunique()}  cues: {len(CUES)}")
    overall = df.groupby("cue")[["pbet_trial1", "opening", "late"]].mean()
    spread = df.groupby("cue")[["opening"]].std()["opening"]
    print(f"\n{'cue':>5s} {'trial1':>8s} {'opening':>9s} {'late':>7s} {'open-cue':>9s} {'sd(net)':>8s}")
    for cue in CUES:
        print(f"{cue:5.1f} {overall.loc[cue, 'pbet_trial1']:8.3f} {overall.loc[cue, 'opening']:9.3f} "
              f"{overall.loc[cue, 'late']:7.3f} {overall.loc[cue, 'opening'] - cue:+9.3f} {spread.loc[cue]:8.3f}")

    print("\nper-network slopes (ideal 1.0 for opening / trial1; late follows the step rule, not the cue):")
    for column in ("pbet_trial1_slope", "opening_slope", "late_slope"):
        values = slopes[column]
        print(f"  {column:22s} mean {values.mean():+.3f}  sd {values.std():.3f}  "
              f"min {values.min():+.3f}  max {values.max():+.3f}")
    print("\nmean absolute deviation from the cue (per network, then averaged):")
    for column in ("pbet_trial1_abs_dev", "opening_abs_dev"):
        print(f"  {column:24s} {slopes[column].mean():.4f}")

    plot(df, slopes, analysis_root, args.hidden_size)


if __name__ == "__main__":
    main()
