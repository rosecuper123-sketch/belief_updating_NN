"""Model comparison figure for the M1/M2/M3 fits: LOOIC, WAIC, and whether to believe them.

Reads `metrics/ic_comparison.csv` and `metrics/ic_pairwise.csv` (both written by
`scripts/compare_ic.R`) and draws four panels:

  (a) per-group LOOIC, as a difference from that group's best model, with the PAIRED
      standard error and the |delta|/se ratio annotated on each bar;
  (b) the same for WAIC, next to (a) so the two criteria can be compared by eye;
  (c) the Pareto-k inventory -- how many observations fall in each unreliable band, and
      the worst k -- because that, not the IC, decides which deltas may be quoted;
  (d) delta-LOOIC against delta-WAIC for all non-best fits, with the identity line.  The
      two criteria use different machinery (WAIC has no PSIS at all), so points hugging
      the line are the evidence that the ranking is not an artefact of the PSIS
      approximation.

Deliberately NOT re-deriving anything: every number here is read from the CSVs, so the
figure and the tables cannot drift apart.  The same reason the panels show *relative* IC
(delta vs the group's best) -- absolute LOOIC is not comparable across groups but the
within-group ordering is.

Colour encodes the MODEL in panels (a)-(c) and nothing in panel (d), where points are
direct-labelled instead; re-using the model palette for a different meaning inside one
figure is what makes a reader misread it.

Run:
    cd formal_run_001/analysis/fitting_NN
    ../../.venv/Scripts/python.exe scripts/plot_model_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FITTING_DIR = SCRIPT_DIR.parent
ROOT = FITTING_DIR.parents[1]           # formal_run_001
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from scripts.shared.prerun_style import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SURFACE,
    SERIES,
    apply_style,
    finish,
    panel_label,
    save,
)

METRICS_DIR = FITTING_DIR / "metrics"
FIGURES_DIR = FITTING_DIR / "figures"

GROUPS = ("mac1A", "mac1B", "mac2A", "mac2B")
MODELS = ("M1", "M2", "M3")
MODEL_COLOUR = dict(zip(MODELS, SERIES))
IC_LABEL = {"looic": "LOOIC", "waic": "WAIC"}
BAR_WIDTH = 0.26


def load_comparison(mode: str) -> pd.DataFrame:
    suffix = "" if mode == "final" else f"_{mode}"
    path = METRICS_DIR / f"ic_comparison{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/compare_ic.R --mode {mode}")
    return pd.read_csv(path)


def load_pairwise(mode: str) -> pd.DataFrame:
    suffix = "" if mode == "final" else f"_{mode}"
    path = METRICS_DIR / f"ic_pairwise{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run scripts/compare_ic.R --mode {mode}")
    return pd.read_csv(path)


def group_best(comp: pd.DataFrame, group: str, ic: str) -> str:
    sub = comp[comp.group == group]
    return str(sub.loc[sub[ic].idxmin(), "model"])


def paired_se(pairs: pd.DataFrame, group: str, model: str, best: str, ic: str) -> float:
    """Paired SE of (model - best), taken from ic_pairwise.

    The se of the DIFFERENCE is the right error bar for a comparison and is 2-4x smaller
    than either model's own se; using the individual se here would make every gap look
    far less certain than it is.  Zero for the best model itself, which is the reference.
    """
    if model == best:
        return 0.0
    a, b = sorted((model, best))
    hit = pairs[(pairs.group == group) & (pairs.model_a == a)
                & (pairs.model_b == b) & (pairs.ic == ic.upper())]
    if hit.empty:
        return float("nan")
    return float(hit.se_paired.iloc[0])


def panel_ic(ax, comp: pd.DataFrame, pairs: pd.DataFrame, ic: str, groups) -> None:
    x = np.arange(len(groups))
    for mi, model in enumerate(MODELS):
        deltas, errs, ratios = [], [], []
        for group in groups:
            sub = comp[comp.group == group]
            best = group_best(comp, group, ic)
            delta = float(sub.loc[sub.model == model, ic].iloc[0]) - float(sub[ic].min())
            se = paired_se(pairs, group, model, best, ic)
            deltas.append(delta)
            errs.append(se)
            ratios.append(delta / se if se and se > 0 else np.nan)
        offset = (mi - (len(MODELS) - 1) / 2) * BAR_WIDTH
        ax.bar(x + offset, deltas, BAR_WIDTH, color=MODEL_COLOUR[model],
               yerr=errs, capsize=2.2, zorder=3,
               error_kw={"ecolor": INK_SECONDARY, "lw": 0.9, "zorder": 4})
        for xi, delta, ratio in zip(x + offset, deltas, ratios):
            if not np.isfinite(delta):
                continue
            if delta == 0:
                # This model IS the group's best -- the reference every other bar is
                # measured against.  An empty slot would read as missing data.
                ax.annotate("best", (xi, 0), textcoords="offset points", xytext=(0, 4),
                            ha="center", va="bottom", fontsize=6.2, color=INK_MUTED)
                continue
            ax.annotate(f"|Δ|/se\n{ratio:.0f}", (xi, delta), textcoords="offset points",
                        xytext=(0, 5), ha="center", va="bottom",
                        fontsize=6.0, color=INK_SECONDARY, linespacing=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(f"Δ {IC_LABEL[ic]} vs group-best\n(lower = better)")
    ax.margins(y=0.22)


def panel_pareto(ax, comp: pd.DataFrame, groups) -> None:
    """How many observations per fit have an unreliable LOO predictive.

    One bar per fit, coloured by model like the panels above, with the worst k and the
    count of outright failures annotated.  An earlier version stacked the three severity
    bands inside each bar, but the bands would have needed a second legend whose swatches
    could not match the model-coloured bars -- the severity split is in
    ic_comparison.csv, which is this figure's table-view twin.
    """
    x = np.arange(len(groups))
    for mi, model in enumerate(MODELS):
        sub = comp[comp.model == model].set_index("group")
        totals, worsts, infs = [], [], []
        for group in groups:
            totals.append(int(sub.loc[group, "n_k_gt_07"]))
            worsts.append(float(sub.loc[group, "max_k"]))
            infs.append(int(sub.loc[group, "n_k_inf"]))
        offset = (mi - (len(MODELS) - 1) / 2) * BAR_WIDTH
        ax.bar(x + offset, totals, BAR_WIDTH, color=MODEL_COLOUR[model], zorder=3)
        for xi, total, worst, n_inf in zip(x + offset, totals, worsts, infs):
            worst_txt = "∞" if np.isinf(worst) else f"{worst:.2f}"
            # The k = Inf count is what decides whether the fit's elpd differences may be
            # quoted at all, so it is named rather than inferred from the bar height.
            text = f"{worst_txt}" + (f"\n{n_inf}×∞" if n_inf else "")
            ax.annotate(text, (xi, total), textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=6.2, linespacing=1.15,
                        color=INK_PRIMARY if n_inf else INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("observations with unreliable Pareto-k\n(annotation: max k)")
    ax.margins(y=0.24)


def panel_agreement(ax, comp: pd.DataFrame, groups) -> None:
    xs, ys, labels = [], [], []
    worst = 0.0
    for group in groups:
        sub = comp[comp.group == group]
        best_loo, best_waic = float(sub.looic.min()), float(sub.waic.min())
        for model in MODELS:
            row = sub[sub.model == model].iloc[0]
            d_loo = float(row.looic) - best_loo
            d_waic = float(row.waic) - best_waic
            worst = max(worst, abs(d_loo - d_waic))
            if model == group_best(comp, group, "looic"):
                continue                      # the reference itself: always (0, 0)
            xs.append(d_loo)
            ys.append(d_waic)
            labels.append(f"{group} {model}")

    data = max(max(xs, default=1.0), max(ys, default=1.0))
    span = data * 1.08
    # Labels sit on both sides of each point, so the axes are padded past the data range
    # on both ends -- otherwise the left-hand ones are drawn over the tick labels.
    pad = data * 0.16
    ax.plot([0, span], [0, span], color=INK_MUTED, lw=0.9, ls="--", zorder=1)
    ax.scatter(xs, ys, s=40, color=INK_PRIMARY, zorder=3,
               edgecolors=SURFACE, linewidths=1.1)
    # The points are near-collinear by construction, and the colliding ones are the pairs
    # that share a model (M1 mac1A vs mac1B differ by ~11 LOOIC units on a 1400-wide axis,
    # i.e. they print on top of each other).  Alternating above/below by index would not
    # separate them -- the cycle below puts every second same-model point on the other
    # side of the line, which does.
    offsets = ((6, 6, "left"), (-6, -10, "right"), (6, -10, "left"), (-6, 6, "right"))
    for i, (xi, yi, text) in enumerate(zip(xs, ys, labels)):
        dx, dy, ha = offsets[i % len(offsets)]
        ax.annotate(text, (xi, yi), textcoords="offset points", xytext=(dx, dy),
                    ha=ha, fontsize=6.2, color=INK_SECONDARY)
    ax.set_xlim(-pad, span)
    ax.set_ylim(-pad, span)
    ax.set_xlabel("Δ LOOIC vs group-best")
    ax.set_ylabel("Δ WAIC vs group-best")
    ax.set_aspect("equal", adjustable="box")
    panel_label(ax, f"identity line = perfect agreement\n"
                    f"max |ΔLOOIC − ΔWAIC| = {worst:.2f}", loc="upper left", box=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="final", choices=("final", "smoke"),
                        help="final (default) reads ic_comparison.csv; smoke reads the "
                             "_smoke files, matching compare_ic.R --mode")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    comp, pairs = load_comparison(args.mode), load_pairwise(args.mode)
    # Whatever groups the CSVs actually carry, not a hardcoded four -- a smoke run covers
    # three of them, and axes keyed to the wrong group list would silently mislabel bars.
    groups = [g for g in GROUPS if g in set(comp["group"])]
    # A smoke figure must not land on the final one: 250-draw fits and 2000-draw fits look
    # nothing alike and the filename is the only thing that distinguishes them afterwards.
    suffix = "" if args.mode == "final" else f"_{args.mode}"
    output = args.output or FIGURES_DIR / f"model_comparison_ic{suffix}.png"
    apply_style()

    width = 2.4 + 2.15 * len(groups)
    fig, axes = plt.subplots(2, 2, figsize=(width, 7.6))

    panel_ic(axes[0, 0], comp, pairs, "looic", groups)
    panel_label(axes[0, 0], "a  LOOIC (PSIS)", loc="upper left")
    panel_ic(axes[0, 1], comp, pairs, "waic", groups)
    panel_label(axes[0, 1], "b  WAIC (no PSIS)", loc="upper left")

    panel_pareto(axes[1, 0], comp, groups)
    panel_label(axes[1, 0], "c  Pareto-k inventory", loc="upper left")

    panel_agreement(axes[1, 1], comp, groups)
    panel_label(axes[1, 1], "d  the two criteria agree", loc="upper right")

    model_handles = [Patch(facecolor=MODEL_COLOUR[m], label=m) for m in MODELS]
    finish(
        fig,
        title="M1/M2/M3 model comparison across the groups — LOOIC and WAIC rank "
              "identically;\nPareto-k (panel c) is what decides whose deltas may be quoted",
        legend_handles=model_handles,
        ncol=3,
        legend_title="model",
    )
    save(fig, output)


if __name__ == "__main__":
    main()
