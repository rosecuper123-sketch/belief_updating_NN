"""Plot the M3-fitted dynamic learning-rate sequence for all four groups.

Ported from formal_batch/analysis/fitting_NN/scripts/plot_lr_sequence.py, with two
changes and one addition:

* all four groups instead of mac1A/mac1B only;
* **the evidence gate is its own panel row, not a twin axis.**  The old figure overlaid
  w(t) on a secondary y-axis; this codebase's `prerun_style` rules (and the dataviz
  guidance behind them) forbid dual axes, and splitting costs nothing -- the two rows
  share an x-axis so they read against each other directly.
* the fitted belief p(t) is shown next to the gate, since the gate opens as the belief
  moves and the two are easier to read together.

Reporting convention: the per-trial curve is the MEDIAN across the 30 networks of each
network's posterior-mean trajectory, with the IQR band.  Median, not mean -- the
subject-level LR distributions are right-skewed / bimodal and the mean hides the
pos/neg ordering reversal (see the batch README).

Run:
    cd formal_run_001
    ./.venv/Scripts/python.exe analysis/fitting_NN/scripts/plot_lr_sequence.py
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

from scripts.shared.prerun_style import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    SERIES,
    apply_style,
    finish,
    panel_label,
    save,
)

METRICS_DIR = FITTING_DIR / "metrics"
FIGURES_DIR = FITTING_DIR / "figures"

GROUPS = ("mac1A", "mac1B", "mac2A", "mac2B")
CUE = {"mac1A": 0.3, "mac1B": 0.7, "mac2A": 0.3, "mac2B": 0.7}
TRUE = {"mac1A": 0.6, "mac1B": 0.4, "mac2A": 0.5, "mac2B": 0.5}
C_POS = SERIES[0]
C_NEG = SERIES[1]


def load(group: str) -> pd.DataFrame:
    path = METRICS_DIR / f"lr_sequence_{group}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run extract_lr_sequence.R first")
    return pd.read_csv(path)


def load_params(group: str) -> dict:
    """Group-level readable parameters, for the parameter-layer crossover test."""
    path = METRICS_DIR / f"key_params_group_{group}.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return dict(zip(frame["variable"], frame["mean"]))


def sigma(values):
    """Elementwise logistic.  Monotone, so it commutes with quantiles."""
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def crossover_text(params: dict, pos, neg) -> str:
    """The two crossover questions, as the panel annotation.

    Both comparisons are ORDER tests, so any monotone transform of the inputs leaves the
    text unchanged -- which is what makes the assertion in main() a real check that the
    two spaces agree rather than a hope.
    """
    p_early, p_late = params.get("lr_pos_mu1"), params.get("lr_pos_mu2")
    n_early, n_late = params.get("lr_neg_mu1"), params.get("lr_neg_mu2")
    if p_early is None:
        param_txt = "params: n/a"
    else:
        crosses = (p_early > n_early) != (p_late > n_late)
        param_txt = (f"params: α⁺{'>' if p_early > n_early else '<'}α⁻ early → "
                     f"{'crosses' if crosses else 'stays'}")
    early = int(np.median(pos[:20]) < np.median(neg[:20]))
    late = int(np.median(pos[-20:]) < np.median(neg[-20:]))
    return f"{param_txt}\ntraj: {'crosses' if early != late else 'no crossover'}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--space", choices=("prob", "sigmoid"), default="prob",
        help="prob (default): plot alpha as-is.  sigmoid: apply ANOTHER logistic on top "
             "of the already-probability alpha -- a second transform, NOT a different "
             "parameterisation of the model; see the figure caption.")
    args = parser.parse_args()
    second = args.space == "sigmoid"
    transform = sigma if second else (lambda v: np.asarray(v, dtype=float))

    frames = {g: load(g) for g in GROUPS}
    key_params = {g: load_params(g) for g in GROUPS}
    apply_style()

    # The top row always shares one y-range across the four groups.  In sigmoid space the
    # data only span ~0.50-0.73, so a fixed 0-1 axis would squash every curve flat.
    curves = {}
    for group in GROUPS:
        df = frames[group]
        curves[group] = {k: transform(df[f"lr_{k}"].to_numpy())
                         for k in ("pos_median", "pos_q25", "pos_q75",
                                   "neg_median", "neg_q25", "neg_q75")}
    top_values = np.concatenate([v for c in curves.values() for v in c.values()])
    lo, hi = float(top_values.min()), float(top_values.max())
    pad = 0.06 * (hi - lo)
    top_lim = (0.0, 1.0) if not second else (lo - pad, hi + pad)

    fig, axes = plt.subplots(2, len(GROUPS), figsize=(15.0, 6.4), sharex=True)

    for column, group in enumerate(GROUPS):
        df = frames[group]
        c = curves[group]
        t = df["trial"].to_numpy()
        n_trials = int(t.max())
        ax = axes[0, column]
        ax.plot(t, c["pos_median"], color=C_POS, lw=1.8, label="α⁺ (win)")
        ax.fill_between(t, c["pos_q25"], c["pos_q75"], color=C_POS, alpha=0.18, lw=0)
        ax.plot(t, c["neg_median"], color=C_NEG, lw=1.8, ls="--", label="α⁻ (loss)")
        ax.fill_between(t, c["neg_q25"], c["neg_q75"], color=C_NEG, alpha=0.18, lw=0)
        ax.set_ylim(*top_lim)
        ax.set_xlim(1, n_trials)
        ax.set_title(f"{group}   cue {CUE[group]:.2f} → true {TRUE[group]:.2f}",
                     fontsize=9, color=INK_PRIMARY)
        if column == 0:
            ax.set_ylabel("σ(α) — second transform,\nNOT the model's learning rate"
                          if second else "learning rate\n(posterior median)")
            ax.legend(loc="upper right", fontsize=7.5)

        # Two different crossover questions, and they can disagree:
        #
        #  * PARAMETER layer -- is the early component (gate closed) ordered differently
        #    from the late component (gate open)?  This is what the old pipeline reported
        #    and what the model's structure actually asserts.
        #  * TRAJECTORY layer -- does the plotted w(t)-interpolated curve visibly cross?
        #    When theta is small the gate saturates within ~15 trials, the early component
        #    is barely visited, and the trajectory shows no crossing even though the
        #    parameters do.  Reporting only the trajectory would hide a real parameter
        #    crossover, so both are annotated.
        par = key_params.get(group, {})
        shown = crossover_text(
            {k: (float(transform(v)) if k.startswith("lr_") else v) for k, v in par.items()},
            c["pos_median"], c["neg_median"])
        text = shown
        if second:
            # Both layers are order tests, so the sigmoid figure must carry the SAME
            # verdict as the probability one.  If this ever fires, the transform was
            # applied somewhere it should not have been.
            raw = crossover_text(par, df["lr_pos_median"].to_numpy(),
                                 df["lr_neg_median"].to_numpy())
            if shown != raw:
                raise SystemExit(f"{group}: crossover verdict changed under a monotone "
                                 f"transform ({raw!r} vs {shown!r}) — bug")
            span = (f"α {df.lr_pos_median.min():.3f}–{df.lr_pos_median.max():.3f} → "
                    f"σ {c['pos_median'].min():.3f}–{c['pos_median'].max():.3f}")
            text = f"{shown}\n{span}"
        panel_label(ax, text, loc="lower right", box=True)

        # Gate and belief share one axis because both are probabilities on [0,1] -- no
        # second y-axis is needed, and the reference lines read against the same scale.
        # Untouched by --space: the transform is about alpha only.
        ax = axes[1, column]
        ax.plot(t, df["w_median"], color=INK_PRIMARY, lw=1.6, label="gate  w(t)")
        ax.plot(t, df["p_median"], color=SERIES[2], lw=1.6, label="belief  p(t)")
        ax.axhline(CUE[group], color=INK_MUTED, lw=0.9, ls="--")
        ax.axhline(TRUE[group], color=INK_MUTED, lw=0.9, ls="-.")
        ax.set_ylim(0, 1)
        ax.set_xlabel("conflict-round trial")
        panel_label(ax, "dashed = cue, dash-dot = true", loc="lower right", box=True)
        if column == 0:
            ax.set_ylabel("gate / belief")
            ax.legend(loc="upper left", fontsize=7.5)

    if second:
        title = (
            "σ(α) — a SECOND logistic applied to an α that is ALREADY a probability\n"
            "α comes out of the model as Phi_approx(.) inside (0,1), so this axis is a "
            "re-scaling of the plotted quantity, NOT the model's learning rate\n"
            "(mac1A α⁺ collapses 0.207–0.843 → 0.552–0.699).  Ordering is preserved, so the "
            "crossover verdicts below are identical to lr_sequence_condA.png"
        )
    else:
        title = (
            "M3 dynamic learning rate — formal_osla h=16, 30 networks, one conflict round\n"
            "top: posterior median across networks of each network's mean α⁺/α⁻ trajectory, "
            "band = IQR (median, not mean: the subject distributions are right-skewed)\n"
            "bottom: the evidence gate w(t) and the fitted belief p(t), same probability axis"
        )
    finish(fig, title=title, legend_handles=None)
    save(fig, FIGURES_DIR / ("lr_sequence_condA_sigmoid.png" if second
                             else "lr_sequence_condA.png"))


if __name__ == "__main__":
    main()
