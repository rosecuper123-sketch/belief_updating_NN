"""Shared figure style for the pre-run plots.

Palette and rules come from the `dataviz` skill.  The three categorical slots
used for the three network seeds were validated with

    node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --surface "#fcfcfb"

which passes the lightness band, chroma floor, CVD separation (worst adjacent
deutan Delta E 9.2) and the normal-vision floor (worst adjacent 27.6).  It emits
one WARN: aqua `#1baf7a` sits at 2.74:1 against the light surface, below the 3:1
gate, which obligates *relief* -- visible direct labels or a table view.  These
figures ship both: every seed line is direct-labelled at its endpoint, and the
underlying numbers are written to `metrics/*.csv` as the table-view twin.

Ordered (not categorical) quantities -- the update budgets 200/400/600 -- use the
single-hue blue ordinal ramp, whose light-mode steps start no lighter than 250.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- categorical slots (identity: one per seed, fixed order, never cycled) ---
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SERIES_NAMES = ("seed 20260928", "seed 20260929", "seed 20260930")

# --- ordinal ramp (ordered magnitude: update budget) ---
ORDINAL = ("#86b6ef", "#2a78d6", "#104281")

# --- chrome & ink ---
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

ROOT = Path(__file__).resolve().parents[2]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.titleweight": "regular",
            "axes.titlecolor": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "figure.dpi": 110,
            "savefig.dpi": 300,
        }
    )


def reference_line(ax, y: float, *, label: str | None = None, style: str = "--") -> None:
    """A cue/true reference level: annotation, not a series -- neutral ink.

    Pass ``label=None`` in dense small-multiple layouts; the level is then
    identified by the figure legend plus the per-panel text, and nothing is drawn
    into the gutter between panels.
    """
    ax.axhline(y, color=INK_MUTED, lw=0.8, ls=style, zorder=1)
    if label is None:
        return
    ax.annotate(
        label,
        xy=(1.0, y),
        xycoords=("axes fraction", "data"),
        xytext=(2, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7,
        color=INK_MUTED,
        annotation_clip=False,
    )


def finish(fig, *, title: str, legend_handles=None, ncol: int = 4, legend_title: str | None = None) -> None:
    """Tighten the layout, then title at the top and legend along the bottom.

    Figure-level chrome keeps every in-panel legend from landing on the data,
    which it otherwise does whenever a curve ends where the legend sits.  The
    reserved bands are computed from the figure height so a multi-line title or a
    wrapped legend never overlaps the panels.
    """
    height = fig.get_figheight()
    title_lines = title.count("\n") + 1
    bottom = 0.0
    if legend_handles:
        rows = max(1, -(-len(legend_handles) // ncol))
        bottom = (rows * 0.20 + 0.06) / height
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=ncol,
            title=legend_title,
            handlelength=1.8,
            columnspacing=1.6,
        )
    top = 1.0 - (title_lines * 0.17 + 0.06) / height
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.suptitle(title, fontsize=9.5, color=INK_PRIMARY, y=1.0 - 0.05 / height)


def budget_markers(ax, budgets=(200, 400, 600)) -> None:
    """Candidate stopping points are a threshold -- dashing is meaningful here."""
    for budget in budgets:
        ax.axvline(budget, color=INK_MUTED, lw=0.7, ls=":", zorder=1)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.name} (+ .pdf)")


def panel_label(ax, text: str, *, loc: str = "upper left", box: bool = False) -> None:
    positions = {
        "upper left": (0.02, 0.97, "left", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "upper right": (0.98, 0.97, "right", "top"),
        "lower right": (0.98, 0.03, "right", "bottom"),
    }
    x, y, ha, va = positions[loc]
    kwargs = {}
    if box:
        # A surface-colored backing (no edge) keeps the text legible where a
        # trajectory passes through the corner it sits in.
        kwargs["bbox"] = {"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5}
    ax.text(
        x, y, text, transform=ax.transAxes, ha=ha, va=va,
        fontsize=7.5, color=INK_SECONDARY, **kwargs,
    )
