"""Diagnose how much leverage the one-shot cue has *inside* a multi-round episode.

Motivation.  In `human_default` / `legacy_3plus1` the conflict round opens at the
opposite extreme of its own cue (mac1A opens at ~0.985 while its cue is 0.30),
because the preceding consistent round's cue is the complement of the conflict
cue and the hidden state is deliberately never reset.  An independent audit of
both the experiment path and the analysis path found no coding bug, and the
saved artifacts confirm the per-round cue assignment is correct.  What is left
to measure is the *size* of the effect, which the canonical protocols cannot
show because they never vary the cue independently of the carried state.

Three zero-code-change experiments, driven through `formal_test_rounds(custom=...)`:

* **E1  cue gain** -- hold the carried state fixed (one preceding round, always at
  the group's consistent cue) and sweep the *conflict-round* cue.  If the opening
  P(bet) barely moves, the cue has almost no leverage at an in-episode boundary.
* **E2  preceding-cue dose** -- hold the conflict round fixed and sweep the
  *preceding* round's cue, including the no-preceding-round case.  A monotone
  response confirms the opening is set by carried state, not by the conflict cue.
* **E3  round independence (decisive)** -- a three-round all-consistent episode
  with three *different* cues, which is the shape training actually uses (each
  round's probability is drawn independently).  Compare each round's opening
  P(bet) against single-round references at every cue: does round 3 follow its
  own cue, or lag to round 2's?  A lag would mean the multi-round *training*
  distribution is being solved from carried state that carries no information
  about the current round -- a real defect.  Following its own cue means the
  canonical multi-round protocols are merely a limited *test*, and training is fine.
"""

from __future__ import annotations

import argparse
import json
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

from scripts.shared.prerun_style import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    ROOT,
    SERIES,
    apply_style,
    finish,
    save,
)
import matplotlib.pyplot as plt  # noqa: E402  (prerun_style pins the Agg backend)

from scripts.plotting.plot_pbet_structure import TRIALS_PER_ROUND, summarize  # noqa: E402
from scripts.testing.run_prerun_tests import check_outputs  # noqa: E402

from formal_osla.protocols import FORMAL_CONFLICTS  # noqa: E402
from formal_osla.test_runner import run_frozen_tests  # noqa: E402

CUE_GRID = (0.30, 0.50, 0.70)
E3_ORDERS = ("E3_asc", "E3_desc")
INVESTIGATED_GROUPS = ("mac1A", "mac1B")
DIAGNOSTIC_HIDDEN = 32
DIAGNOSTIC_BUDGET = 200
EPISODES = 200
TEST_SEED = 20261107
OPENING_WINDOW = 5
# A round is judged to "follow its own cue" if its opening value lands within this
# of the single-round reference at that cue.  Fixed here, before any data is seen.
OWN_CUE_TOLERANCE = 0.10


def consistent(value: float) -> tuple[str, float, float]:
    return ("consistent", float(value), float(value))


def conflict(cue: float, true: float) -> tuple[str, float, float]:
    return ("conflict", float(cue), float(true))


def build_variants(group: str) -> dict[str, list[tuple[str, float, float]]]:
    spec = FORMAL_CONFLICTS[group]
    cc, xc, xt = spec["consistent_cue"], spec["conflict_cue"], spec["conflict_true"]
    variants: dict[str, list[tuple[str, float, float]]] = {}
    for cue in CUE_GRID:
        variants[f"E1_cue{cue:.2f}"] = [consistent(cc), conflict(cue, xt)]
    variants["E2_preNone"] = [conflict(xc, xt)]
    for pre in CUE_GRID:
        variants[f"E2_pre{pre:.2f}"] = [consistent(pre), conflict(xc, xt)]
    for cue in CUE_GRID:
        variants[f"E3_ref{cue:.2f}"] = [consistent(cue)]
    variants["E3_asc"] = [consistent(0.30), consistent(0.70), consistent(0.50)]
    variants["E3_desc"] = [consistent(0.70), consistent(0.30), consistent(0.50)]
    return variants


def dedupe(variants: dict[str, list]) -> dict[tuple, list[str]]:
    """Several names can describe one round list (E1 cue=xc == E2 pre=cc)."""
    grouped: dict[tuple, list[str]] = {}
    for name, rounds in variants.items():
        grouped.setdefault(tuple(rounds), []).append(name)
    return grouped


def canonical_key(group: str) -> tuple:
    spec = FORMAL_CONFLICTS[group]
    return tuple([consistent(spec["consistent_cue"]), conflict(spec["conflict_cue"], spec["conflict_true"])])


def round_metrics(frame: pd.DataFrame) -> dict:
    traj = (
        frame.groupby("trial_id")["choice_probability"].mean()
        .reindex(range(1, TRIALS_PER_ROUND + 1)).to_numpy(dtype=float)
    )
    return {
        "cue": float(frame["cue_probability"].iloc[0]),
        "true": float(frame["true_probability"].iloc[0]),
        "mismatch": int(frame["mismatch_flag"].iloc[0]),
        "pbet_t1": float(traj[0]),
        "pbet_t1_5": float(np.nanmean(traj[:OPENING_WINDOW])),
        "pbet_last20": float(np.nanmean(traj[-20:])),
        "pbet_round_mean": float(np.nanmean(traj)),
        "traj_range": float(np.nanmax(traj) - np.nanmin(traj)),
        # Round reward, used to price the cost of the one-round lag: a round that
        # opens on the previous round's policy forgoes the reward it could have
        # earned by acting on its own cue from the first trial.
        "mean_reward": float(frame["reward"].mean()),
        "reward_ceiling": float(max(2.0 * frame["true_probability"].iloc[0] - 1.0, 0.0)),
    }


def checkpoint_for(output_root: Path, seed: int) -> Path:
    return (
        output_root / "arms" / f"h{DIAGNOSTIC_HIDDEN:02d}_b{DIAGNOSTIC_BUDGET}"
        / "runs" / f"formal_osla_seed{seed}" / "best.pt"
    )


def collect(diag_root: Path, checkpoint: Path, rounds: list, *, label: str, group: str) -> Path:
    out_root = diag_root / label
    target = out_root / "custom" / checkpoint.parent.name / group
    if not target.exists():
        run_frozen_tests(
            checkpoint,
            out_root=out_root,
            protocol="custom",
            groups=(group,),
            episodes=EPISODES,
            seed=TEST_SEED,
            custom_rounds=rounds,
        )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose cue leverage inside a multi-round episode.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "outputs_prerun")
    parser.add_argument("--groups", nargs="+", default=list(INVESTIGATED_GROUPS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260928, 20260929, 20260930])
    args = parser.parse_args()

    if int(torch.get_num_threads()) != 1:
        raise SystemExit(f"torch.get_num_threads() == {torch.get_num_threads()}, expected 1")

    output_root = args.output_root.resolve()
    diag_root = output_root / "diagnostics"
    diag_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for group in args.groups:
        grouped = dedupe(build_variants(group))
        print(f"\n=== {group}: {len(build_variants(group))} names over {len(grouped)} distinct round lists ===",
              flush=True)
        for seed in args.seeds:
            checkpoint = checkpoint_for(output_root, seed)
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            for rounds, names in grouped.items():
                label = f"{group}_{names[0]}"
                target = collect(diag_root, checkpoint, list(rounds), label=label, group=group)
                checks = check_outputs(
                    target, episodes=EPISODES, hidden_size=DIAGNOSTIC_HIDDEN, rounds=len(rounds)
                )
                if not checks["pass"]:
                    failed = [k for k, v in checks.items() if isinstance(v, tuple) and not v[0]]
                    raise SystemExit(f"structural check failed for {label}: {failed}")

                trials = pd.read_csv(target / "trials.csv")
                boundary = pd.read_csv(target / "boundary_events.csv")
                # One row per NAME, not per round list: a single round list can be the
                # canonical member of two experiments at once (E1 at the group's conflict
                # cue IS E2 with the preceding cue set to the group's consistent cue), and
                # both experiments need to see that point.
                aliases = "+".join(names)
                pre_cue = rounds[0][1] if len(rounds) >= 2 else float("nan")
                for round_id in sorted(trials["round_id"].unique()):
                    metrics = round_metrics(trials[trials["round_id"] == round_id])
                    br = boundary[boundary["round_id"] == round_id]
                    for name in names:
                        records.append(
                            {
                                "group": group,
                                "seed": seed,
                                "variant": name,
                                "aliases": aliases,
                                "round_id": int(round_id),
                                "n_rounds": len(rounds),
                                "pre_cue": float(pre_cue),
                                "p_boundary": float(br["p_boundary"].mean()),
                                **metrics,
                            }
                        )

                # Free cross-check: on multi-round variants the last round must agree
                # between `summarize` (the code behind the canonical figures) and the
                # local per-round path, or one of the two is mis-indexed.
                if len(rounds) >= 2:
                    ref = summarize(trials)
                    mine = round_metrics(trials[trials["round_id"] == trials["round_id"].max()])
                    for key, theirs in (("pbet_t1", ref["start_pbet"]),
                                        ("pbet_t1_5", ref["opening_pbet"]),
                                        ("pbet_last20", ref["late_pbet"])):
                        if abs(theirs - mine[key]) > 1e-12:
                            raise SystemExit(
                                f"{label} seed {seed}: summarize() vs round_metrics() disagree on "
                                f"{key} ({theirs} vs {mine[key]})"
                            )
            print(f"  seed {seed}: done", flush=True)

    frame = pd.DataFrame(records)
    frame.to_csv(diag_root / "round_structure.csv", index=False)
    print(f"\nwrote {diag_root / 'round_structure.csv'}  ({len(frame)} rows)")

    reconcile(diag_root, output_root, frame, args.groups)
    write_verdict(diag_root, frame, args.groups)
    plot(diag_root, frame, args.groups)


def reconcile(diag_root: Path, output_root: Path, frame: pd.DataFrame, groups) -> None:
    """The E1 variant at the group's own conflict cue must reproduce `human_default`.

    `run_frozen_tests` builds ONE generator and iterates the groups in a fixed order,
    so a group's outcome stream depends on how many groups ran before it.  The first
    group must therefore match bit-exactly, while later groups may differ by the
    floating-point noise of a different (but still balanced) outcome permutation.
    Anything larger than a hair means the round lists are not equivalent.
    """
    from formal_osla.protocols import FORMAL_GROUPS

    rows = []
    for group in groups:
        # The canonical multi-round test IS the E1 variant at the group's own conflict cue.
        key = canonical_key(group)
        variant = dedupe(build_variants(group))[key][0]
        for seed in sorted(frame.seed.unique()):
            diag = frame[(frame.group == group) & (frame.seed == seed) & (frame.variant == variant)]
            if not len(diag):
                continue
            diag = diag[diag.round_id == diag.round_id.max()]
            canon = (
                output_root / "frozen_tests" / f"h{DIAGNOSTIC_HIDDEN:02d}_b{DIAGNOSTIC_BUDGET}"
                / "human_default" / f"formal_osla_seed{seed}" / group / "trials.csv"
            )
            if not canon.exists():
                continue
            ref = summarize(pd.read_csv(canon))
            rows.append(
                {
                    "group": group,
                    "canonical_position": list(FORMAL_GROUPS).index(group) + 1,
                    "seed": seed,
                    "variant": variant,
                    "diag_start_pbet": float(diag.pbet_t1.iloc[0]),
                    "canonical_start_pbet": ref["start_pbet"],
                    "delta": float(diag.pbet_t1.iloc[0]) - ref["start_pbet"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(diag_root / "reconciliation_vs_canonical.csv", index=False)
    print("\nreconciliation: E1(cue = conflict cue) vs canonical human_default")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    if not len(out):
        return
    first = out[out.canonical_position == 1]
    later = out[out.canonical_position > 1]
    worst_first = float(first["delta"].abs().max()) if len(first) else 0.0
    worst_later = float(later["delta"].abs().max()) if len(later) else 0.0
    print(f"first canonical group: worst |delta| = {worst_first:.3e} "
          f"-> {'bit-exact' if worst_first == 0.0 else 'NOT bit-exact, investigate'}")
    if len(later):
        print(f"later canonical groups: worst |delta| = {worst_later:.3e} -> "
              f"{'consistent with the shared-generator stream offset (groups share one RNG)' if worst_later < 1e-3 else 'TOO LARGE: round lists may not be equivalent'}")
    if worst_first != 0.0 or worst_later >= 1e-3:
        raise SystemExit("reconciliation failed: diagnostic round lists are not equivalent to canonical")


def write_verdict(diag_root: Path, frame: pd.DataFrame, groups) -> None:
    """E3 decision rule, fixed before the run (see the plan)."""
    refs = frame[frame.variant.str.startswith("E3_ref")].groupby("cue")["pbet_t1_5"].mean()
    ref_by_cue = {float(k): float(v) for k, v in refs.items()}
    grid = sorted(ref_by_cue)

    def nearest(cue: float) -> float:
        return min(grid, key=lambda c: abs(c - cue))

    verdict: dict = {
        "own_cue_tolerance": OWN_CUE_TOLERANCE,
        "single_round_reference_pbet_t1_5": {f"{c:.2f}": ref_by_cue[c] for c in grid},
        "groups": {},
        "rounds_after_first": [],
    }
    for group in groups:
        e3 = frame[(frame.group == group) & (frame.variant.isin(E3_ORDERS))]
        entries = []
        for _, row in e3.iterrows():
            own = nearest(row["cue"])
            entries.append(
                {
                    "variant": row["variant"],
                    "seed": int(row["seed"]),
                    "round_id": int(row["round_id"]),
                    "own_cue": float(row["cue"]),
                    "pbet_t1_5": float(row["pbet_t1_5"]),
                    "ref_at_own_cue": ref_by_cue[own],
                    "abs_dev_from_own_cue_ref": abs(float(row["pbet_t1_5"]) - ref_by_cue[own]),
                }
            )
        verdict["groups"][group] = entries

        # Only rounds after the first can "lag": round 1 starts from a blank state.
        for variant in E3_ORDERS:
            for seed in sorted(e3.seed.unique()):
                ordered = e3[(e3.variant == variant) & (e3.seed == seed)].sort_values("round_id")
                for i in range(1, len(ordered)):
                    row, prev = ordered.iloc[i], ordered.iloc[i - 1]
                    own = nearest(row["cue"])
                    prev_cue = nearest(prev["cue"])
                    d_own = abs(float(row["pbet_t1_5"]) - ref_by_cue[own])
                    d_prev = abs(float(row["pbet_t1_5"]) - ref_by_cue[prev_cue])
                    verdict["rounds_after_first"].append(
                        {
                            "group": group,
                            "variant": variant,
                            "seed": int(seed),
                            "round_id": int(row["round_id"]),
                            "own_cue": float(row["cue"]),
                            "prev_cue": float(prev["cue"]),
                            "pbet_t1_5": float(row["pbet_t1_5"]),
                            "dist_to_own_cue_ref": d_own,
                            "dist_to_prev_cue_ref": d_prev,
                            "follows": "own_cue" if d_own <= d_prev else "previous_cue",
                            "within_tolerance": bool(d_own <= OWN_CUE_TOLERANCE),
                        }
                    )

    lagging = [r for r in verdict["rounds_after_first"] if r["follows"] == "previous_cue"]
    deviating = [r for r in verdict["rounds_after_first"] if not r["within_tolerance"]]
    verdict["n_rounds_after_first"] = len(verdict["rounds_after_first"])
    verdict["n_following_previous_cue"] = len(lagging)
    verdict["n_deviating_beyond_tolerance"] = len(deviating)
    verdict["E3_verdict"] = (
        "TRAINING DOMAIN OK: later rounds follow their own cue"
        if not lagging and not deviating
        else "LAGGING ROUNDS DETECTED: later rounds do not follow their own cue"
    )

    # Price the lag: compare the three-round episode's reward against the sum of the
    # same model's own single-round rewards at each of those cues (its blank-state,
    # no-lag behaviour).  A large gap means the lag, not the within-round policy, is
    # what is holding the training return down.
    cost = []
    ref_reward = frame[frame.variant.str.startswith("E3_ref")].groupby("cue")["mean_reward"].mean()
    for group in groups:
        for variant in E3_ORDERS:
            for seed in sorted(frame.seed.unique()):
                ep = frame[(frame.group == group) & (frame.variant == variant) & (frame.seed == seed)]
                if not len(ep):
                    continue
                ep = ep.sort_values("round_id")
                cues = [min(ref_reward.index, key=lambda c: abs(c - row["cue"])) for _, row in ep.iterrows()]
                no_lag = float(sum(ref_reward[c] for c in cues))
                actual = float(ep["mean_reward"].sum())
                ceiling = float(ep["reward_ceiling"].sum())
                cost.append(
                    {
                        "group": group,
                        "variant": variant,
                        "seed": int(seed),
                        "actual_episode_reward_per_trial": actual,
                        "no_lag_reference": no_lag,
                        "reward_ceiling": ceiling,
                        "lag_cost": no_lag - actual,
                        "lag_cost_as_frac_of_reference": (no_lag - actual) / no_lag if no_lag else float("nan"),
                        "actual_as_frac_of_ceiling": actual / ceiling if ceiling else float("nan"),
                    }
                )
    verdict["lag_cost"] = cost
    if cost:
        frac = float(np.mean([c["lag_cost_as_frac_of_reference"] for c in cost]))
        verdict["mean_lag_cost_as_frac_of_reference"] = frac
        print(f"  lag cost: {frac * 100:.1f}% of the model's own no-lag reward "
              f"(mean over {len(cost)} episodes)")
    (diag_root / "e3_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\nE3 verdict: {verdict['E3_verdict']}")
    print("  single-round references (P(bet) trials 1-5): "
          + ", ".join(f"cue {c:.2f} -> {ref_by_cue[c]:.3f}" for c in grid))
    print(f"  rounds after the first: {verdict['n_rounds_after_first']} | "
          f"following previous cue: {verdict['n_following_previous_cue']} | "
          f"deviating beyond {OWN_CUE_TOLERANCE}: {verdict['n_deviating_beyond_tolerance']}")


def plot(diag_root: Path, frame: pd.DataFrame, groups) -> None:
    apply_style()
    group = groups[0]
    sub = frame[frame.group == group]
    seeds = sorted(sub.seed.unique())
    refs = sub[sub.variant.str.startswith("E3_ref")].groupby("cue")["pbet_t1_5"].mean()
    ref_by_cue = {float(k): float(v) for k, v in refs.items()}
    grid = sorted(ref_by_cue)

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.2))

    # --- E1: cue gain with the carried state fixed -------------------------
    ax = axes[0]
    e1 = sub[sub.variant.str.startswith("E1_")]
    e1 = e1[e1.round_id == e1.round_id.max()]
    for i, seed in enumerate(seeds):
        pts = e1[e1.seed == seed].sort_values("cue")
        ax.plot(pts["cue"], pts["pbet_t1_5"], color=SERIES[i], lw=1.6, marker="o", ms=4.5)
    ax.plot(grid, [ref_by_cue[c] for c in grid], color=INK_MUTED, lw=1.2, ls="--", marker="s", ms=4.5)
    ax.annotate("single round from a blank state\n(same cue, no carried state)",
                xy=(grid[1], ref_by_cue[grid[1]]), xytext=(0, -28), textcoords="offset points",
                fontsize=6.8, color=INK_MUTED, ha="center")
    ax.set_xticks(grid)
    ax.set_ylabel("P(bet), trials 1-5")
    ax.set_title(f"E1  Cue gain with the carried state held fixed  ({group})", fontsize=9,
                 color=INK_PRIMARY)

    # --- E2: preceding-cue dose, conflict round fixed ----------------------
    ax = axes[1]
    e2 = sub[sub.variant.str.startswith("E2_")]
    order = ["E2_preNone"] + [f"E2_pre{c:.2f}" for c in grid]
    conflict_cue = float(e2[e2.variant == "E2_preNone"]["cue"].iloc[0])
    for i, seed in enumerate(seeds):
        ys = []
        for name in order:
            sel = e2[(e2.seed == seed) & (e2.variant == name)]
            ys.append(float(sel[sel.round_id == sel.round_id.max()]["pbet_t1_5"].iloc[0]))
        ax.plot(range(len(order)), ys, color=SERIES[i], lw=1.6, marker="o", ms=4.5)
    ax.axhline(ref_by_cue[min(grid, key=lambda c: abs(c - conflict_cue))],
               color=INK_MUTED, lw=1.2, ls="--")
    ax.annotate("single-round reference at the conflict cue", xy=(0.02, 0.93),
                xycoords="axes fraction", fontsize=6.8, color=INK_MUTED, va="top")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(["none\n(conflict_only)"] + [f"{c:.2f}" for c in grid])
    ax.set_xlabel("cue of the preceding round")
    ax.set_ylabel("P(bet), trials 1-5")
    ax.set_title(f"E2  Preceding-cue dose, conflict cue fixed at {conflict_cue:.2f}  ({group})",
                 fontsize=9, color=INK_PRIMARY)

    # --- E3: round independence -------------------------------------------
    ax = axes[2]
    e3 = sub[sub.variant.isin(E3_ORDERS)]
    for cue in grid:
        ax.axhline(ref_by_cue[cue], color=INK_MUTED, lw=0.9, ls=":")
        ax.annotate(f"single-round ref @ cue {cue:.2f}", xy=(3.62, ref_by_cue[cue]), fontsize=6.3,
                    color=INK_MUTED, va="center")
    ticks = sorted(e3.round_id.unique())
    cues = [float(e3[(e3.round_id == r) & (e3.variant == "E3_asc")]["cue"].iloc[0]) for r in ticks]
    for i, seed in enumerate(seeds):
        for variant, style, xoff in (("E3_asc", "-", -0.07), ("E3_desc", (0, (4, 2)), 0.07)):
            pts = e3[(e3.seed == seed) & (e3.variant == variant)].sort_values("round_id")
            if not len(pts):
                continue
            ax.plot(pts["round_id"], pts["pbet_t1_5"], color=SERIES[i], lw=1.1, alpha=0.55, ls=style,
                    marker="o", ms=4)
            # Hollow marker = where this round *should* open if it followed its own cue.
            own = [ref_by_cue[min(grid, key=lambda c: abs(c - cu))] for cu in pts["cue"]]
            ax.plot(pts["round_id"] + xoff, own, ls="none", marker="D", ms=5.5,
                    markerfacecolor="none", markeredgecolor=SERIES[i], markeredgewidth=1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"round {r}\ncue {c:.2f}" for r, c in zip(ticks, cues)])
    ax.set_xlim(0.7, 4.15)
    ax.set_ylabel("P(bet), trials 1-5")
    ax.set_title(f"E3  Does each round follow its own cue, or lag one round?  ({group})", fontsize=9,
                 color=INK_PRIMARY)
    ax.annotate("hollow diamond = that round's own-cue target", xy=(0.02, 0.04),
                xycoords="axes fraction", fontsize=6.6, color=INK_MUTED)

    handles = [plt.Line2D([], [], color=SERIES[i], lw=1.6, marker="o", ms=4.5, label=f"seed {s}")
               for i, s in enumerate(seeds)]
    handles.append(plt.Line2D([], [], color=INK_MUTED, lw=1.2, ls="--", marker="s", ms=4.5,
                              label="single-round reference (blank state)"))
    handles.append(plt.Line2D([], [], color=INK_MUTED, lw=1.1, alpha=0.55, marker="o", ms=4,
                              label="E3 observed (asc / desc)"))
    handles.append(plt.Line2D([], [], color=INK_MUTED, ls="none", marker="D", ms=5.5,
                              markerfacecolor="none", label="E3 own-cue target"))
    finish(
        fig,
        title="Round-structure diagnostic: how much leverage does the one-shot cue have inside an episode?\n"
              "E1/E2 hold one factor fixed and sweep the other; E3 is the decisive test of how training shapes later rounds",
        legend_handles=handles,
        ncol=5,
    )
    save(fig, diag_root.parent / "figures" / "diag_round_structure.png")


if __name__ == "__main__":
    main()
