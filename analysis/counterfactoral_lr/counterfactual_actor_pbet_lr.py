"""Counterfactual effective update rates read directly from the trained actor.

For each stored pre-action hidden state h_t, replay the next input twice while
forcing the previous outcome to 1 or 0. The trained actor, rather than a fitted
decoder, reads all three states:

    p_t    = sigmoid(actor(h_t))
    p_pos  = sigmoid(actor(GRU(x_{t+1}[outcome=1], h_t)))
    p_neg  = sigmoid(actor(GRU(x_{t+1}[outcome=0], h_t)))
    alpha+ = (p_pos - p_t) / (1 - p_t)
    alpha- = (p_t - p_neg) / p_t

The ratios are reported as policy effective update rates. They are undefined
near saturated policy probabilities, so the primary analysis excludes p below
0.01 or above 0.99 and reports threshold-sensitivity results.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from analysis.counterfactoral_lr.pbet_stats import (  # noqa: E402
    WINDOWS,
    ci95,
    fdr_bh,
    one_sample,
    r2_through_origin,
    sem,
)
from analysis.formal_behav_analysis.loading import GROUPS, episode_split  # noqa: E402
from analysis.counterfactoral_lr.replay_gate import (  # noqa: E402
    _batch_counterfactual_inputs,
    _batch_inputs,
    require_single_thread,
)
from formal_osla.test_runner import load_checkpoint  # noqa: E402
from scripts.shared.prerun_style import (  # noqa: E402
    INK_MUTED,
    INK_PRIMARY,
    SERIES,
    apply_style,
    finish,
    save,
)


DEFAULT_SEEDS = tuple(range(20260804, 20260834))
DEFAULT_BATCH_RUNS = ROOT / "outputs" / "output_formal" / "runs"
DEFAULT_ANALYSIS_ROOT = ROOT / "analysis" / "counterfactoral_lr" / "outputs" / "result"
SENSITIVITY_EPS = (0.005, 0.01, 0.02, 0.05)
SENSITIVITY_WINDOWS = {
    "trial_1_20": (1, 20),
    "trial_51_119": (51, 119),
    "full_1_119": (1, 119),
}


def actor_probability(model: torch.nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the frozen policy head to a pre-action hidden state."""
    return torch.sigmoid(model.actor(model.dropout(hidden)).squeeze(-1))


def run_cell(
    *,
    batch_runs: Path,
    arm: str,
    protocol: str,
    seed: int,
    group: str,
    denom_eps: float,
    use_heldout_only: bool,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    network_id = f"formal_osla_seed{seed}"
    run_dir = batch_runs / "frozen_tests" / arm / protocol / network_id / group
    checkpoint = batch_runs / "arms" / arm / "runs" / network_id / "best.pt"
    if not (run_dir / "trials.csv").exists():
        raise FileNotFoundError(run_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model, _config, _checkpoint = load_checkpoint(checkpoint, device=device)
    trials = pd.read_csv(run_dir / "trials.csv")
    with np.load(run_dir / "hidden.npz") as data:
        hidden = np.asarray(data["hidden"], dtype=np.float32)

    n_episodes, n_trials, _hidden_size = hidden.shape
    ordered = trials.sort_values(["episode_id", "round_id", "trial_id"]).reset_index(drop=True)
    if len(ordered) != n_episodes * n_trials:
        raise ValueError(f"{run_dir}: trials.csv and hidden.npz disagree on row count")
    feedback_trials = int(ordered["trial_id"].max())
    if n_trials != feedback_trials:
        raise ValueError(f"{run_dir}: expected one round of {feedback_trials}, got {n_trials}")

    disk_p = ordered["choice_probability"].to_numpy(float).reshape(n_episodes, n_trials)
    analysis_episode = ~episode_split(n_episodes) if use_heldout_only else np.ones(n_episodes, dtype=bool)
    state = torch.tensor(hidden, dtype=torch.float32, device=device)

    episode_rows: list[dict] = []
    seed_rows: list[dict] = []
    with torch.no_grad():
        for trial in range(1, n_trials):
            next_rows = ordered[ordered["trial_id"] == trial + 1]
            h_t = state[:, trial - 1, :]
            target_next = state[:, trial, :]

            h_replay, _, p_replay_t = model.step(
                _batch_inputs(next_rows, device=device, feedback_trials=feedback_trials),
                h_t,
            )
            h_pos, _, p_pos_t = model.step(
                _batch_counterfactual_inputs(
                    next_rows, device=device, feedback_trials=feedback_trials, outcome=1.0
                ),
                h_t,
            )
            h_neg, _, p_neg_t = model.step(
                _batch_counterfactual_inputs(
                    next_rows, device=device, feedback_trials=feedback_trials, outcome=0.0
                ),
                h_t,
            )
            p_t = actor_probability(model, h_t)

            p = p_t.cpu().numpy().astype(float)
            p_pos = p_pos_t.cpu().numpy().astype(float)
            p_neg = p_neg_t.cpu().numpy().astype(float)
            p_replay = p_replay_t.cpu().numpy().astype(float)
            delta_pos = p_pos - p
            delta_neg = p_neg - p
            gain = p_pos - p_neg
            common_drift = p * delta_pos + (1.0 - p) * delta_neg

            finite = np.isfinite(p) & np.isfinite(p_pos) & np.isfinite(p_neg)
            valid = analysis_episode & finite & (p >= denom_eps) & (p <= 1.0 - denom_eps)
            alpha_pos = np.full_like(p, np.nan)
            alpha_neg = np.full_like(p, np.nan)
            alpha_pos[valid] = delta_pos[valid] / (1.0 - p[valid])
            alpha_neg[valid] = -delta_neg[valid] / p[valid]
            asymmetry = alpha_pos - alpha_neg
            gain_from_alpha = alpha_pos * (1.0 - p) + alpha_neg * p

            replay_hidden_error = float((h_replay - target_next).abs().max())
            actor_disk_error = float(np.max(np.abs(p - disk_p[:, trial - 1])))
            replay_p_error = float(np.max(np.abs(p_replay - disk_p[:, trial])))
            identity_error = (
                float(np.max(np.abs(gain[valid] - gain_from_alpha[valid]))) if valid.any() else np.nan
            )

            for episode_id in np.where(analysis_episode)[0]:
                episode_rows.append(
                    {
                        "seed": seed,
                        "network_id": network_id,
                        "group": group,
                        "protocol": protocol,
                        "episode_id": int(episode_id),
                        "trial_id": trial,
                        "p": float(p[episode_id]),
                        "p_pos": float(p_pos[episode_id]),
                        "p_neg": float(p_neg[episode_id]),
                        "delta_pos": float(delta_pos[episode_id]),
                        "delta_neg": float(delta_neg[episode_id]),
                        "alpha_pos": float(alpha_pos[episode_id]),
                        "alpha_neg": float(alpha_neg[episode_id]),
                        "A": float(asymmetry[episode_id]),
                        "G": float(gain[episode_id]),
                        "G_from_alpha": float(gain_from_alpha[episode_id]),
                        "C": float(common_drift[episode_id]),
                        "valid_ratio": bool(valid[episode_id]),
                    }
                )

            selected = analysis_episode
            seed_rows.append(
                {
                    "seed": seed,
                    "network_id": network_id,
                    "group": group,
                    "protocol": protocol,
                    "trial_id": trial,
                    "n_episodes": int(selected.sum()),
                    "n_valid": int(valid.sum()),
                    "valid_fraction": float(valid.sum() / selected.sum()),
                    "p_mean": float(np.mean(p[selected])),
                    "p_sd": float(np.std(p[selected])),
                    "delta_pos": float(np.mean(delta_pos[selected])),
                    "delta_neg": float(np.mean(delta_neg[selected])),
                    "alpha_pos": float(np.mean(alpha_pos[valid])) if valid.any() else np.nan,
                    "alpha_neg": float(np.mean(alpha_neg[valid])) if valid.any() else np.nan,
                    "A": float(np.mean(asymmetry[valid])) if valid.any() else np.nan,
                    "G": float(np.mean(gain[selected])),
                    "C": float(np.mean(common_drift[selected])),
                    "frac_negative_alpha_pos": float(np.mean(alpha_pos[valid] < 0.0)) if valid.any() else np.nan,
                    "frac_negative_alpha_neg": float(np.mean(alpha_neg[valid] < 0.0)) if valid.any() else np.nan,
                    "max_abs_actor_disk_error": actor_disk_error,
                    "max_abs_replay_hidden_error": replay_hidden_error,
                    "max_abs_replay_p_error": replay_p_error,
                    "max_abs_G_identity_error": identity_error,
                }
            )
    return episode_rows, seed_rows


def summarize_trials(seed_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "p_mean",
        "p_sd",
        "delta_pos",
        "delta_neg",
        "alpha_pos",
        "alpha_neg",
        "A",
        "G",
        "C",
        "valid_fraction",
        "frac_negative_alpha_pos",
        "frac_negative_alpha_neg",
    ]
    rows = []
    for (group, trial), block in seed_df.groupby(["group", "trial_id"], sort=True):
        row = {"group": group, "trial_id": int(trial), "n_seeds": int(block["seed"].nunique())}
        for metric in metrics:
            values = block[metric].to_numpy(float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sem"] = sem(values)
            row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = ci95(values)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_windows(seed_df: pd.DataFrame, episode_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for window, (lo, hi) in WINDOWS.items():
            block = seed_df[seed_df["group"].eq(group) & seed_df["trial_id"].between(lo, hi)]
            if block.empty:
                continue
            per_seed = block.groupby("seed", as_index=False)[
                ["alpha_pos", "alpha_neg", "A", "G", "C", "valid_fraction", "delta_pos", "delta_neg"]
            ].mean()
            ep_block = episode_df[
                episode_df["group"].eq(group) & episode_df["trial_id"].between(lo, hi)
            ]
            rw_rows = []
            for seed, ep_seed in ep_block.groupby("seed"):
                ok = ep_seed["valid_ratio"].astype(bool).to_numpy()
                slope_pos, r2_pos = r2_through_origin(
                    1.0 - ep_seed.loc[ok, "p"].to_numpy(float),
                    ep_seed.loc[ok, "delta_pos"].to_numpy(float),
                )
                slope_neg, r2_neg = r2_through_origin(
                    ep_seed.loc[ok, "p"].to_numpy(float),
                    -ep_seed.loc[ok, "delta_neg"].to_numpy(float),
                )
                rw_rows.append(
                    {
                        "seed": seed,
                        "rw_slope_pos": slope_pos,
                        "rw_slope_neg": slope_neg,
                        "rw_r2_pos": r2_pos,
                        "rw_r2_neg": r2_neg,
                    }
                )
            rw_frame = pd.DataFrame(
                rw_rows,
                columns=["seed", "rw_slope_pos", "rw_slope_neg", "rw_r2_pos", "rw_r2_neg"],
            )
            merged = per_seed.merge(rw_frame, on="seed", how="left")
            for metric in [
                "alpha_pos",
                "alpha_neg",
                "A",
                "G",
                "C",
                "valid_fraction",
                "delta_pos",
                "delta_neg",
                "rw_slope_pos",
                "rw_slope_neg",
                "rw_r2_pos",
                "rw_r2_neg",
            ]:
                values = merged[metric].to_numpy(float)
                if metric == "valid_fraction":
                    values = values[np.isfinite(values)]
                    ci_lo, ci_hi = ci95(values)
                    result = {
                        "n": int(values.size),
                        "mean": float(values.mean()) if values.size else np.nan,
                        "sem": sem(values),
                        "t": np.nan,
                        "p": np.nan,
                        "ci_lo": ci_lo,
                        "ci_hi": ci_hi,
                    }
                else:
                    result = one_sample(values, 0.0)
                result.update({"group": group, "window": window, "metric": metric})
                rows.append(result)
    result = pd.DataFrame(rows)
    result["q_fdr"] = fdr_bh(result["p"].tolist())
    return result[["group", "window", "metric", "n", "mean", "sem", "ci_lo", "ci_hi", "t", "p", "q_fdr"]]


def summarize_sensitivity(episode_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for eps in SENSITIVITY_EPS:
        valid = episode_df["p"].between(eps, 1.0 - eps)
        work = episode_df.loc[valid, ["seed", "group", "trial_id", "p", "delta_pos", "delta_neg"]].copy()
        work["alpha_pos"] = work["delta_pos"] / (1.0 - work["p"])
        work["alpha_neg"] = -work["delta_neg"] / work["p"]
        for window, (lo, hi) in SENSITIVITY_WINDOWS.items():
            part = work[work["trial_id"].between(lo, hi)]
            per_seed = part.groupby(["seed", "group"], as_index=False)[["alpha_pos", "alpha_neg"]].mean()
            total = episode_df[episode_df["trial_id"].between(lo, hi)].groupby(["seed", "group"]).size()
            kept = part.groupby(["seed", "group"]).size()
            fractions = (kept / total).rename("valid_fraction").reset_index()
            per_seed = per_seed.merge(fractions, on=["seed", "group"], how="left")
            for group in GROUPS:
                block = per_seed[per_seed["group"].eq(group)]
                if block.empty:
                    continue
                for metric in ["alpha_pos", "alpha_neg", "valid_fraction"]:
                    values = block[metric].to_numpy(float)
                    lo_ci, hi_ci = ci95(values)
                    rows.append(
                        {
                            "denom_eps": eps,
                            "window": window,
                            "group": group,
                            "metric": metric,
                            "n_seeds": int(np.isfinite(values).sum()),
                            "mean": float(np.nanmean(values)),
                            "sem": sem(values),
                            "ci_lo": lo_ci,
                            "ci_hi": hi_ci,
                        }
                    )
    return pd.DataFrame(rows)


def diagnostics(seed_df: pd.DataFrame, episode_df: pd.DataFrame, denom_eps: float) -> pd.DataFrame:
    valid = episode_df["valid_ratio"].astype(bool).to_numpy()
    rows = [
        ("denom_eps", denom_eps),
        ("valid_fraction", float(valid.mean())),
        ("p_min", float(episode_df["p"].min())),
        ("p_max", float(episode_df["p"].max())),
        ("alpha_pos_median", float(episode_df.loc[valid, "alpha_pos"].median())),
        ("alpha_neg_median", float(episode_df.loc[valid, "alpha_neg"].median())),
        ("alpha_pos_p01", float(episode_df.loc[valid, "alpha_pos"].quantile(0.01))),
        ("alpha_pos_p99", float(episode_df.loc[valid, "alpha_pos"].quantile(0.99))),
        ("alpha_neg_p01", float(episode_df.loc[valid, "alpha_neg"].quantile(0.01))),
        ("alpha_neg_p99", float(episode_df.loc[valid, "alpha_neg"].quantile(0.99))),
        ("max_abs_actor_disk_error", float(seed_df["max_abs_actor_disk_error"].max())),
        ("max_abs_replay_hidden_error", float(seed_df["max_abs_replay_hidden_error"].max())),
        ("max_abs_replay_p_error", float(seed_df["max_abs_replay_p_error"].max())),
        ("max_abs_G_identity_error", float(seed_df["max_abs_G_identity_error"].max())),
    ]
    return pd.DataFrame(rows, columns=["diagnostic", "value"])


def plot_alpha_timeseries(summary: pd.DataFrame, output_root: Path, denom_eps: float) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.8), sharex=True, sharey=False)
    for ax, group in zip(axes.ravel(), GROUPS):
        block = summary[summary["group"].eq(group)]
        x = block["trial_id"].to_numpy()
        for metric, label, color, linestyle in [
            ("alpha_pos", "positive outcome", SERIES[0], "-"),
            ("alpha_neg", "negative outcome", SERIES[1], "--"),
        ]:
            mean = block[f"{metric}_mean"].to_numpy()
            lo = block[f"{metric}_ci_lo"].to_numpy()
            hi = block[f"{metric}_ci_hi"].to_numpy()
            ax.plot(x, mean, color=color, ls=linestyle, lw=1.5, label=label)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.axhline(0.0, color=INK_MUTED, lw=0.8, ls=":")
        ax.axvspan(1, 20, color=SERIES[2], alpha=0.07, linewidth=0)
        ax.set_title(group, color=INK_PRIMARY)
        ax.set_xlim(1, 119)
        ax.set_xlabel("Trial within conflict round")
        ax.set_ylabel("Actor p-bet effective update rate")
    axes[0, 0].legend(loc="upper right")
    finish(
        fig,
        title=(
            "Counterfactual update rates from the frozen actor\n"
            f"mean and 95% CI across network seeds; ratios require {denom_eps:.3f} <= p <= {1-denom_eps:.3f}"
        ),
        legend_handles=None,
    )
    save(fig, output_root / "figures" / "actor_pbet_lr_timeseries.png")


def plot_diagnostics(summary: pd.DataFrame, output_root: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(4, 2, figsize=(10.8, 10.2), sharex=True)
    for row, group in enumerate(GROUPS):
        block = summary[summary["group"].eq(group)]
        x = block["trial_id"].to_numpy()
        ax_delta, ax_valid = axes[row]
        for metric, label, color, linestyle in [
            ("delta_pos", "p+ - p", SERIES[0], "-"),
            ("delta_neg", "p- - p", SERIES[1], "--"),
        ]:
            ax_delta.plot(
                x,
                block[f"{metric}_mean"],
                color=color,
                ls=linestyle,
                lw=1.35,
                label=label,
            )
        ax_delta.axhline(0.0, color=INK_MUTED, lw=0.8, ls=":")
        ax_delta.set_title(f"{group}: unnormalized actor change")
        ax_delta.set_ylabel("Change in p-bet")
        ax_valid.plot(x, block["valid_fraction_mean"], color=SERIES[2], lw=1.35)
        ax_valid.set_ylim(0.0, 1.03)
        ax_valid.set_title(f"{group}: retained observations")
        ax_valid.set_ylabel("Valid fraction")
        for ax in (ax_delta, ax_valid):
            ax.set_xlim(1, 119)
            ax.set_xlabel("Trial within conflict round")
    axes[0, 0].legend(loc="upper right")
    finish(
        fig,
        title="Direct-actor counterfactual diagnostics\nraw probability changes and denominator-filter retention",
        legend_handles=None,
    )
    save(fig, output_root / "figures" / "actor_pbet_lr_diagnostics.png")


def write_report(
    output_root: Path,
    window_summary: pd.DataFrame,
    diagnostic_df: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    denom_eps: float,
    use_heldout_only: bool,
) -> None:
    diagnostic = dict(zip(diagnostic_df["diagnostic"], diagnostic_df["value"]))
    lines = [
        "# Direct-actor counterfactual p-bet update rates",
        "",
        "The readout is the frozen network's own actor probability. No post-hoc decoder is used.",
        "The resulting ratios are policy effective update rates, not independently identified latent-belief learning rates.",
        "",
        f"- Episodes: {'held-out side of the deterministic episode split' if use_heldout_only else 'all episodes'}.",
        f"- Primary denominator threshold: `{denom_eps:g}`.",
        f"- Overall retained fraction: `{diagnostic['valid_fraction']:.4f}`.",
        f"- Max actor-vs-disk p-bet error: `{diagnostic['max_abs_actor_disk_error']:.3e}`.",
        f"- Max natural replay hidden-state error: `{diagnostic['max_abs_replay_hidden_error']:.3e}`.",
        f"- Max natural replay p-bet error: `{diagnostic['max_abs_replay_p_error']:.3e}`.",
        f"- Max gain identity error: `{diagnostic['max_abs_G_identity_error']:.3e}`.",
        "",
        "## Trial 1-20",
        "",
    ]
    view = window_summary[
        window_summary["window"].eq("trial_1_20")
        & window_summary["metric"].isin(["alpha_pos", "alpha_neg", "valid_fraction"])
    ]
    for group in GROUPS:
        lines.append(f"### {group}")
        for _, row in view[view["group"].eq(group)].iterrows():
            lines.append(
                f"- {row['metric']}: mean={row['mean']:+.5f}, 95% CI "
                f"[{row['ci_lo']:+.5f}, {row['ci_hi']:+.5f}], n={int(row['n'])}."
            )
    lines.extend(["", "## Denominator sensitivity", ""])
    sens = sensitivity_df[
        sensitivity_df["window"].eq("full_1_119")
        & sensitivity_df["metric"].isin(["alpha_pos", "alpha_neg", "valid_fraction"])
    ]
    for eps in SENSITIVITY_EPS:
        block = sens[np.isclose(sens["denom_eps"], eps)]
        retained = block[block["metric"].eq("valid_fraction")]["mean"].mean()
        lines.append(f"- eps={eps:g}: mean retained fraction across groups={retained:.4f}.")
    lines.extend(
        [
            "",
            "The raw delta-p trajectories should be inspected alongside the ratios. Stable signs in delta-p with "
            "ratio changes near reduced retention indicate a denominator effect rather than a reversal in policy updating.",
        ]
    )
    (output_root / "metrics" / "actor_pbet_lr_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-runs", type=Path, default=DEFAULT_BATCH_RUNS)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--protocol", default="conflict_only")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--denom-eps",
        type=float,
        default=0.01,
        help="Exclude p below eps or above 1-eps before computing ratio update rates.",
    )
    parser.add_argument("--all-episodes", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.denom_eps < 0.5:
        parser.error("--denom-eps must be between 0 and 0.5")

    require_single_thread()
    batch_runs = args.batch_runs.resolve()
    output_root = args.analysis_root.resolve()
    (output_root / "metrics").mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    arm = f"h{args.hidden_size:02d}_b{args.budget}"

    episode_rows: list[dict] = []
    seed_rows: list[dict] = []
    for seed in args.seeds:
        for group in args.groups:
            erows, srows = run_cell(
                batch_runs=batch_runs,
                arm=arm,
                protocol=args.protocol,
                seed=seed,
                group=group,
                denom_eps=args.denom_eps,
                use_heldout_only=not args.all_episodes,
                device=torch.device("cpu"),
            )
            episode_rows.extend(erows)
            seed_rows.extend(srows)
        print(f"seed {seed} done", flush=True)

    episode_df = pd.DataFrame(episode_rows)
    seed_df = pd.DataFrame(seed_rows)
    trial_summary = summarize_trials(seed_df)
    window_summary = summarize_windows(seed_df, episode_df)
    sensitivity_df = summarize_sensitivity(episode_df)
    diagnostic_df = diagnostics(seed_df, episode_df, args.denom_eps)

    metrics_dir = output_root / "metrics"
    episode_df.to_csv(metrics_dir / "actor_pbet_lr_episode.csv", index=False)
    seed_df.to_csv(metrics_dir / "actor_pbet_lr_seed_trial.csv", index=False)
    trial_summary.to_csv(metrics_dir / "actor_pbet_lr_trial_summary.csv", index=False)
    window_summary.to_csv(metrics_dir / "actor_pbet_lr_window_summary.csv", index=False)
    sensitivity_df.to_csv(metrics_dir / "actor_pbet_lr_denom_sensitivity.csv", index=False)
    diagnostic_df.to_csv(metrics_dir / "actor_pbet_lr_diagnostics.csv", index=False)

    plot_alpha_timeseries(trial_summary, output_root, args.denom_eps)
    plot_diagnostics(trial_summary, output_root)
    write_report(
        output_root,
        window_summary,
        diagnostic_df,
        sensitivity_df,
        args.denom_eps,
        use_heldout_only=not args.all_episodes,
    )

    print("\nDirect-actor counterfactual p-bet analysis complete")
    for _, row in diagnostic_df.iterrows():
        print(f"  {row['diagnostic']}: {row['value']:.6g}")
    key = window_summary[
        window_summary["window"].eq("trial_1_20")
        & window_summary["metric"].isin(["alpha_pos", "alpha_neg"])
    ]
    for _, row in key.iterrows():
        print(f"  {row['group']} {row['metric']}: {row['mean']:+.4f} +/- {row['sem']:.4f}")


if __name__ == "__main__":
    main()
