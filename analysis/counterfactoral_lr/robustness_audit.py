"""Independent, seed-level audit of saved direct-actor counterfactual rows.

Run after counterfactual_actor_pbet_lr.py has written heldout and all_episodes
subdirectories. This script never loads human choices or changes model weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WINDOWS = {"early_1_20": (1, 20), "late_51_119": (51, 119), "late_90_119": (90, 119)}
EPS = (0.005, 0.01, 0.02, 0.05)
BINS = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
COLUMNS = ["seed", "group", "episode_id", "trial_id", "p", "p_pos", "p_neg",
           "delta_pos", "delta_neg", "alpha_pos", "alpha_neg", "A", "G", "valid_ratio"]


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def load_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=COLUMNS, dtype={"group": "category"})
    if frame.duplicated(["seed", "group", "episode_id", "trial_id"]).any():
        raise ValueError(f"duplicate episode/trial rows: {path}")
    if not frame["trial_id"].between(1, 119).all():
        raise ValueError(f"unexpected trial range: {path}")
    if not np.isfinite(frame[["p", "p_pos", "p_neg"]].to_numpy()).all():
        raise ValueError(f"nonfinite probability: {path}")
    return frame


def independently_check_rows(frame: pd.DataFrame, heldout: bool) -> dict:
    p, pos, neg = (frame[c].to_numpy(float) for c in ("p", "p_pos", "p_neg"))
    dp, dn = (frame[c].to_numpy(float) for c in ("delta_pos", "delta_neg"))
    valid = (p >= 0.01) & (p <= 0.99)
    saved_valid = frame["valid_ratio"].to_numpy(bool)
    if not np.array_equal(valid, saved_valid):
        raise ValueError("saved validity mask differs from independent formula")
    errors = {
        "delta_pos": float(np.max(np.abs(dp - (pos - p)))),
        "delta_neg": float(np.max(np.abs(dn - (neg - p)))),
        "gain": float(np.max(np.abs(frame["G"].to_numpy(float) - (pos - neg)))),
        "alpha_pos": float(np.max(np.abs(frame.loc[valid, "alpha_pos"].to_numpy(float) - dp[valid] / (1 - p[valid])))),
        "alpha_neg": float(np.max(np.abs(frame.loc[valid, "alpha_neg"].to_numpy(float) + dn[valid] / p[valid]))),
        "asymmetry": float(np.max(np.abs(frame.loc[valid, "A"].to_numpy(float) - (dp[valid] / (1 - p[valid]) + dn[valid] / p[valid])))),
    }
    if max(errors.values()) > 1e-12:
        raise ValueError(f"formula mismatch: {errors}")
    counts = frame.groupby(["seed", "group"], observed=True).agg(
        rows=("trial_id", "size"), episodes=("episode_id", "nunique"),
        min_episode=("episode_id", "min"), max_episode=("episode_id", "max"))
    expected = (60, 140, 199) if heldout else (200, 0, 199)
    if len(counts) != 120 or not ((counts.rows == expected[0] * 119) &
                                  (counts.episodes == expected[0]) &
                                  (counts.min_episode == expected[1]) &
                                  (counts.max_episode == expected[2])).all():
        raise ValueError(f"incomplete or wrong episode grid: {counts.describe()}")
    return {"rows": int(len(frame)), "cells": len(counts), "valid_fraction": float(valid.mean()),
            "p_min": float(p.min()), "p_max": float(p.max()), "max_formula_error": errors}


def summarize(frame: pd.DataFrame, source: str, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows, summary_rows = [], []
    for eps in EPS:
        work = frame[["seed", "group", "trial_id", "p", "delta_pos", "delta_neg"]].copy()
        good = work["p"].between(eps, 1 - eps)
        work["alpha_pos"] = np.where(good, work["delta_pos"] / (1 - work["p"]), np.nan)
        work["alpha_neg"] = np.where(good, -work["delta_neg"] / work["p"], np.nan)
        work["A"] = work["alpha_pos"] - work["alpha_neg"]
        work["valid"] = good.astype(float)
        for window, (lo, hi) in WINDOWS.items():
            part = work[work.trial_id.between(lo, hi)]
            per_seed = part.groupby(["seed", "group"], observed=True)[
                ["alpha_pos", "alpha_neg", "A", "delta_pos", "delta_neg", "valid"]
            ].mean().reset_index()
            per_seed.insert(0, "source", source)
            per_seed.insert(1, "eps", eps)
            per_seed.insert(2, "window", window)
            seed_rows.append(per_seed)
            for group, block in per_seed.groupby("group", observed=True):
                row = {"source": source, "eps": eps, "window": window, "group": group,
                       "n_seeds": len(block), "retained_fraction": float(block.valid.mean())}
                for metric in ("alpha_pos", "alpha_neg", "A", "delta_pos", "delta_neg"):
                    vals = block[metric].dropna().to_numpy(float)
                    row[f"{metric}_mean"] = float(vals.mean()) if len(vals) else np.nan
                    row[f"{metric}_positive_seeds"] = int((vals > 0).sum())
                    row[f"{metric}_negative_seeds"] = int((vals < 0).sum())
                    if len(vals) >= 2:
                        row[f"{metric}_boot_lo"], row[f"{metric}_boot_hi"] = bootstrap_ci(vals, rng)
                        loo = (vals.sum() - vals) / (len(vals) - 1)
                        row[f"{metric}_loo_min"] = float(loo.min())
                        row[f"{metric}_loo_max"] = float(loo.max())
                    else:
                        row[f"{metric}_boot_lo"] = row[f"{metric}_boot_hi"] = np.nan
                        row[f"{metric}_loo_min"] = row[f"{metric}_loo_max"] = np.nan
                summary_rows.append(row)
    return pd.concat(seed_rows, ignore_index=True), pd.DataFrame(summary_rows)


def p_bins(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame[["seed", "group", "p", "delta_pos", "delta_neg"]].copy()
    work["p_bin"] = pd.cut(work.p, BINS, include_lowest=True)
    work["valid_001"] = work.p.between(0.01, 0.99)
    work["alpha_pos"] = np.where(work.valid_001, work.delta_pos / (1 - work.p), np.nan)
    work["alpha_neg"] = np.where(work.valid_001, -work.delta_neg / work.p, np.nan)
    work["A"] = work.alpha_pos - work.alpha_neg
    return work.groupby(["group", "p_bin"], observed=True).agg(
        rows=("p", "size"), p_mean=("p", "mean"), valid_fraction=("valid_001", "mean"),
        delta_pos=("delta_pos", "mean"), delta_neg=("delta_neg", "mean"),
        alpha_pos=("alpha_pos", "mean"), alpha_neg=("alpha_neg", "mean"), A=("A", "mean")
    ).reset_index()


def early_late_contrasts(seeds: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (source, eps, group), block in seeds.groupby(["source", "eps", "group"], observed=True):
        wide = block.pivot(index="seed", columns="window", values="A")
        paired = wide[["early_1_20", "late_51_119"]].dropna()
        diff = (paired.late_51_119 - paired.early_1_20).to_numpy(float)
        lo, hi = bootstrap_ci(diff, rng) if len(diff) >= 2 else (np.nan, np.nan)
        rows.append({"source": source, "eps": eps, "group": group, "n_seeds": len(diff),
                     "late_minus_early_A": float(diff.mean()) if len(diff) else np.nan,
                     "positive_seeds": int((diff > 0).sum()), "negative_seeds": int((diff < 0).sum()),
                     "boot_lo": lo, "boot_hi": hi})
    return pd.DataFrame(rows)


def compare_table(new: Path, old: Path) -> dict:
    a, b = pd.read_csv(new), pd.read_csv(old)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        raise ValueError(f"table shape/columns differ: {new.name}")
    numeric = list(a.select_dtypes(include="number").columns)
    string = [c for c in a.columns if c not in numeric]
    if not a[string].fillna("").equals(b[string].fillna("")):
        raise ValueError(f"table labels differ: {new.name}")
    differences = np.abs(a[numeric].to_numpy(float) - b[numeric].to_numpy(float))
    if not np.array_equal(np.isnan(a[numeric]), np.isnan(b[numeric])):
        raise ValueError(f"table NaN pattern differs: {new.name}")
    return {"rows": len(a), "max_abs_numeric_difference": float(np.nanmax(differences))}


def check_trial_protocol(runs: Path) -> dict:
    files = sorted((runs / "frozen_tests" / "h16_b200" / "conflict_only").glob("*/mac*/trials.csv"))
    if len(files) != 120:
        raise ValueError(f"expected 120 trial files, found {len(files)}")
    total_rows = no_bet = 0
    max_reward_error = max_prev_error = max_token_error = 0.0
    for path in files:
        cols = ["episode_id", "round_id", "trial_id", "action", "outcome", "reward",
                "prev_action", "prev_outcome", "prev_token_delta", "current_total_tokens"]
        x = pd.read_csv(path, usecols=cols).sort_values(["episode_id", "round_id", "trial_id"])
        if len(x) != 24000 or x.episode_id.nunique() != 200 or x.round_id.nunique() != 1:
            raise ValueError(f"unexpected episode or round shape: {path}")
        if not np.array_equal(x.trial_id.to_numpy().reshape(200, 120),
                              np.tile(np.arange(1, 121), (200, 1))):
            raise ValueError(f"unexpected trial grid: {path}")
        for col in ("action", "outcome"):
            if not x[col].isin([0, 1]).all():
                raise ValueError(f"nonbinary {col}: {path}")
        action = x.action.to_numpy(float).reshape(200, 120)
        outcome = x.outcome.to_numpy(float).reshape(200, 120)
        reward = x.reward.to_numpy(float).reshape(200, 120)
        expected_reward = action * (2 * outcome - 1)
        max_reward_error = max(max_reward_error, float(np.max(np.abs(reward - expected_reward))))
        for actual, previous in (("prev_action", action), ("prev_outcome", outcome),
                                 ("prev_token_delta", reward)):
            observed = x[actual].to_numpy(float).reshape(200, 120)
            expected = np.column_stack([np.zeros(200), previous[:, :-1]])
            max_prev_error = max(max_prev_error, float(np.max(np.abs(observed - expected))))
        expected_tokens = 100 + np.column_stack([np.zeros(200), np.cumsum(reward[:, :-1], axis=1)])
        tokens = x.current_total_tokens.to_numpy(float).reshape(200, 120)
        max_token_error = max(max_token_error, float(np.max(np.abs(tokens - expected_tokens))))
        total_rows += len(x)
        no_bet += int((action == 0).sum())
    if max(max_reward_error, max_prev_error, max_token_error) != 0:
        raise ValueError("causal trial protocol check failed")
    return {"files": len(files), "rows": total_rows, "no_bet_rows": no_bet,
            "max_reward_error": max_reward_error, "max_previous_field_error": max_prev_error,
            "max_token_error": max_token_error,
            "note": "Conflict-only exports contain one 120-feedback-trial round; no-feedback phase is absent."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rng = np.random.default_rng(20261109)
    checks, seed_frames, summary_frames = {}, [], []
    for source in ("heldout", "all_episodes"):
        path = root / source / "metrics" / "actor_pbet_lr_episode.csv"
        frame = load_rows(path)
        checks[source] = independently_check_rows(frame, source == "heldout")
        seed, summary = summarize(frame, source, rng)
        seed_frames.append(seed)
        summary_frames.append(summary)
        if source == "heldout":
            p_bins(frame).to_csv(root / "heldout_p_bins.csv", index=False)
        del frame
    seeds = pd.concat(seed_frames, ignore_index=True)
    summary = pd.concat(summary_frames, ignore_index=True)
    seeds.to_csv(root / "seed_window_sensitivity.csv", index=False)
    summary.to_csv(root / "robustness_summary.csv", index=False)
    early_late_contrasts(seeds, rng).to_csv(root / "early_late_contrasts.csv", index=False)
    comparisons = {}
    for name in ("seed_trial", "trial_summary", "window_summary", "denom_sensitivity", "diagnostics"):
        filename = f"actor_pbet_lr_{name}.csv"
        comparisons[name] = compare_table(root / "heldout" / "metrics" / filename,
                                          args.published / filename)
    checks["published_comparison"] = comparisons
    checks["trial_protocol"] = check_trial_protocol(args.runs.resolve())
    (root / "audit_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
