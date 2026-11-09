"""summarise_fits.py — one consolidated table of the 12 main fits.

Merges the two things that are normally read from separate places:

  * **convergence** -- from `fits/<M>_condA_<group>_final/{parameter_summary.csv,
    diagnostic_summary.csv}`, split into FITTED PARAMETERS and GENERATED QUANTITIES.
    Mixing the two hides real failures: a generated quantity is a deterministic function
    of the parameters and is numerically harder, so a low GQ ESS alone means little.
  * **model comparison** -- from `fits/loo_compare_condA_<group>_final.csv` (PSIS-LOO).
    `elpd_diff` is relative to the best model of THAT group (always M3 here), so it is
    not comparable across rows -- read each group as its own ranking.

Writes `metrics/fit_summary.csv` and prints a markdown table for pasting.

Read-only: no fitting, no writes outside that one CSV.

Run:
    cd formal_run_001/analysis/fitting_NN
    ../../.venv/Scripts/python.exe scripts/summarise_fits.py [--markdown]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FITTING_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_diagnostics import LAYERS, NOT_A_QUANTITY, family  # noqa: E402

FITS_DIR = FITTING_DIR / "fits"
METRICS_DIR = FITTING_DIR / "metrics"
MODELS = ("M1", "M2", "M3")
GROUPS = ("mac1A", "mac1B", "mac2A", "mac2B")
FITTED_BLOCKS = ("parameters", "transformed parameters")

# A parameter above this Rhat is the conventional "did not converge" line; ESS 400 is the
# usual warning threshold and 100 is where mixing is plainly too slow to trust.
RHAT_BAR, ESS_WARN, ESS_BAD = 1.05, 400, 100


def convergence(model: str, group: str) -> dict:
    fit_dir = FITS_DIR / f"{model}_condA_{group}_final"
    with (fit_dir / "parameter_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fitted = set().union(*(LAYERS[model][b] for b in FITTED_BLOCKS)) - NOT_A_QUANTITY
    param = [r for r in rows if family(r["variable"]) in fitted]

    def worst(members, column, want_max):
        values = [(float(r[column]), r["variable"]) for r in members
                  if r[column] not in ("NA", "")]
        return (max(values) if want_max else min(values)) if values else (float("nan"), "")

    gq = [r for r in rows
          if family(r["variable"]) not in fitted and family(r["variable"]) != "lp__"]
    out = {
        "par_max_rhat": worst(param, "rhat", True)[0],
        "par_min_ess": worst(param, "ess_bulk", False)[0],
        "gq_max_rhat": worst(gq, "rhat", True)[0],
        "gq_min_ess": worst(gq, "ess_bulk", False)[0],
        "n_par_rhat_gt_bar": sum(1 for r in param
                                 if r["rhat"] not in ("NA", "") and float(r["rhat"]) > RHAT_BAR),
        "n_par_ess_lt_400": sum(1 for r in param if r["ess_bulk"] not in ("NA", "")
                                and float(r["ess_bulk"]) < ESS_WARN),
    }
    with (fit_dir / "diagnostic_summary.csv").open(newline="", encoding="utf-8") as handle:
        chains = list(csv.DictReader(handle))
    out["divergent"] = sum(int(c["num_divergent"]) for c in chains)
    out["treedepth"] = sum(int(c["num_max_treedepth"]) for c in chains)
    out["ebfmi"] = min(float(c["ebfmi"]) for c in chains)
    return out


def comparison(group: str) -> dict[str, dict]:
    path = FITS_DIR / f"loo_compare_condA_{group}_final.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {r["model"]: r for r in csv.DictReader(handle)}


def verdict(row: dict) -> str:
    """Four levels, so "0 parameters over the bar" and "9 of them" stay distinguishable."""
    if row["n_par_rhat_gt_bar"] > 10 or row["par_min_ess"] < 20:
        return "failed"
    if row["n_par_rhat_gt_bar"] > 0:
        return "poor"
    if row["par_min_ess"] < ESS_BAD or row["n_par_ess_lt_400"] > 5:
        return "marginal"
    if row["par_min_ess"] < ESS_WARN:
        return "marginal"
    return "clean"


def loo_verdict(row: dict) -> str:
    """Separate from `verdict`: A fit can sample perfectly and still have unusable LOO.

    M1 mac1B is exactly that case -- every Rhat and ESS is healthy, but 11 observations
    hit Pareto k = Inf, which means PSIS cannot approximate their LOO predictive at all.
    Folding that into the convergence verdict would hide it.
    """
    if row["n_k_inf"] > 0:
        return f"失效({row['n_k_inf']}×k=Inf)"
    if row["max_k"] > 1.0:
        return "不可靠(k>1)"
    if row["max_k"] > 0.7:
        return f"偏弱({row['n_k_gt_07']}×k>0.7)"
    return "可靠"


def ic_rows() -> dict[tuple[str, str], dict]:
    """LOOIC / WAIC from compare_ic.R, if it has been run.  Absent is fine."""
    path = METRICS_DIR / "ic_comparison.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {(r["model"], r["group"]): r for r in csv.DictReader(handle)}


COLUMNS = ["model", "group", "divergent", "treedepth", "ebfmi",
           "par_max_rhat", "par_min_ess", "n_par_rhat_gt_bar", "n_par_ess_lt_400",
           "gq_max_rhat", "gq_min_ess",
           "elpd_diff", "se_diff", "elpd_loo", "p_loo", "max_k", "n_k_gt_07", "n_k_inf",
           "looic", "se_looic", "waic", "se_waic", "p_waic",
           "verdict", "loo_verdict", "p_waic_over_n", "fit_reliability"]


def build() -> list[dict]:
    out = []
    ic = ic_rows()
    for group in GROUPS:
        cmp_rows = comparison(group)
        for model in MODELS:
            row = {"model": model, "group": group}
            row.update(convergence(model, group))
            extra = ic.get((model, group))
            for col in ("looic", "se_looic", "waic", "se_waic", "p_waic"):
                row[col] = float(extra[col]) if extra else float("nan")
            row["p_waic_over_n"] = (float(extra["p_waic_over_n"])
                                    if extra and "p_waic_over_n" in extra else float("nan"))
            row["fit_reliability"] = (extra.get("fit_reliability", "") if extra else "")
            comp = cmp_rows.get(model, {})
            row["elpd_diff"] = float(comp["elpd_diff"]) if comp else float("nan")
            row["se_diff"] = float(comp["se_diff"]) if comp else float("nan")
            row["elpd_loo"] = float(comp["elpd_loo"]) if comp else float("nan")
            row["p_loo"] = float(comp["p_loo"]) if comp else float("nan")
            row["max_k"] = float(comp["max_k"]) if comp else float("nan")
            row["n_k_gt_07"] = int(comp["n_k_gt_07"]) if comp else -1
            row["n_k_inf"] = int(comp["n_k_inf"]) if comp else -1
            row["verdict"] = verdict(row)
            row["loo_verdict"] = loo_verdict(row)
            # `loo_verdict` is graded here from loo_compare's Pareto-k; `fit_reliability`
            # was graded in R from ic_comparison's.  Two files, two scripts, one number --
            # since compare_ic.R now reads LOO through the same fit$loo() path as
            # compare_loo.R, the verdicts must agree; a mismatch means one file is stale.
            if row["fit_reliability"] and row["fit_reliability"] != row["loo_verdict"]:
                raise SystemExit(
                    f"{model} {group}: ic_comparison.csv says {row['fit_reliability']!r} but "
                    f"loo_compare says {row['loo_verdict']!r} — re-run scripts/compare_ic.R "
                    f"and scripts/compare_loo.R so both read the same fits")
            out.append(row)
    return out


def markdown(rows: list[dict]) -> str:
    header = ("| 模型 | 组 | 发散 | 撞树 | E-BFMI | 参数 maxRhat | 参数 minESS | "
              "生成量 maxRhat | 生成量 minESS | 采样判定 | Δelpd (vs 本组最佳) | se | "
              "elpd_loo | Pareto max k | k>0.7 | k=Inf | LOO 可用性 |")
    rule = "|" + "---|" * 17
    lines = [header, rule]
    for r in rows:
        max_k = "**Inf**" if r["max_k"] == float("inf") else f"{r['max_k']:.3f}"
        lines.append(
            f"| {r['model']} | {r['group']} | {r['divergent']} | {r['treedepth']} | "
            f"{r['ebfmi']:.3f} | {r['par_max_rhat']:.4f} | {r['par_min_ess']:.1f} | "
            f"{r['gq_max_rhat']:.4f} | {r['gq_min_ess']:.1f} | {r['verdict']} | "
            f"{r['elpd_diff']:.1f} | {r['se_diff']:.1f} | {r['elpd_loo']:.1f} | "
            f"{max_k} | {r['n_k_gt_07']} | {r['n_k_inf']} | {r['loo_verdict']} |")
    return "\n".join(lines)


def ic_markdown(rows: list[dict]) -> str:
    """The focused LOOIC/WAIC view, one block per group with the best model on top.

    LOOIC = -2 * elpd_loo, so it ranks identically to the elpd_diff column of the main
    table -- it is here because it is the conventional absolute scale (lower is better),
    which the relative elpd_diff is not.  WAIC is NOT redundant: it uses no PSIS, so it
    can speak to the M1 fits whose LOO is broken by Pareto k = Inf.
    p_waic/n_obs above ~0.4 would mean the WAIC approximation itself is suspect.
    """
    lines = ["| 组 | 模型 | LOOIC | ΔLOOIC | WAIC | ΔWAIC | p_waic | p_waic/n | 可用性 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for group in GROUPS:
        block = [r for r in rows if r["group"] == group]
        best_looic = min(r["looic"] for r in block)
        best_waic = min(r["waic"] for r in block)
        for r in sorted(block, key=lambda x: x["looic"]):
            p_over_n = r["p_waic_over_n"]
            if p_over_n != p_over_n:                       # NaN: older ic_comparison.csv
                p_over_n = r["p_waic"] / 3600.0
            lines.append(
                f"| {group} | {r['model']} | {r['looic']:.1f} | "
                f"{r['looic'] - best_looic:+.1f} | {r['waic']:.1f} | "
                f"{r['waic'] - best_waic:+.1f} | {r['p_waic']:.1f} | {p_over_n:.3f} | "
                f"{r['fit_reliability'] or '—'} |")
    return "\n".join(lines)


def pairwise_markdown() -> str:
    """Model-vs-model differences with the PAIRED standard error.

    `ic_comparison.csv` also carries each model's own se, but those are the wrong ones for
    a comparison: two models scored on the same 3600 observations share their observation
    level noise, so the se of the DIFFERENCE is 2-4x smaller.  Deliberately not shown
    side by side, because a reader would divide by the wrong one.
    """
    path = METRICS_DIR / "ic_pairwise.csv"
    if not path.exists():
        return "(先跑 scripts/compare_ic.R)"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_pair: dict[tuple[str, str, str], dict[str, dict]] = {}
    for r in rows:
        by_pair.setdefault((r["group"], r["model_a"], r["model_b"]), {})[r["ic"]] = r

    def cell(record: dict | None) -> str:
        if record is None:
            return "— | — | —"
        return (f"{float(record['delta']):+.1f} | {float(record['se_paired']):.1f} | "
                f"{float(record['ratio']):.1f}")

    lines = ["| 组 | 比较 | Δ LOOIC | 配对 se | \\|Δ\\|/se | Δ WAIC | 配对 se | \\|Δ\\|/se | 判定 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(by_pair):
        group, a, b = key
        pair = by_pair[key]
        # Per pair, not per row: compare_ic.R grades the pair once and writes the same
        # verdict onto both of its rows, so either row carries it.
        verdict = next((r.get("pair_reliability", "") for r in pair.values()), "")
        lines.append(f"| {group} | {a} vs {b} | {cell(pair.get('LOOIC'))} | "
                     f"{cell(pair.get('WAIC'))} | {verdict or '—'} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="print only the table")
    args = parser.parse_args()
    rows = build()

    out = METRICS_DIR / "fit_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_pass = sum(1 for r in rows if r["par_max_rhat"] <= RHAT_BAR)
    if not args.markdown:
        print(f"{len(rows)} 拟合;参数层 Rhat <= {RHAT_BAR} 的有 {n_pass}/{len(rows)};"
              f"其中 clean {sum(1 for r in rows if r['verdict'] == 'clean')} 个\n")
    print(markdown(rows))
    if not args.markdown:
        print(f"\n wrote {out}")
        print(" 注:Δelpd 是相对于**本组**最佳模型的差值,跨行不可比;"
              "Pareto k>0.7 说明该观测的 PSIS 近似不可靠,k=Inf 表示完全失效。")

    if any(r["looic"] == r["looic"] for r in rows):     # any non-NaN
        print("\n### LOOIC / WAIC\n")
        print(ic_markdown(rows))
        agree = all(
            [r["model"] for r in sorted([x for x in rows if x["group"] == g],
                                        key=lambda x: x["looic"])]
            == [r["model"] for r in sorted([x for x in rows if x["group"] == g],
                                           key=lambda x: x["waic"])]
            for g in GROUPS)
        print(f"\n WAIC 与 LOOIC 的组内排序{'完全一致' if agree else '**不一致**'}"
              "(n_obs = 3600/组,所以 p_waic/n 都很小,两个准则本就该接近)")
        print("\n### 两两比较(配对标准误)\n")
        print(pairwise_markdown())
        print("\n 配对 se 才是比较两个模型该用的量;每个模型各自的 se 大 2–4 倍,"
              "拿它去除 Δ 会低估显著性。")
        print(" WAIC 不用 PSIS,所以它对 M1 有效的那些组(k=Inf 时 LOO 失效)是独立的佐证。")
    elif not args.markdown:
        print("\n (LOOIC/WAIC 缺失 —— 先跑 scripts/compare_ic.R)")


if __name__ == "__main__":
    main()
