"""audit_diagnostics.py — explain every NA in the fits' parameter summaries, and report
strict convergence for all 12 fits (params and generated quantities SEPARATELY).

Why this exists
---------------
`M3_condA_mac2B_final/parameter_summary.csv` has 8182 rows containing `NA` out of 25579,
which reads alarming.  It is not a data problem and not a fitting failure: every one of
those NA sits in a *diagnostic* column (`rhat` / `ess_bulk` / `ess_tail`), and the six
columns that hold actual posterior values (`mean`, `median`, `sd`, `mad`, `q5`, `q95`)
have ZERO missing entries in any of the 12 fits.  Nothing is NaN.

The rule behind the diagnostics (posterior 1.7.0, established by direct probe, see
`README.md` §3.5):

  * `rhat` / `ess_bulk` = NA  <=>  the quantity is IDENTICAL in every draw (zero variance).
  * `ess_tail` = min(ess_quantile(.05), ess_quantile(.95)) = NA  <=>  at least one of the
    indicator vectors `1(x <= q_p)` is CONSTANT, which happens when that quantile lands on
    an extreme of the sample (e.g. `tau` is declared `<upper=3>` and its posterior piles up
    at 3, so q95 == 3 == max and `1(tau <= 3)` is trivially all-true).

So the NA split into three benign causes: a quantity that is a deterministic function of
the data (`L_time`, `p_time` at t=1), a gate that saturates to a float boundary
(`w_time` pinned at exactly 1.0), and a binary / tail-pinned quantity (`y_pred`, and the
hard-bounded `tau` / `lr_pos`).

This script re-derives all of that from the CSVs so the claim is computed rather than
left to prose -- the same convention `compare_loo.R` follows for its Pareto-k column.

What it prints
--------------
1. Strict convergence table: max Rhat and min ESS_bulk over the FITTED PARAMETERS and over
   the GENERATED QUANTITIES separately (mixing the two hides real problems -- generated
   quantities are deterministic functions of the parameters and are numerically harder).
2. NA classification table: every (variable family, missing columns) group with its cause.
3. `metrics/diagnostic_na_audit.csv` with the same content.

It then ASSERTS the invariant that matters: no `rhat` or `ess_bulk` NA ever falls on a
fitted parameter, in any fit.  If that assertion ever fires, the summary is hiding a real
non-convergence and the README's claim is wrong.

Read-only: it never re-runs a fit.

Run:
    cd formal_run_001/analysis/fitting_NN
    ../../.venv/Scripts/python.exe scripts/audit_diagnostics.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FITTING_DIR = SCRIPT_DIR.parent
FITS_DIR = FITTING_DIR / "fits"
METRICS_DIR = FITTING_DIR / "metrics"

DIAG_COLS = ("rhat", "ess_bulk", "ess_tail")
VALUE_COLS = ("mean", "median", "sd", "mad", "q5", "q95")

# Variable -> which Stan block it is declared in.  Written out rather than inferred: the
# summary CSV only carries names, and the whole point of the params-vs-GQ split is that
# these two groups get judged by different standards.  Mirrors the block structure of
# stan/M1.1.stan, M2.1.stan, M3.1.stan.
LAYERS: dict[str, dict[str, tuple[str, ...]]] = {
    "M1": {
        "parameters": ("lp__", "lr_mu_raw", "tau_mu_raw", "lr_sd_raw", "tau_sd_raw",
                       "lr_raw", "tau_raw"),
        "transformed parameters": ("lr", "tau"),
        "generated quantities": ("lr_mu", "tau_mu", "log_lik"),
    },
    "M2": {
        "parameters": ("lp__", "lr_pos_mu_raw", "lr_neg_mu_raw", "tau_mu_raw",
                       "lr_pos_sd_raw", "lr_neg_sd_raw", "tau_sd_raw",
                       "lr_pos_raw", "lr_neg_raw", "tau_raw"),
        "transformed parameters": ("lr_pos", "lr_neg", "tau"),
        "generated quantities": ("lr_pos_mu", "lr_neg_mu", "tau_mu", "log_lik"),
    },
    "M3": {
        "parameters": ("lp__", "lr_pos_mu_raw1", "lr_pos_mu_raw2", "lr_neg_mu_raw1",
                       "lr_neg_mu_raw2", "tau_mu_raw", "theta_mu", "lr_pos_sd_raw1",
                       "lr_pos_sd_raw2", "lr_neg_sd_raw1", "lr_neg_sd_raw2", "tau_sd_raw",
                       "theta_sd", "lr_pos_raw1", "lr_pos_raw2", "lr_neg_raw1",
                       "lr_neg_raw2", "tau_raw", "theta_raw"),
        "transformed parameters": ("lr_pos1", "lr_pos2", "lr_neg1", "lr_neg2", "tau", "theta"),
        "generated quantities": ("lr_pos_mu1", "lr_pos_mu2", "lr_neg_mu1", "lr_neg_mu2",
                                 "tau_mu", "theta_mu_out", "log_lik", "p_time", "L_time",
                                 "w_time", "lr_pos_time", "lr_neg_time", "y_pred"),
    },
}

# Declared hard upper bounds, used only to name the cause of a tail-pinned `ess_tail` NA.
UPPER_BOUND = {
    "lr": 1.0, "lr_pos": 1.0, "lr_neg": 1.0,
    "lr_pos1": 1.0, "lr_pos2": 1.0, "lr_neg1": 1.0, "lr_neg2": 1.0,
    "lr_pos_mu": 1.0, "lr_neg_mu": 1.0, "lr_pos_mu1": 1.0, "lr_pos_mu2": 1.0,
    "lr_neg_mu1": 1.0, "lr_neg_mu2": 1.0,
    "tau": 3.0, "tau_mu": 3.0,
    "p_time": 1.0, "w_time": 1.0,
}

# `lp__` is the unnormalised log posterior density, not a model quantity.  It is reported
# separately so it cannot drive -- or hide -- the fitted-parameter verdict.
NOT_A_QUANTITY = {"lp__"}


def family(variable: str) -> str:
    """'lr_pos_time[12,3]' -> 'lr_pos_time'."""
    return re.sub(r"\[.*", "", variable)


def layer_of(model: str, fam: str) -> str | None:
    for layer, names in LAYERS[model].items():
        if fam in names:
            return layer
    return None


def read_summary(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def extreme(rows: list[dict], column: str, *, want_max: bool = True) -> tuple[float, str] | None:
    """Largest/smallest finite value of `column` over rows, with the variable that holds it."""
    best: tuple[float, str] | None = None
    for row in rows:
        value = as_float(row[column])
        if value is None:
            continue
        if best is None or (value > best[0] if want_max else value < best[0]):
            best = (value, row["variable"])
    return best


def convergence_row(model: str, group: str, rows: list[dict], diag_path: Path) -> dict:
    """Max Rhat / min ESS_bulk for the fitted parameters and for the GQs, separately."""
    buckets: dict[str, list[dict]] = {"fitted parameters": [], "generated quantities": [],
                                      "lp__": []}
    for row in rows:
        fam = family(row["variable"])
        if fam in NOT_A_QUANTITY:
            buckets["lp__"].append(row)
            continue
        layer = layer_of(model, fam)
        if layer is None:
            # A name the registry does not know. Loud, not silent: an unclassified quantity
            # would otherwise be excluded from the verdict without anyone noticing.
            raise SystemExit(f"{model}: variable {row['variable']!r} matches no Stan block")
        target = ("fitted parameters" if layer in ("parameters", "transformed parameters")
                  else "generated quantities")
        buckets[target].append(row)

    out = {"model": model, "group": group}
    for key, label in (("fitted parameters", "par"), ("generated quantities", "gq")):
        worst_rhat = extreme(buckets[key], "rhat", want_max=True)
        worst_ess = extreme(buckets[key], "ess_bulk", want_max=False)
        out[f"{label}_max_rhat"] = worst_rhat[0] if worst_rhat else None
        out[f"{label}_max_rhat_var"] = worst_rhat[1] if worst_rhat else ""
        out[f"{label}_min_ess"] = worst_ess[0] if worst_ess else None
        out[f"{label}_min_ess_var"] = worst_ess[1] if worst_ess else ""

    if diag_path.exists():
        with diag_path.open(newline="", encoding="utf-8") as handle:
            chains = list(csv.DictReader(handle))
        out["divergent"] = sum(int(c["num_divergent"]) for c in chains)
        out["treedepth"] = sum(int(c["num_max_treedepth"]) for c in chains)
        out["ebfmi_min"] = min(float(c["ebfmi"]) for c in chains)
    else:
        out["divergent"] = out["treedepth"] = out["ebfmi_min"] = None

    # The invariant this whole script exists to defend.
    offenders = [
        f"{r['variable']}/{col}"
        for r in buckets["fitted parameters"]
        for col in ("rhat", "ess_bulk")
        if r[col] in ("NA", "")
    ]
    if offenders:
        raise SystemExit(
            f"{model}/{group}: diagnostic NA landed on a FITTED PARAMETER: {offenders[:5]} "
            "-- this is a real non-convergence, not a display artifact"
        )
    return out


def classify(row: dict) -> str:
    """Name the cause of one NA row.  Returns '' when the row has no NA."""
    missing = [c for c in DIAG_COLS if row[c] in ("NA", "")]
    if not missing:
        return ""
    sd = as_float(row["sd"])
    if set(missing) == set(DIAG_COLS):
        # Every diagnostic is undefined, which is exactly what a zero-variance sample gives.
        return f"constant across all draws (sd={sd:g})" if sd == 0 else \
               f"all three diagnostics NA (sd={sd:g})"
    if missing == ["ess_tail"]:
        q95 = as_float(row["q95"])
        q5 = as_float(row["q5"])
        fam = family(row["variable"])
        bound = UPPER_BOUND.get(fam)
        if bound is not None and q95 == bound:
            return f"upper-tail pinned at the declared bound {bound:g} (q95 == max)"
        if q95 is not None and q5 is not None and q95 > q5:
            return "binary / heavily tied (1(x<=q95) is all-true)"
        return "mode pinned at the sample max (1(x<=q95) is all-true)"
    if "rhat" in missing and "ess_bulk" not in missing:
        # `rhat` = max(rank-normalised, FOLDED rhat), and the folded branch is computed on
        # |x - median|.  For a binary quantity whose interpolated median lands exactly on
        # 0.5, that folded sample is constant (|0-0.5| == |1-0.5| == 0.5), so the folded
        # Rhat is undefined and takes the whole Rhat with it -- while ess_bulk, which is
        # rank-based and never folds, is unaffected.  Seen on 3 y_pred cells that came out
        # an exact 4000/4000 split over the 8000 draws.
        median = as_float(row["median"])
        return (f"folded-Rhat degenerate: binary split exactly 50/50, interpolated "
                f"median = {median:g} so |x - median| is constant")
    return "unclassified"


def main() -> None:
    rows_out: list[dict] = []
    audit: list[dict] = []
    n_value_na = 0  # cells of mean/median/sd/mad/q5/q95 that are missing -- must stay 0

    for fit_dir in sorted(FITS_DIR.glob("*_final")):
        name = fit_dir.name
        match = re.match(r"^(M[123])_condA_(\w+)_final$", name)
        if not match:
            print(f"skipping unrecognised fit dir: {name}", file=sys.stderr)
            continue
        model, group = match.group(1), match.group(2)
        summary = read_summary(fit_dir / "parameter_summary.csv")

        rows_out.append(convergence_row(model, group, summary,
                                        fit_dir / "diagnostic_summary.csv"))

        groups: dict[tuple[str, tuple[str, ...]], list[dict]] = {}
        for row in summary:
            missing = tuple(c for c in DIAG_COLS if row[c] in ("NA", ""))
            if missing:
                groups.setdefault((family(row["variable"]), missing), []).append(row)
        n_missing = sum(1 for row in summary if any(row[c] in ("NA", "") for c in DIAG_COLS))
        n_value_na += sum(1 for row in summary for c in VALUE_COLS if row[c] in ("NA", ""))

        for (fam, missing), members in groups.items():
            layer = layer_of(model, fam) or "?"
            causes = {classify(row) for row in members}
            audit.append({
                "model": model,
                "group": group,
                "layer": layer,
                "variable_family": fam,
                "missing_columns": "+".join(missing),
                "n_rows": len(members),
                "n_rows_total": sum(1 for r in summary if family(r["variable"]) == fam),
                "cause": " | ".join(sorted(causes)),
                "fitted_parameter": layer in ("parameters", "transformed parameters"),
            })

        if n_missing:
            print(f"   {name}: {n_missing}/{len(summary)} rows carry a diagnostic NA", flush=True)

    # ---------------------------------------------------------------- strict convergence
    print("\n===== strict convergence (params vs generated quantities, separately) =====")
    header = (f"{'model':<4} {'group':<7} {'div':>4} {'tree':>5} {'ebfmi':>6} "
              f"{'par_maxRhat':>11} {'par_minESS':>10} {'gq_maxRhat':>10} {'gq_minESS':>9}")
    print(header)
    print("-" * len(header))
    for row in rows_out:
        flag = ""
        if row["par_max_rhat"] is not None and row["par_max_rhat"] > 1.05:
            flag = "   <-- PARAMS NOT CONVERGED"
        print(f"{row['model']:<4} {row['group']:<7} {row['divergent']:>4} {row['treedepth']:>5} "
              f"{row['ebfmi_min']:>6.3f} {row['par_max_rhat']:>11.4f} {row['par_min_ess']:>10.1f} "
              f"{row['gq_max_rhat']:>10.4f} {row['gq_min_ess']:>9.1f}{flag}")

    bad = [r for r in rows_out if r["par_max_rhat"] is not None and r["par_max_rhat"] > 1.05]
    print(f"\n {len(rows_out) - len(bad)}/{len(rows_out)} fits pass the fitted-parameter "
          f"Rhat <= 1.05 bar; {len(bad)} fail")

    # --------------------------------------------------------------------- NA accounting
    print("\n===== diagnostic-NA accounting =====")
    header = f"{'model':<4} {'group':<7} {'layer':<23} {'family':<13} {'missing':<27} {'rows':>5}  cause"
    print(header)
    print("-" * 150)
    for row in audit:
        print(f"{row['model']:<4} {row['group']:<7} {row['layer']:<23} "
              f"{row['variable_family']:<13} {row['missing_columns']:<27} {row['n_rows']:>5}  "
              f"{row['cause']}")

    n_param_na = sum(r["n_rows"] for r in audit if r["fitted_parameter"])
    print(f"\n NA rows falling on FITTED PARAMETERS: {n_param_na} "
          f"(of which rhat/ess_bulk: 0 -- asserted)")
    print(f" NA in the value columns (mean/median/sd/mad/q5/q95): {n_value_na} "
          f"(computed, not asserted)")
    if n_value_na:
        raise SystemExit(
            f"{n_value_na} missing posterior values in the value columns -- that WOULD be a "
            "real data problem, unlike the diagnostic-column NA this script explains"
        )

    # A cause the classifier does not recognise means this README section is incomplete.
    # Fail loudly rather than letting a new failure mode hide under "unclassified".
    unexplained = [r for r in audit if "unclassified" in r["cause"]]
    if unexplained:
        raise SystemExit(
            "unclassified diagnostic NA -- the cause table needs updating:\n  "
            + "\n  ".join(f"{r['model']}/{r['group']} {r['variable_family']} "
                          f"({r['missing_columns']}, {r['n_rows']} rows)"
                          for r in unexplained)
        )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "diagnostic_na_audit.csv"
    fields = ["model", "group", "layer", "variable_family", "missing_columns", "n_rows",
              "n_rows_total", "cause", "fitted_parameter"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    print(f"\n wrote {out_path}")


if __name__ == "__main__":
    main()
