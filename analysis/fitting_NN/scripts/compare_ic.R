# compare_ic.R — LOOIC and WAIC for every fitted model, in one table.
#
# Why this exists on top of compare_loo.R:
#
#   * LOOIC is just -2 * elpd_loo, so it ranks identically to the elpd_diff column the
#     existing script reports.  It is included here because it is the conventional
#     absolute scale (lower is better, comparable across groups), which the relative
#     elpd_diff is not.
#   * WAIC is NOT redundant here.  PSIS-LOO fails outright on some observations in the M1
#     fits (Pareto k = Inf, see README 3.2), and WAIC uses no PSIS at all -- it is a plain
#     function of the pointwise log-likelihood.  So WAIC can, in principle, say something
#     about M1 where LOO could not.  Whether it actually rescues the comparison is the
#     question this script answers.
#
# The comparison stays honest by reporting both side by side, plus p_waic: when
# p_waic/n_obs climbs past ~0.4 the WAIC approximation itself is suspect and the numbers
# should not be leaned on.
#
# A table of LOOIC and WAIC alone is not enough to judge a comparison, because the two
# tables this pipeline writes do not overlap: this one carries the criteria but no
# Pareto-k, and compare_loo.R's carries Pareto-k but no WAIC.  Knowing whether a given
# delta may be quoted therefore meant hand-joining the two.  Everything needed to decide
# is now written into the two CSVs instead:
#
#   ic_comparison.csv  + n_k_gt_07, n_k_gt_10, n_k_inf, max_k, p_waic_over_n,
#                        rank_looic, rank_waic, criterion_agree, fit_reliability
#   ic_pairwise.csv    + rank_agree, pair_reliability, note
#
# `fit_reliability` uses the SAME four levels as summarise_fits.py's loo_verdict(); the
# two must agree, so if you change one, change the other.  `pair_reliability` adds the
# two things a single fit's verdict cannot see: whether the two criteria agree about this
# pair, and whether the difference clears its own paired standard error.
#
# Existing columns keep their names and order -- summarise_fits.py and README 3.2 read
# them by name, and new columns are appended.
#
# Every model emits `log_lik` as an `array[nTrials, nSubjects]` generated quantity, so the
# input is available without refitting.  Reads fit.rds one at a time and frees it before
# the next (the M3 objects are ~1 GB resident and the box has limited free RAM).
#
# Usage:
#   Rscript compare_ic.R [--mode final] [--groups mac1A,mac1B,mac2A,mac2B] [--models M1,M2,M3]

script_args <- commandArgs(trailingOnly = FALSE)
try(Sys.setlocale("LC_CTYPE", "English_United States.utf8"), silent = TRUE)
file_arg <- script_args[grep("--file=", script_args)]
script_path <- if (length(file_arg)) sub("--file=", "", file_arg[1]) else "."
script_dir  <- dirname(normalizePath(script_path, mustWork = FALSE))
fitting_dir <- dirname(script_dir)

rlib <- Sys.getenv("M31_R_LIB", "")
if (!(nzchar(rlib) && dir.exists(rlib))) {
  rlib <- NULL
  for (root in c(dirname(dirname(dirname(fitting_dir))),
                 dirname(dirname(dirname(dirname(fitting_dir)))))) {
    for (d in list.dirs(root, recursive = FALSE, full.names = TRUE)) {
      p <- file.path(d, "analysis", "model_fitting", "Rlib")
      if (dir.exists(p)) { rlib <- p; break }
    }
    if (!is.null(rlib)) break
  }
  if (is.null(rlib)) stop("Rlib not found near ", fitting_dir, " — set M31_R_LIB")
}
.libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages({ library(cmdstanr); library(posterior); library(loo) })

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop("Missing value after ", flag)
  args[[i + 1L]]
}
mode   <- get_arg("--mode", "final")
groups <- trimws(strsplit(get_arg("--groups", "mac1A,mac1B,mac2A,mac2B"), ",")[[1]])
models <- trimws(strsplit(get_arg("--models", "M1,M2,M3"), ",")[[1]])

fit_root <- file.path(fitting_dir, "fits")

# --- reliability helpers -----------------------------------------------------------
# Thresholds: k > 0.7 is the PSIS bar (Vehtari, Gelman & Gabry 2017, Stat Comput 27:1413;
# the sample-size-specific refinement in the PSIS paper caps at 0.7 for any realistic S),
# k >= 1 means the LOO predictive for that observation is not estimable at all, and
# p_waic/n > 0.4 is where the WAIC approximation itself breaks down.
PARETO_BAR <- 0.7

pareto_counts <- function(k) {
  list(n_k_gt_07 = sum(k > PARETO_BAR, na.rm = TRUE),
       n_k_gt_10 = sum(k > 1.0, na.rm = TRUE),
       n_k_inf   = sum(is.infinite(k)),
       max_k     = if (length(k)) max(k, na.rm = TRUE) else NA_real_)
}

# Four levels, MIRRORING summarise_fits.py:loo_verdict() -- a fit can sample perfectly and
# still have unusable LOO (M1 mac1B is exactly that: every Rhat/ESS healthy, 11
# observations at k = Inf), so this is deliberately not folded into the convergence verdict.
fit_reliability <- function(p) {
  if (p$n_k_inf > 0)   return(sprintf("失效(%d×k=Inf)", p$n_k_inf))
  if (p$max_k > 1.0)   return("不可靠(k>1)")
  if (p$max_k > PARETO_BAR) return(sprintf("偏弱(%d×k>0.7)", p$n_k_gt_07))
  "可靠"
}

# Six levels.  The first three nest fit_reliability()'s levels and are inherited from the
# fits themselves -- a pair can be no more trustworthy than its worse side, so the k > 1
# rung has to be its own level.  Without it a pair involving an 不可靠(k>1) fit came out
# 偏弱, i.e. the comparison looked BETTER than the fit it rests on.  The last three are
# about this pair specifically and cannot be read off either fit alone.
pair_reliability <- function(pa, pb, d_loo, d_waic, ratio) {
  if (pa$n_k_inf > 0 || pb$n_k_inf > 0) return("不可用(一侧 LOO 失效)")
  if (pa$max_k > 1.0 || pb$max_k > 1.0) return("不可靠(一侧 k>1)")
  if (pa$max_k > PARETO_BAR || pb$max_k > PARETO_BAR) return("偏弱(Pareto-k 警告)")
  if (sign(d_loo) != sign(d_waic))  return("两准则不一致")
  if (ratio < 2)                    return("不确定(|Δ|/se < 2)")
  "可靠"
}

# mac2B uses a different episode draw AND a non-default sampler from the other three
# groups (see README 3.10); the caveat has to travel with the numbers, not sit in prose
# three sections away.  Keep this string in sync with metrics/fit_provenance.json.
GROUP_NOTE <- c(mac2B = "mac2B 用 draw 20261003 + adapt_delta 0.99（从多次抽样中选收敛者），与其余三组（20260930 / 0.95）不同（README §3.10）")
note_for <- function(grp) {
  n <- GROUP_NOTE[grp]
  if (is.na(n)) "" else unname(n)
}
rows <- list()
# Pointwise contributions, kept so the model-vs-model comparison can use the PAIRED
# standard error.  Comparing two models on the same observations, the se of the difference
# is far smaller than the se of either LOOIC on its own -- quoting the individual se would
# understate every difference by roughly a factor of two to four.  Each vector is 3600
# doubles, so keeping all twelve costs nothing.
point <- list()
# Pareto-k counts per fit, carried into the pairwise stage so a comparison can be graded
# without re-reading the fits.
pareto_by_fit <- list()

for (grp in groups) {
  for (mid in models) {
    fit_dir <- file.path(fit_root, paste0(mid, "_condA_", grp, "_", mode))
    fit_file <- file.path(fit_dir, "fit.rds")
    if (!file.exists(fit_file)) { cat("missing:", fit_file, "\n"); next }

    cat("reading", mid, grp, "...\n")
    fit <- readRDS(fit_file)
    ll  <- as.matrix(posterior::as_draws_matrix(fit$draws("log_lik")))
    # LOO via the fit's own method, NOT loo::loo(ll).  CmdStanFit$loo() derives the
    # relative efficiency from the draws; loo.matrix() defaults to r_eff = 1.  The two do
    # NOT agree -- M1 mac1A max k is 1.437 one way and 1.696 the other, and elpd_loo moves
    # by ~3e-5 (M3: ~1.4e-3).  compare_loo.R uses the fit method, and its Pareto-k columns
    # are the ones fit_summary.csv already reports; taking this side from the same place is
    # what makes the two tables consistent rather than merely close.  (Before this, the
    # loo_compare-derived elpd_loo and the ic_comparison-derived looic in fit_summary.csv
    # were not exact negatives of each other.)
    l <- suppressWarnings(fit$loo())
    rm(fit); invisible(gc(verbose = FALSE))

    n_obs <- ncol(ll)
    w <- suppressWarnings(loo::waic(ll))
    rm(ll); invisible(gc(verbose = FALSE))

    we <- w$estimates
    le <- l$estimates
    # Pareto-k comes from the same loo object the LOOIC is read off, so the criterion and
    # its own reliability check can never refer to different fits.  Indices line up with
    # the pointwise vectors below (both are the flattened [trial, subject] order).
    pc <- pareto_counts(l$diagnostics$pareto_k)
    key <- paste(mid, grp, sep = "|")
    point[[key]] <- list(
      looic = -2 * l$pointwise[, "elpd_loo"],
      waic  = -2 * w$pointwise[, "elpd_waic"])
    pareto_by_fit[[key]] <- pc
    rows[[length(rows) + 1L]] <- data.frame(
      model = mid, group = grp, n_obs = n_obs,
      elpd_waic = we["elpd_waic", "Estimate"], se_elpd_waic = we["elpd_waic", "SE"],
      p_waic    = we["p_waic", "Estimate"],    se_p_waic    = we["p_waic", "SE"],
      waic      = we["waic", "Estimate"],      se_waic      = we["waic", "SE"],
      elpd_loo  = le["elpd_loo", "Estimate"],  se_elpd_loo  = le["elpd_loo", "SE"],
      p_loo     = le["p_loo", "Estimate"],     se_p_loo     = le["p_loo", "SE"],
      looic     = le["looic", "Estimate"],     se_looic     = le["looic", "SE"],
      n_k_gt_07 = pc$n_k_gt_07, n_k_gt_10 = pc$n_k_gt_10, n_k_inf = pc$n_k_inf,
      max_k     = pc$max_k,
      p_waic_over_n = we["p_waic", "Estimate"] / n_obs,
      stringsAsFactors = FALSE
    )
    cat(sprintf("   WAIC %.1f (p_waic %.1f, p/n %.3f)   LOOIC %.1f (p_loo %.1f)\n",
                we["waic", "Estimate"], we["p_waic", "Estimate"], we["p_waic", "Estimate"] / n_obs,
                le["looic", "Estimate"], le["p_loo", "Estimate"]))
  }
}

if (length(rows) == 0) stop("no fits found — run fit_model.R first")
out <- do.call(rbind, rows)
out <- out[order(out$group, out$model), ]

# Ranks are computed WITHIN each group (1 = best = lowest IC).  The criteria are absolute
# scales, so within-group rank is the only part comparable across groups.
out$rank_looic <- NA_integer_
out$rank_waic  <- NA_integer_
for (grp in unique(out$group)) {
  idx <- which(out$group == grp)
  out$rank_looic[idx] <- rank(out$looic[idx])
  out$rank_waic[idx]  <- rank(out$waic[idx])
}
# Per-fit: do the two criteria put THIS model in the same place?  The group-level answer is
# all() over its rows, printed below and no longer merely printed.
out$criterion_agree <- out$rank_looic == out$rank_waic
out$fit_reliability <- vapply(seq_len(nrow(out)), function(i)
  fit_reliability(pareto_by_fit[[paste(out$model[i], out$group[i], sep = "|")]]),
  character(1))

# --- cross-check against compare_loo.R --------------------------------------------
# compare_loo.R writes fits/loo_compare_condA_<group>_<mode>.csv from the same fits via the
# same fit$loo() path.  Asserting agreement turns a silent divergence into a loud failure;
# without it the pipeline can emit two incompatible tables and nothing notices (which is
# what fit_summary.csv did before this change -- its elpd_loo came from loo_compare and its
# looic from ic_comparison, and the two were not exact negatives of one another).
for (grp in unique(out$group)) {
  ref_path <- file.path(fit_root, paste0("loo_compare_condA_", grp, "_", mode, ".csv"))
  if (!file.exists(ref_path)) {
    cat("  (no ", basename(ref_path), " -- skipping cross-check)\n", sep = "")
    next
  }
  ref <- read.csv(ref_path, stringsAsFactors = FALSE)
  for (i in which(out$group == grp)) {
    r <- match(out$model[i], ref$model)
    if (is.na(r)) next
    for (col in c("elpd_loo", "looic")) {
      a <- out[[col]][i]; b <- ref[[col]][r]
      rel <- abs(a - b) / max(1, abs(b))
      if (!is.na(rel) && rel > 1e-6)
        stop(sprintf("cross-check failed: %s %s %s = %.10g, compare_loo.R has %.10g",
                     out$model[i], grp, col, a, b))
    }
  }
}

res_path <- file.path(fitting_dir, "metrics",
                      paste0("ic_comparison", if (mode == "final") "" else paste0("_", mode),
                             ".csv"))
write.csv(out, res_path, row.names = FALSE)

cat("\n===== WAIC / LOOIC =====\n")
print(out[, c("model", "group", "elpd_waic", "p_waic", "waic", "se_waic",
              "elpd_loo", "p_loo", "looic", "se_looic",
              "max_k", "n_k_gt_07", "n_k_inf", "fit_reliability")], row.names = FALSE)

# The two should agree in ranking; a disagreement is the interesting outcome, not noise.
cat("\nrank check (WAIC vs LOOIC, 1 = best within each group):\n")
for (grp in unique(out$group)) {
  sub <- out[out$group == grp, ]
  cat(sprintf("  %-6s WAIC %s   LOOIC %s   %s\n", grp,
              paste(sub$model[order(sub$rank_waic)], collapse = " < "),
              paste(sub$model[order(sub$rank_looic)], collapse = " < "),
              ifelse(all(sub$criterion_agree), "same", "DIFFERENT")))
}
cat(sprintf("\n  两准则组内排序:%d/%d 个拟合一致\n", sum(out$criterion_agree), nrow(out)))
cat("saved:", res_path, "\n")

# --- pairwise differences, with the PAIRED standard error --------------------------
# The se of a difference between two models scored on the SAME observations is much
# smaller than the se of either total, because the observation-level errors cancel.
# Quoting the individual se would understate every difference here by roughly 2-4x.
cat("\npairwise (delta < 0 means the first model fits better; se is paired):\n")
cmp <- list()
for (grp in groups) {
  ms <- models[vapply(models, function(m) !is.null(point[[paste(m, grp, sep = "|")]]),
                      logical(1))]
  if (length(ms) < 2) next
  for (i in seq_len(length(ms) - 1)) for (j in (i + 1L):length(ms)) {
    a <- ms[i]; b <- ms[j]
    ka <- pareto_by_fit[[paste(a, grp, sep = "|")]]
    kb <- pareto_by_fit[[paste(b, grp, sep = "|")]]
    pa <- point[[paste(a, grp, sep = "|")]]
    pb <- point[[paste(b, grp, sep = "|")]]
    delta <- list(); se <- list()
    for (ic in c("looic", "waic")) {
      d <- pa[[ic]] - pb[[ic]]
      delta[[ic]] <- sum(d)
      se[[ic]]    <- sqrt(length(d)) * sd(d)
    }
    ratio <- vapply(c("looic", "waic"), function(ic)
      if (se[[ic]] > 0) abs(delta[[ic]]) / se[[ic]] else Inf, numeric(1))
    # One verdict and one agreement flag per PAIR, not per row: whether the two criteria
    # disagree is a property of the pair, as is the reliability of the two fits.  The
    # decisiveness test takes the MINIMUM of the two ratios, so a pair is only called
    # reliable when both criteria clear |d|/se >= 2, not just the friendlier one.
    verdict <- pair_reliability(ka, kb, delta$looic, delta$waic, min(ratio))
    agree   <- sign(delta$looic) == sign(delta$waic)
    for (ic in c("looic", "waic")) {
      cat(sprintf("  %-6s %s vs %s  d%-5s %+9.1f   se %7.1f   |d|/se %5.1f\n",
                  grp, a, b, toupper(ic), delta[[ic]], se[[ic]], ratio[[ic]]))
      cmp[[length(cmp) + 1L]] <- data.frame(
        group = grp, model_a = a, model_b = b, ic = toupper(ic),
        delta = delta[[ic]], se_paired = se[[ic]], ratio = ratio[[ic]],
        criterion_agree = agree, pair_reliability = verdict, note = note_for(grp),
        stringsAsFactors = FALSE)
    }
    cat(sprintf("         -> %s\n", verdict))
  }
}
pair_path <- file.path(fitting_dir, "metrics",
                       paste0("ic_pairwise", if (mode == "final") "" else paste0("_", mode),
                              ".csv"))
write.csv(do.call(rbind, cmp), pair_path, row.names = FALSE)
cat("\nsaved:", pair_path, "\n")
