# compare_loo.R — PSIS-LOO model comparison across fitted M-models
#
# Ported from formal_batch/analysis/fitting_NN/scripts/compare_loo.R.  Reads every
# fit.rds under fits/ for the requested groups/mode, computes leave-one-out
# cross-validation (M1/M2/M3 all emit the same log_lik [nTrials, nSubjects] array, so the
# comparison is valid), and writes the loo_compare table.
#
# Two changes:
#   * the R-library search and paths point at this codebase;
#   * it now prints a Pareto-k summary next to each comparison.  The old pipeline
#     reported k>0.7 warnings only in prose in its summary doc, which made the caveat easy
#     to drop; here it is computed and written to the CSV's companion column set.
#
# Default groups is all four, since this batch fits all four.
#
# Usage:
#   Rscript compare_loo.R [--mode final] [--groups mac1A,mac1B,mac2A,mac2B] [--models M1,M2,M3]

script_args <- commandArgs(trailingOnly = FALSE)
try(Sys.setlocale("LC_CTYPE", "English_United States.utf8"), silent = TRUE)
file_arg <- script_args[grep("--file=", script_args)]
script_path <- if (length(file_arg)) sub("--file=", "", file_arg[1]) else "."
script_dir  <- dirname(normalizePath(script_path, mustWork = FALSE))
fitting_dir <- dirname(script_dir)

rlib <- Sys.getenv("M31_R_LIB", "")
if (!(nzchar(rlib) && dir.exists(rlib))) {
  rlib <- NULL
  for (root in c(dirname(dirname(dirname(fitting_dir))), dirname(dirname(dirname(dirname(fitting_dir)))))) {
    for (d in list.dirs(root, recursive = FALSE, full.names = TRUE)) {
      p <- file.path(d, "analysis", "model_fitting", "Rlib")
      if (dir.exists(p)) { rlib <- p; break }
    }
    if (!is.null(rlib)) break
  }
  if (is.null(rlib)) stop("Rlib not found near ", fitting_dir, " — set M31_R_LIB")
}
.libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages({ library(cmdstanr); library(loo) })

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop("Missing value after ", flag)
  args[[i + 1L]]
}
mode   <- get_arg("--mode", "final")
groups <- strsplit(get_arg("--groups", "mac1A,mac1B,mac2A,mac2B"), ",")[[1]]
models <- trimws(strsplit(get_arg("--models", "M1,M2,M3"), ",")[[1]])

fit_root <- file.path(fitting_dir, "fits")

for (grp in groups) {
  suffix <- paste0("_condA_", grp, "_", mode)
  fit_dirs <- list.dirs(fit_root, recursive = FALSE, full.names = TRUE)
  fit_dirs <- fit_dirs[endsWith(basename(fit_dirs), suffix)]
  fit_dirs <- fit_dirs[sub(suffix, "", basename(fit_dirs)) %in% models]

  fits <- list()
  for (fd in fit_dirs) {
    mid <- sub(suffix, "", basename(fd))
    fits[[mid]] <- readRDS(file.path(fd, "fit.rds"))
    cat("loaded", mid, grp, "(", length(fits), ")\n")
  }
  if (length(fits) == 0) { cat("no fits for", grp, "— run fit_model.R first\n"); next }

  loos <- lapply(fits, function(f) f$loo())
  cmp  <- loo::loo_compare(loos)

  cat("\n===== LOO comparison:", grp, "=====\n")
  print(cmp, simplify = FALSE)

  # Pareto-k diagnostics, computed rather than left to prose: a k > 0.7 means the PSIS
  # approximation to that observation's LOO predictive is unreliable, and the old
  # pipeline's own summary warns that some cells hit Inf.  The ranking direction is more
  # trustworthy than the elpd differences whenever this fires.
  pareto <- do.call(rbind, lapply(names(loos), function(mid) {
    k <- loos[[mid]]$diagnostics$pareto_k
    data.frame(model = mid,
               n_obs = length(k),
               n_k_gt_07 = sum(k > 0.7, na.rm = TRUE),
               n_k_gt_10 = sum(k > 1.0, na.rm = TRUE),
               n_k_inf  = sum(is.infinite(k)),
               max_k = max(k, na.rm = TRUE))
  }))
  cat("--- Pareto-k diagnostics ---\n")
  print(pareto, row.names = FALSE)

  cmp_df <- as.data.frame.matrix(cmp)
  if (!"model" %in% names(cmp_df)) {
    cmp_df <- data.frame(model = rownames(cmp), cmp_df, row.names = NULL, check.names = FALSE)
  }
  # Attach the Pareto-k counts so the caveat travels with the numbers.
  cmp_df <- merge(cmp_df, pareto, by = "model", all.x = TRUE, sort = FALSE)

  out <- file.path(fit_root, paste0("loo_compare_condA_", grp, "_", mode, ".csv"))
  write.csv(cmp_df, out, row.names = FALSE)
  cat("saved:", out, "\n")
}
