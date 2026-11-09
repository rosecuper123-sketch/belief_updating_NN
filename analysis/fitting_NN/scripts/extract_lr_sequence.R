# extract_lr_sequence.R — extract the M3 dynamic-LR sequence + key params from a fit.rds
#
# Ported from formal_batch/analysis/fitting_NN/scripts/extract_lr_sequence.R.  The
# extraction logic and the reshape cross-check are unchanged.  Two edits:
#
#   * the R-library search and the default paths point at this codebase;
#   * the subject-block column offsets in the cross-check are derived from nSubjects
#     instead of being written as 1:30 / 31:60 / 91:120 / 121:150.  Behaviour is
#     identical here (nSubjects is 30 in both batches) but the literal version silently
#     reads the wrong columns for any other subject count.
#
# Produces:
#   1. lr_sequence_<group>.csv  — trial-by-trial lr_pos / lr_neg / gate w / belief p,
#      each collapsed per subject to its posterior mean, then summarised ACROSS subjects.
#   2. key_params_group_<group>.csv   — group-level readable parameters.
#   3. key_params_subject_<group>.csv — per-subject posterior means.
#
# Reporting convention: the subject-level LR distributions are right-skewed / bimodal
# (a few locked seeds), so the MEAN hides the pos/neg ordering reversal.  Median is the
# primary summary, mean and IQR are reported alongside.
#
# Usage:
#   Rscript extract_lr_sequence.R --fit fits/M3_condA_mac1A_final/fit.rds \
#           --data data/formal_osla__h16_b200__mac1A.rds --group mac1A

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
suppressPackageStartupMessages({ library(cmdstanr); library(posterior) })

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop("Missing value after ", flag)
  args[[i + 1L]]
}
fit_path  <- get_arg("--fit")
data_path <- get_arg("--data")
group     <- get_arg("--group", "mac1A")
out_dir   <- file.path(fitting_dir, get_arg("--out", "metrics"))
n_check   <- as.integer(get_arg("--n-check", "5"))

if (is.null(fit_path))  stop("--fit <fit.rds> required")
if (is.null(data_path)) stop("--data <data.rds> required")
if (!file.exists(fit_path))  stop("fit not found: ", fit_path)
if (!file.exists(data_path)) stop("data not found: ", data_path)

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("Loading fit:", fit_path, "\n")
fit <- readRDS(fit_path)

cat("Loading data:", data_path, "\n")
dlist <- readRDS(data_path)
nT <- dlist$nTrials
nS <- dlist$nSubjects
reward <- dlist$reward               # [nT, nS]
p0     <- as.numeric(dlist$mean_p)   # [nS]
k      <- if (!is.null(dlist$k)) dlist$k else 10

# --------------------------------------------------------------------------- #
# 1. Reshape helper: Stan stores array[nTrials, nSubjects] column-major, so the
#    flat element (t, s) sits at (s-1)*nT + t. matrix(x, nrow=nT, ncol=nS) fills
#    column-major, therefore column s == subject s. Accumulate the posterior mean
#    per (trial, subject) with a low-memory loop over draws.
# --------------------------------------------------------------------------- #
draws_mean <- function(fit, var, n1, n2) {
  d <- posterior::as_draws_matrix(fit$draws(var))   # [n_draws, n1*n2]
  acc <- matrix(0.0, n1, n2)
  for (dd in seq_len(nrow(d))) acc <- acc + matrix(d[dd, ], nrow = n1, ncol = n2)
  acc / nrow(d)
}

lr_pos_mean <- draws_mean(fit, "lr_pos_time", nT, nS)
lr_neg_mean <- draws_mean(fit, "lr_neg_time", nT, nS)
w_mean      <- draws_mean(fit, "w_time",      nT, nS)
p_mean      <- draws_mean(fit, "p_time",      nT, nS)

group_stat <- function(m) {
  cbind(mean   = rowMeans(m),
        median = apply(m, 1, median),
        q25    = apply(m, 1, function(x) as.numeric(quantile(x, 0.25))),
        q75    = apply(m, 1, function(x) as.numeric(quantile(x, 0.75))))
}
lr_pos_g <- group_stat(lr_pos_mean)
lr_neg_g <- group_stat(lr_neg_mean)
w_g      <- group_stat(w_mean)
p_g      <- group_stat(p_mean)

# --------------------------------------------------------------------------- #
# 2. Cross-validate the reshape: recompute lr_pos/lr_neg from a few draws' own
#    parameters and compare to the reshaped generated quantities at those draws.
# --------------------------------------------------------------------------- #
recompute_lr <- function(p0, reward, lr_pos1, lr_pos2, lr_neg1, lr_neg2, theta, k) {
  nT <- nrow(reward); nS <- ncol(reward)
  z <- qlogis(p0)
  L <- rep(0.0, nS)
  inc_r1 <- -log(2) - log(p0)
  inc_r0 <- -log(2) - log1p(-p0)
  ap <- matrix(NA_real_, nT, nS)
  an <- matrix(NA_real_, nT, nS)
  for (t in seq_len(nT)) {
    w  <- plogis(k * (L - theta))
    ap_t <- (1 - w) * lr_pos1 + w * lr_pos2
    an_t <- (1 - w) * lr_neg1 + w * lr_neg2
    ap[t, ] <- ap_t
    an[t, ] <- an_t
    p   <- plogis(z)
    pe  <- reward[t, ] - p
    win <- reward[t, ] == 1
    z <- z + ifelse(win, ap_t, an_t) * pe
    L <- L + win * inc_r1 + (!win) * inc_r0
  }
  list(ap = ap, an = an)
}

# Subject-level parameter draws, blocked by subject. Offsets derived from nS rather
# than written out, so a different subject count cannot silently mis-slice.
pd <- posterior::as_draws_matrix(fit$draws(
  c("lr_pos1", "lr_pos2", "lr_neg1", "lr_neg2", "theta")))
col_lr_pos1 <- (0 * nS + 1):(1 * nS)
col_lr_pos2 <- (1 * nS + 1):(2 * nS)
col_lr_neg1 <- (2 * nS + 1):(3 * nS)
col_lr_neg2 <- (3 * nS + 1):(4 * nS)
col_theta   <- (4 * nS + 1):(5 * nS)
if (ncol(pd) != 5 * nS) stop("unexpected subject-block width: ", ncol(pd), " vs ", 5 * nS)

lr_pos_draws <- posterior::as_draws_matrix(fit$draws("lr_pos_time"))
lr_neg_draws <- posterior::as_draws_matrix(fit$draws("lr_neg_time"))
max_err <- 0
for (dd in sample.int(nrow(pd), min(n_check, nrow(pd)))) {
  rr <- recompute_lr(p0, reward,
                     as.numeric(pd[dd, col_lr_pos1]),
                     as.numeric(pd[dd, col_lr_pos2]),
                     as.numeric(pd[dd, col_lr_neg1]),
                     as.numeric(pd[dd, col_lr_neg2]),
                     as.numeric(pd[dd, col_theta]), k)
  got_ap <- matrix(lr_pos_draws[dd, ], nrow = nT, ncol = nS)
  got_an <- matrix(lr_neg_draws[dd, ], nrow = nT, ncol = nS)
  err <- max(abs(rr$ap - got_ap), abs(rr$an - got_an))
  max_err <- max(max_err, err)
  cat(sprintf("  draw %d: max |recompute - generated| = %.3e\n", dd, err))
}
# FP accumulation over the trial recurrence is ~1e-6; an indexing error is O(1), so
# 1e-4 cleanly separates the two.
if (max_err > 1e-4) {
  stop("reshape cross-check FAILED: max err = ", max_err)
} else {
  cat("reshape cross-check OK (max err", sprintf("%.3e", max_err), "< 1e-4)\n")
}

# --------------------------------------------------------------------------- #
# 3. Key parameters. Uses fit$draws() + base-R stats (fit$summary() segfaults on
#    large fits).
# --------------------------------------------------------------------------- #
group_vars <- c("lr_pos_mu1", "lr_pos_mu2", "lr_neg_mu1", "lr_neg_mu2",
                "tau_mu", "theta_mu_out")
group_rows <- do.call(rbind, lapply(group_vars, function(v) {
  d <- as.numeric(posterior::as_draws_matrix(fit$draws(v)))
  data.frame(variable = v, mean = mean(d), sd = sd(d),
             q5 = as.numeric(quantile(d, 0.05)),
             q95 = as.numeric(quantile(d, 0.95)))
}))

subj_vars <- c("lr_pos1", "lr_pos2", "lr_neg1", "lr_neg2", "tau", "theta")
subj_rows <- do.call(rbind, lapply(subj_vars, function(v) {
  d <- posterior::as_draws_matrix(fit$draws(v))
  data.frame(variable = paste0(v, "[", seq_len(ncol(d)), "]"),
             mean = colMeans(d))
}))

# --------------------------------------------------------------------------- #
# 4. Write CSVs.
# --------------------------------------------------------------------------- #
lr_seq_df <- data.frame(
  trial          = seq_len(nT),
  lr_pos_mean    = lr_pos_g[, 1],
  lr_pos_median  = lr_pos_g[, 2],
  lr_pos_q25     = lr_pos_g[, 3],
  lr_pos_q75     = lr_pos_g[, 4],
  lr_neg_mean    = lr_neg_g[, 1],
  lr_neg_median  = lr_neg_g[, 2],
  lr_neg_q25     = lr_neg_g[, 3],
  lr_neg_q75     = lr_neg_g[, 4],
  w_mean         = w_g[, 1],
  w_median       = w_g[, 2],
  p_mean         = p_g[, 1],
  p_median       = p_g[, 2]
)
lr_seq_out <- file.path(out_dir, paste0("lr_sequence_", group, ".csv"))
write.csv(lr_seq_df, lr_seq_out, row.names = FALSE)
cat("wrote", lr_seq_out, "\n")

group_out <- file.path(out_dir, paste0("key_params_group_", group, ".csv"))
write.csv(group_rows, group_out, row.names = FALSE)
cat("wrote", group_out, "\n")

subj_out <- file.path(out_dir, paste0("key_params_subject_", group, ".csv"))
write.csv(subj_rows, subj_out, row.names = FALSE)
cat("wrote", subj_out, "\n")

cat("\n--- group-level key params (", group, ") ---\n", sep = "")
print(group_rows, row.names = FALSE)
cat("done\n")
