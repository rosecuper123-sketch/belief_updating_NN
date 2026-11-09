# fit_multi_episode.R — fit the multi-episode M3 jointly over N episodes per network.
#
# Why this exists
# ---------------
# The main pipeline (extract_m3_subjects.py -> build_rds.R -> fit_model.R) represents each
# network by ONE randomly drawn episode out of the 200 it has on disk.  That single draw is
# noisy, and mac2B showed it plainly: the fit had to be re-drawn with a different sampling
# seed to converge at all (README 3.9), leaving theta = 7.32 +/- 2.57 with min ESS 96.5.
#
# `stan/M3.1_multi_episode.stan` instead treats the episode as a DIMENSION: the episodes of
# one network share that network's cognitive parameters, each resets its own latent state,
# and all their trial-level likelihoods enter the same posterior.  This script is the
# self-contained driver for that model.  It touches none of the existing three scripts and
# writes nothing into the existing data/ fits/ metrics/ trees.
#
# Two things this script deliberately does NOT assume
# ---------------------------------------------------
#   * It does not roll out any network.  All 200 episodes per (network, group) already
#     exist in the frozen_tests tree; this is a SELECTION problem, not a generation one.
#   * It does not flatten (network, episode) into 300 subjects.  The model wants
#     nSubjects = 30 (networks) and nEpisodes = 10, with choice[nSubjects, nEpisodes,
#     nTrials].  Flattening would discard the entire point -- that the 10 episodes share
#     one network's parameters -- and silently fit 300 independent subjects instead.
#     NOTE the axis order is the REVERSE of M3.1.stan's [nTrials, nSubjects].
#
# The old "30 networks must face 30 distinct outcome sequences" invariant is deliberately
# NOT carried over.  It existed to stop 30 independent subjects sharing a sequence; here
# the episodes are repeated measures within a network and the model knows it.  All networks
# in a group actually do share the same outcome-by-episode sequences (same TEST_SEED), so
# drawing one shared set of episodes makes episode e mean the same thing for every network.
#
# Usage:
#   Rscript fit_multi_episode.R [--group mac2B] [--n-episodes 10] [--sampling-seed 20261002]
#                               [--mode final] [--chains 4] [--warmup 1000] [--sampling 1000]
#                               [--threads 4] [--stage all|extract|build|fit|report]
#
# Outputs (all under multi_episode/, nothing outside it):
#   metrics/formal_osla__<arm>__<group>_m3_ep<N>_subjects.csv   long table, the auditable data
#   metrics/<group>_ep<N>_manifest.csv                          which episodes were drawn
#   data/formal_osla__<arm>__<group>_ep<N>.rds                  Stan data list
#   fits/<model>_<group>_<mode>_ep<N>/{fit.rds,parameter_summary.csv,diagnostic_summary.csv,meta.json}
#   metrics/<group>_ep<N>_vs_single.csv                         group params vs the 1-episode fit

script_args <- commandArgs(trailingOnly = FALSE)
try(Sys.setlocale("LC_CTYPE", "English_United States.utf8"), silent = TRUE)
file_arg <- script_args[grep("--file=", script_args)]
script_path <- if (length(file_arg)) sub("--file=", "", file_arg[1]) else "."
script_dir  <- dirname(normalizePath(script_path, mustWork = FALSE))
fitting_dir <- dirname(script_dir)          # analysis/fitting_NN

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
suppressPackageStartupMessages(library(cmdstanr))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop("Missing value after ", flag)
  args[[i + 1L]]
}

group        <- get_arg("--group", "mac2B")
n_episodes   <- as.integer(get_arg("--n-episodes", "10"))
sampling_seed<- as.integer(get_arg("--sampling-seed", "20261002"))
mode         <- match.arg(get_arg("--mode", "final"), c("check", "smoke", "final"))
stage        <- match.arg(get_arg("--stage", "all"),
                          c("all", "extract", "build", "fit", "report"))
threads      <- as.integer(get_arg("--threads", "4"))
seed         <- as.integer(get_arg("--seed", "13283435"))
arm          <- get_arg("--arm", "h16_b200")
protocol     <- get_arg("--protocol", "conflict_only")

model_id  <- "M3_10ep"
STAN_FILE <- "M3.1_multi_episode.stan"
K_GAIN    <- 10       # M3's fixed gating gain

# --- chain settings ---------------------------------------------------------------
# check: end-to-end plumbing test only -- 1 chain, 20 + 20.  It is meant to finish in a
#        couple of minutes and prove the whole chain (extract -> build -> compile -> sample
#        -> save -> report) works.  Its posterior is meaningless; do not read it.
# smoke: 2 chains x 250 + 250, for a real look at divergences and mixing.
# final: 4 chains x (1000 + 1000 by default).
#
# NOTE ON SIZE before running final.  This model stores 7 generated-quantity arrays of
# [nSubjects, nEpisodes, nTrials]; at 30 x 10 x 120 that is 252,000 values PER DRAW against
# 25,200 for the single-episode M3.  At 4 chains x 1000 draws expect ~10 GB of CmdStan CSVs
# and a fit.rds of ~8 GB.  Check that the machine has the RAM for that (fit$save_object
# materialises the whole thing) or trim the generated quantities in the .stan first.
settings <- if (mode == "check") {
  list(chains = 1L, warmup = 20L, sampling = 20L, adapt_delta = 0.80, max_treedepth = 8L)
} else if (mode == "smoke") {
  list(chains = 2L, warmup = 250L, sampling = 250L, adapt_delta = 0.90, max_treedepth = 12L)
} else {
  list(chains = as.integer(get_arg("--chains", "4")),
       warmup   = as.integer(get_arg("--warmup", "1000")),
       sampling = as.integer(get_arg("--sampling", "1000")),
       adapt_delta = 0.95, max_treedepth = 14L)
}

# --- paths ------------------------------------------------------------------------
GROUPS      <- c(mac1A = 0.3, mac1B = 0.7, mac2A = 0.3, mac2B = 0.7)   # group -> cue
NETWORK_SEED_BASE <- 20260804L
N_NETWORKS        <- 30L

if (!group %in% names(GROUPS)) stop("unknown group: ", group)
cue <- unname(GROUPS[[group]])

PACKAGE_ROOT <- dirname(dirname(fitting_dir))     # formal_run_001
frozen_root <- Sys.getenv(
  "M31_FROZEN_ROOT",
  file.path(PACKAGE_ROOT, "outputs", "output_formal", "runs",
            "frozen_tests", arm, protocol))
out_root <- Sys.getenv("M31_MULTI_ROOT", file.path(fitting_dir, "multi_episode"))
metrics_dir <- file.path(out_root, "metrics")
data_dir    <- file.path(out_root, "data")
fits_dir    <- file.path(out_root, "fits")
for (d in c(metrics_dir, data_dir, fits_dir)) dir.create(d, recursive = TRUE, showWarnings = FALSE)

cmdstan_path <- Sys.getenv("CMDSTAN", "C:/Users/HONOR/.cmdstan/cmdstan-2.38.0")
# ASCII only: CmdStan cannot read its own CSV metadata back through a CJK path, and this
# codebase's directory is literally named 正式版网络代码.  Same trap fit_model.R documents.
runtime_root <- Sys.getenv("M31_RUNTIME_ROOT", "D:/m31_runtime_h16/multi_episode")
if (grepl("[^ -~]", runtime_root)) {
  stop("runtime_root must be ASCII-only (CmdStan CSV metadata cannot hold CJK paths): ",
       runtime_root)
}
runtime_tmp <- file.path(runtime_root, "tmp")
dir.create(runtime_tmp, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(TMPDIR = runtime_tmp, TEMP = runtime_tmp, TMP = runtime_tmp)

stem      <- sprintf("%s_ep%d", group, n_episodes)
long_csv  <- file.path(metrics_dir, sprintf("formal_osla__%s__%s_m3_ep%d_subjects.csv",
                                            arm, group, n_episodes))
rds_file  <- file.path(data_dir, sprintf("formal_osla__%s__%s_ep%d.rds",
                                         arm, group, n_episodes))
manifest_csv <- file.path(metrics_dir, sprintf("%s_manifest.csv", stem))
fit_tag   <- sprintf("%s_%s_%s_ep%d", model_id, group, mode, n_episodes)
fit_out   <- file.path(fits_dir, fit_tag)

network_dir <- function(k) {
  file.path(frozen_root, sprintf("formal_osla_seed%d", NETWORK_SEED_BASE + k - 1L), group)
}

# =================================================================================
# stage 1: extract — pick the shared episode set, write the long table + manifest
# =================================================================================
stage_extract <- function() {
  first <- file.path(network_dir(1L), "trials.csv")
  if (!file.exists(first)) stop("trials.csv not found: ", first)

  available <- sort(unique(read.csv(first, colClasses = c(episode_id = "integer"))$episode_id))
  if (length(available) < n_episodes)
    stop(sprintf("%s: only %d episodes available, need %d", group, length(available), n_episodes))

  # ONE shared draw for the whole group.  Episodes are exogenous here (block-balanced
  # outcome schedules, independent of the network's actions) and every network in a group
  # faces the identical outcome-by-episode sequence, so sharing makes episode e comparable
  # across networks instead of being a different draw for each.
  set.seed(sampling_seed)
  drawn <- sort(sample(available, size = n_episodes, replace = FALSE))
  cat(sprintf("episodes drawn (%s, seed %d, shared across all %d networks): %s\n",
              group, sampling_seed, N_NETWORKS, paste(drawn, collapse = ", ")))

  blocks <- vector("list", N_NETWORKS * n_episodes)
  manifest <- vector("list", N_NETWORKS * n_episodes)
  n_trials <- NULL
  pos <- 0L

  for (k in seq_len(N_NETWORKS)) {
    path <- file.path(network_dir(k), "trials.csv")
    if (!file.exists(path)) stop("trials.csv not found: ", path)
    # Read only what is needed, then keep just the drawn episodes -- holding all 30 files
    # (720k rows) would be wasteful for 10/200 of the content.
    df <- read.csv(path, colClasses = c(episode_id = "integer", trial_id = "integer",
                                        action = "integer", outcome = "integer",
                                        cue_probability = "numeric"))
    df <- df[df$episode_id %in% drawn, , drop = FALSE]
    df <- df[order(df$episode_id, df$trial_id), , drop = FALSE]

    for (e in seq_along(drawn)) {
      block <- df[df$episode_id == drawn[e], , drop = FALSE]
      if (nrow(block) == 0)
        stop(sprintf("%s/%s: episode %d has no rows", group, basename(dirname(path)), drawn[e]))
      if (is.null(n_trials)) n_trials <- nrow(block)
      else if (nrow(block) != n_trials)
        stop(sprintf("network %d ep %d: %d trials, expected %d", k, drawn[e], nrow(block), n_trials))
      if (anyNA(block$action) || anyNA(block$outcome))
        stop(sprintf("network %d ep %d: NA in action/outcome", k, drawn[e]))
      if (!all(block$action %in% c(0L, 1L)))
        stop(sprintf("network %d ep %d: action outside {0,1}", k, drawn[e]))
      if (!all(block$outcome %in% c(0L, 1L)))
        stop(sprintf("network %d ep %d: outcome outside {0,1}", k, drawn[e]))
      if (!isTRUE(all.equal(unique(block$cue_probability), cue)))
        stop(sprintf("network %d ep %d: cue is %s, expected %s", k, drawn[e],
                     paste(unique(block$cue_probability), collapse = ","), cue))

      pos <- pos + 1L
      # choice <- action, reward <- outcome.  NOT the trials.csv `reward` column, which is
      # the +/-1 payoff action*(2*outcome-1) and would silently break bernoulli_logit.
      blocks[[pos]] <- data.frame(
        subject_index  = k - 1L,
        episode_index  = e,
        trial_in_block = seq_len(nrow(block)),
        choice         = as.integer(block$action),
        reward         = as.integer(block$outcome),
        p0             = cue
      )
      manifest[[pos]] <- data.frame(
        group = group, subject_index = k - 1L,
        network_seed = NETWORK_SEED_BASE + k - 1L,
        episode_index = e, episode_id = drawn[e],
        n_trials = nrow(block),
        bet_rate = mean(block$action),
        outcome_rate = mean(block$outcome),
        cue = cue
      )
    }
  }

  long <- do.call(rbind, blocks)
  expected <- N_NETWORKS * n_episodes * n_trials
  if (nrow(long) != expected)
    stop(sprintf("long table has %d rows, expected %d", nrow(long), expected))
  write.csv(long, long_csv, row.names = FALSE)
  write.csv(do.call(rbind, manifest), manifest_csv, row.names = FALSE)

  cat(sprintf("wrote %s  (%d networks x %d episodes x %d trials = %d rows)\n",
              basename(long_csv), N_NETWORKS, n_episodes, n_trials, nrow(long)))
  cat(sprintf("      choice mean %.3f | reward mean %.3f\n",
              mean(long$choice), mean(long$reward)))
  invisible(n_trials)
}

# =================================================================================
# stage 2: build — long table -> Stan data list, [network, episode, trial]
# =================================================================================
stage_build <- function() {
  if (!file.exists(long_csv)) stop("run --stage extract first; missing ", long_csv)
  df <- read.csv(long_csv)
  n_subjects <- length(unique(df$subject_index))
  n_eps      <- length(unique(df$episode_index))
  n_trials   <- length(unique(df$trial_in_block))

  if (n_subjects != N_NETWORKS)
    stop(sprintf("nSubjects is %d, expected %d", n_subjects, N_NETWORKS))
  if (n_eps != n_episodes)
    stop(sprintf("nEpisodes is %d, expected %d (was this file built with another --n-episodes?)",
                 n_eps, n_episodes))
  if (nrow(df) != n_subjects * n_eps * n_trials)
    stop(sprintf("row mismatch: %d rows for %d x %d x %d", nrow(df),
                 n_subjects, n_eps, n_trials))

  # Every (network, episode) cell must be complete, or a silent reshape would pad it.
  cells <- table(df$subject_index, df$episode_index)
  if (any(cells != n_trials))
    stop(sprintf("incomplete cells: %d of %d (network,episode) pairs do not have %d trials",
                 sum(cells != n_trials), length(cells), n_trials))

  choice <- array(NA_integer_, dim = c(n_subjects, n_eps, n_trials))
  reward <- array(NA_integer_, dim = c(n_subjects, n_eps, n_trials))
  for (s in seq_len(n_subjects)) for (e in seq_len(n_eps)) {
    sel <- df$subject_index == (s - 1L) & df$episode_index == e
    ord <- order(df$trial_in_block[sel])
    choice[s, e, ] <- as.integer(df$choice[sel][ord])
    reward[s, e, ] <- as.integer(df$reward[sel][ord])
  }
  p0 <- sapply(seq_len(n_subjects),
               function(s) df$p0[df$subject_index == (s - 1L)][1])
  if (length(unique(p0)) != 1L) stop("p0 is not constant within the group")
  # mean_p is per (network, episode) in the model even though it is currently constant --
  # filling one row would leave the rest uninitialised and Stan would reject it.
  mean_p <- matrix(as.numeric(p0), nrow = n_subjects, ncol = n_eps)

  stopifnot(identical(dim(choice), c(n_subjects, n_eps, n_trials)),
            identical(dim(reward), c(n_subjects, n_eps, n_trials)),
            identical(dim(mean_p), c(n_subjects, n_eps)))
  if (!all(choice %in% c(0L, 1L))) stop("choice outside {0,1}")
  if (!all(reward %in% c(0L, 1L))) stop("reward outside {0,1}")
  if (any(mean_p <= 0 | mean_p >= 1))
    stop("mean_p must be strictly inside (0,1) -- the model rejects otherwise")

  data_list <- list(nSubjects = n_subjects, nEpisodes = n_eps, nTrials = n_trials,
                    choice = choice, reward = reward, mean_p = mean_p, k = K_GAIN)
  saveRDS(data_list, rds_file)
  cat(sprintf("wrote %s  choice[%d,%d,%d] reward[%d,%d,%d] mean_p[%d,%d] k=%g\n",
              basename(rds_file), n_subjects, n_eps, n_trials,
              n_subjects, n_eps, n_trials, n_subjects, n_eps, K_GAIN))
  cat(sprintf("      total observations %d (single-episode pipeline used %d)\n",
              n_subjects * n_eps * n_trials, n_subjects * n_trials))
  invisible(data_list)
}

# =================================================================================
# stage 3: fit
# =================================================================================
build_model <- function() {
  if (!file.exists(rds_file)) stop("run --stage build first; missing ", rds_file)
  stan_file <- file.path(fitting_dir, "stan", STAN_FILE)
  if (!file.exists(stan_file)) stop("Stan file not found: ", stan_file)
  if (!dir.exists(cmdstan_path)) stop("CmdStan not found: ", cmdstan_path)

  model_runtime_dir <- Sys.getenv("M31_MODEL_RUNTIME_ROOT", file.path(runtime_root, "model"))
  dir.create(model_runtime_dir, recursive = TRUE, showWarnings = FALSE)
  staged_stan <- file.path(model_runtime_dir, STAN_FILE)
  if (!file.exists(staged_stan) ||
      !identical(unname(tools::md5sum(stan_file)), unname(tools::md5sum(staged_stan)))) {
    file.copy(stan_file, staged_stan, overwrite = TRUE)
  }
  staged_exe <- sub("\\.stan$", ".exe", staged_stan)
  cmdstanr::set_cmdstan_path(cmdstan_path)

  force_compile <- identical(Sys.getenv("M31_FORCE_COMPILE"), "1")
  if (force_compile && file.exists(staged_exe)) unlink(staged_exe)
  if (file.exists(staged_exe) && !force_compile) {
    cat("Loading existing executable", staged_exe, "\n")
    model <- cmdstanr::cmdstan_model(staged_stan, exe_file = staged_exe, compile = FALSE)
  } else {
    cat("Compiling", STAN_FILE, "(threading)...\n")
    try(cmdstanr::check_cmdstan_toolchain(fix = TRUE), silent = TRUE)
    model <- cmdstanr::cmdstan_model(staged_stan, cpp_options = list(stan_threads = TRUE))
  }
  model
}

# Reply to the README 5.1 note that threads_per_chain silently does nothing.  The old
# pipeline reused an executable compiled WITHOUT stan_threads, so the flag was ignored.
# This model is compiled fresh here, so threading should genuinely work -- but rather than
# assume it across a multi-hour run, spend one minute proving it on a 2-iteration probe.
probe_threads <- function(model, data_list) {
  if (threads <= 1L) return(1L)
  probe_dir <- file.path(runtime_root, "probe")
  # Create it first: cmdstanr errors if output_dir is missing, and that error would be
  # caught below and reported as a threading failure -- sending the run single-threaded
  # for a reason that has nothing to do with threads.
  dir.create(probe_dir, recursive = TRUE, showWarnings = FALSE)
  # show_messages=FALSE and suppressWarnings: with 2 iterations and no adaptation every
  # transition diverges, and cmdstan's warning about that is noise here -- the probe only
  # asks whether the executable accepts threads.
  ok <- tryCatch({
    suppressWarnings(model$sample(
      data = data_list, seed = seed, chains = 1L, parallel_chains = 1L,
      threads_per_chain = threads, iter_warmup = 2L, iter_sampling = 2L,
      adapt_delta = 0.8, max_treedepth = 8L, refresh = 0L,
      show_messages = FALSE,
      output_dir = probe_dir,
      output_basename = paste0("probe_", fit_tag)))
    TRUE
  }, error = function(e) { cat("  thread probe failed:", conditionMessage(e), "\n"); FALSE })
  if (ok) { cat(sprintf("  threading OK -> using %d threads/chain\n", threads)); threads }
  else {
    cat("  falling back to 1 thread/chain (see the error above; if it is not about\n")
    cat("  threading, the probe itself is broken and this fallback is misleading)\n")
    1L
  }
}

stage_fit <- function() {
  data_list <- readRDS(rds_file)
  cat(sprintf("data: nSubjects=%d nEpisodes=%d nTrials=%d\n",
              data_list$nSubjects, data_list$nEpisodes, data_list$nTrials))

  model <- build_model()
  threads_used <- probe_threads(model, data_list)

  cmdstan_output_dir <- file.path(runtime_root, fit_tag, "cmdstan_csv")
  dir.create(cmdstan_output_dir, recursive = TRUE, showWarnings = FALSE)

  cat(sprintf("sampling %s: %d chains x (%d warmup + %d sampling), %d threads/chain\n",
              fit_tag, settings$chains, settings$warmup, settings$sampling, threads_used))
  t0 <- Sys.time()
  fit <- model$sample(
    data = data_list, seed = seed,
    chains = settings$chains, parallel_chains = settings$chains,
    threads_per_chain = threads_used,
    iter_warmup = settings$warmup, iter_sampling = settings$sampling,
    adapt_delta = settings$adapt_delta, max_treedepth = settings$max_treedepth,
    refresh = if (mode == "check") 5L else if (mode == "smoke") 100L else 200L,
    output_dir = cmdstan_output_dir,
    output_basename = paste0(STAN_FILE, "_", fit_tag),
    save_cmdstan_config = TRUE
  )
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

  dir.create(fit_out, recursive = TRUE, showWarnings = FALSE)
  fit$save_object(file.path(fit_out, "fit.rds"))

  # Summarising EVERY variable means Rhat/ESS for ~252,715 of them -- minutes of compute and
  # hundreds of MB of numbers nobody reads, and the seven per-trial arrays are all of it.
  # Summarise the model parameters plus the episode-level likelihoods by default; the full
  # sweep stays available behind --full-summary (e.g. to check the GQ arrays' own Rhat).
  per_trial <- c("log_lik", "p_time", "L_time", "w_time", "lr_pos_time", "lr_neg_time",
                 "y_pred")
  all_vars <- tryCatch(fit$metadata()$stan_variables, error = function(e) NULL)
  if (is.null(all_vars) || !length(all_vars)) {
    cat("  (could not list variables; summarising everything, this will be slow)\n")
    summ <- fit$summary()
  } else {
    keep <- if ("--full-summary" %in% args) all_vars else setdiff(all_vars, per_trial)
    cat(sprintf("summarising %d of %d variables (per-trial arrays skipped)\n",
                length(keep), length(all_vars)))
    summ <- fit$summary(variables = keep)
  }
  write.csv(summ, file.path(fit_out, "parameter_summary.csv"), row.names = FALSE)
  write.csv(fit$diagnostic_summary(), file.path(fit_out, "diagnostic_summary.csv"),
            row.names = FALSE)

  drawn <- unique(read.csv(manifest_csv)$episode_id)
  git_commit <- tryCatch(system("git rev-parse --short HEAD", intern = TRUE), error = function(e) NA)
  meta <- list(
    model = model_id, stan_file = STAN_FILE, group = group,
    n_subjects = data_list$nSubjects, n_episodes = data_list$nEpisodes,
    n_trials = data_list$nTrials, cue = cue,
    sampling_seed = sampling_seed, episode_ids = drawn,
    episode_assignment = "shared across all networks within the group",
    stan_seed = seed, mode = mode, chains = settings$chains,
    warmup = settings$warmup, sampling_iters = settings$sampling,
    adapt_delta = settings$adapt_delta, max_treedepth = settings$max_treedepth,
    threads_per_chain = threads_used, wall_seconds = elapsed,
    r_version = R.version.string, git_commit = git_commit, run_at = format(Sys.time())
  )
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    writeLines(jsonlite::toJSON(meta, auto_unbox = TRUE, pretty = TRUE),
               file.path(fit_out, "meta.json"))
  }

  cat(sprintf("\nfit saved: %s  (%.0f s = %.1f min)\n",
              file.path(fit_out, "fit.rds"), elapsed, elapsed / 60))
  cat("--- diagnostics ---\n")
  print(fit$diagnostic_summary(), row.names = FALSE)
  invisible(fit)
}

# =================================================================================
# stage 4: report — the point of the exercise, next to the 1-episode fit
# =================================================================================
GROUP_PARAMS <- c("lr_pos_mu1", "lr_pos_mu2", "lr_neg_mu1", "lr_neg_mu2",
                  "tau_mu", "theta_mu_out")

stage_report <- function() {
  new_path <- file.path(fit_out, "parameter_summary.csv")
  if (!file.exists(new_path)) stop("no fit to report on; run --stage fit first")
  new <- read.csv(new_path)
  old_path <- file.path(fitting_dir, "fits", sprintf("M3_condA_%s_final", group),
                        "parameter_summary.csv")
  if (!file.exists(old_path)) {
    cat("no single-episode fit at", old_path, "-- printing the multi-episode params only\n")
    print(new[new$variable %in% GROUP_PARAMS, c("variable", "mean", "sd", "rhat", "ess_bulk")],
          row.names = FALSE)
    return(invisible(NULL))
  }
  old <- read.csv(old_path)
  pick <- function(df, v) df[df$variable == v, , drop = FALSE]
  rows <- lapply(GROUP_PARAMS, function(v) {
    a <- pick(old, v); b <- pick(new, v)
    if (!nrow(a) || !nrow(b)) return(NULL)
    data.frame(parameter = v,
               single_mean = a$mean[1], single_sd = a$sd[1],
               multi_mean  = b$mean[1], multi_sd  = b$sd[1],
               sd_ratio = b$sd[1] / a$sd[1],
               multi_rhat = b$rhat[1], multi_ess = b$ess_bulk[1])
  })
  cmp <- do.call(rbind, rows)

  # A --mode check or smoke posterior is not a result, and this table is shaped exactly
  # like one -- it will happily print sd_ratio to four digits from 20 draws.  Refuse to
  # write the comparison file unless the fit actually converged, so a garbage table cannot
  # be mistaken for a finding (or picked up by anything downstream).
  max_rhat <- suppressWarnings(max(cmp$multi_rhat, na.rm = TRUE))
  min_ess  <- suppressWarnings(min(cmp$multi_ess, na.rm = TRUE))
  converged <- is.finite(max_rhat) && max_rhat <= 1.05 && is.finite(min_ess) && min_ess >= 100

  cat(sprintf("\n===== group parameters: %d episodes vs 1 episode (%s) =====\n",
              n_episodes, group))
  print(cmp, row.names = FALSE, digits = 4)
  cat("\nsd_ratio < 1 means the multi-episode posterior is TIGHTER than the single-episode\n")
  cat("one -- which is the whole reason for fitting 10 episodes instead of drawing 1.\n")

  if (!converged) {
    cat(sprintf("\n*** NOT CONVERGED (max Rhat %.2f, min ESS %.0f) -- the table above is\n",
                max_rhat, min_ess))
    cat("*** NOT interpretable.  Expected for --mode check/smoke; it means this fit is\n")
    cat("*** too short to compare.  No comparison CSV written.\n")
    return(invisible(cmp))
  }
  out <- file.path(metrics_dir, sprintf("%s_vs_single.csv", stem))
  write.csv(cmp, out, row.names = FALSE)
  cat("converged (max Rhat", sprintf("%.2f", max_rhat), ", min ESS", sprintf("%.0f", min_ess),
      ") -- saved:", out, "\n")
  invisible(cmp)
}

# =================================================================================
if (stage %in% c("all", "extract")) stage_extract()
if (stage %in% c("all", "build"))   stage_build()
if (stage %in% c("all", "fit"))     stage_fit()
if (stage %in% c("all", "report"))  stage_report()
