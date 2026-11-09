# fit_model.R — fit an M-model (M1, M2, M3) to the new batch's Stan data-list .rds
#
# Ported from formal_batch/analysis/fitting_NN/scripts/fit_model.R.  The fitting logic is
# unchanged; three things are re-pointed at this codebase, two of which fail loudly if
# missed and one of which fails silently:
#
#   1. the project-private R library search (different project root here);
#   2. the rds filename (this batch's naming);
#   3. **the runtime root default must be an ASCII path.**  CmdStan CSV metadata cannot be
#      read back when the path contains CJK characters, and this codebase's directory is
#      named 正式版网络代码.  The old pipeline hit exactly this and added M31_RUNTIME_ROOT
#      for it (see its archive/mac2_header_only_failed_*).  The default is therefore on
#      D: under a pure-ASCII name, not under fitting_dir.
#
# Usage:
#   Rscript fit_model.R --model M3 --group mac1A --mode final [--threads 4] [--seed 13283435]
#   mode: smoke (2 chains x 250+250) | final (4 chains x 2000+2000)

script_args <- commandArgs(trailingOnly = FALSE)
try(Sys.setlocale("LC_CTYPE", "English_United States.utf8"), silent = TRUE)
file_arg <- script_args[grep("--file=", script_args)]
script_path <- if (length(file_arg)) sub("--file=", "", file_arg[1]) else "."
script_dir  <- dirname(normalizePath(script_path, mustWork = FALSE))
fitting_dir <- dirname(script_dir)          # analysis/fitting_NN

rlib <- Sys.getenv("M31_R_LIB", "")
if (nzchar(rlib) && dir.exists(rlib)) {
  # explicit override
} else {
  rlib <- NULL
  # The shared R library lives alongside this repository, not inside it, so scan the
  # parent of the repo root rather than the repo root itself.
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
suppressPackageStartupMessages(library(cmdstanr))

args <- commandArgs(trailingOnly = TRUE)
get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop("Missing value after ", flag)
  args[[i + 1L]]
}
model_id <- get_arg("--model", "M3")
group    <- get_arg("--group", "mac1A")
mode     <- match.arg(get_arg("--mode", "final"), c("smoke", "final"))
threads  <- as.integer(get_arg("--threads", "4"))
seed     <- as.integer(get_arg("--seed", "13283435"))
# Optional sampler overrides.  Left unset they reproduce the historical settings exactly
# (smoke 0.90/12, final 0.95/14), so adding these flags changes nothing for anyone who does
# not pass them.  They exist because two fits (M3 mac2A, M3 mac2B) came out marginal and
# the way to attack that is a smaller step size, not a different model.
adapt_delta_arg   <- get_arg("--adapt-delta", NULL)
max_treedepth_arg <- get_arg("--max-treedepth", NULL)
force_compile <- identical(Sys.getenv("M31_FORCE_COMPILE"), "1")
allow_stale_exe <- identical(Sys.getenv("M31_ALLOW_STALE_EXE"), "1")

MODEL_STAN <- c(M1 = "M1.1.stan", M2 = "M2.1.stan", M3 = "M3.1.stan")
if (!model_id %in% names(MODEL_STAN)) stop("unknown model: ", model_id)

stan_file    <- file.path(fitting_dir, "stan", MODEL_STAN[[model_id]])
cmdstan_path <- Sys.getenv("CMDSTAN", "C:/Users/HONOR/.cmdstan/cmdstan-2.38.0")
# ASCII only — see the header note about CJK paths and CmdStan CSV metadata.
runtime_root <- Sys.getenv("M31_RUNTIME_ROOT", "D:/m31_runtime_h16")
# data/ and fits/ can be relocated; the Stan source stays with the script.  Defaults to
# the fitting dir, so the primary analysis is untouched; the re-draw experiment sets this
# so a second fit of the same group cannot overwrite the archived fit.rds.
data_root    <- Sys.getenv("M31_DATA_ROOT", fitting_dir)
data_dir     <- file.path(data_root, "data")
fit_dir      <- file.path(data_root, "fits")

runtime_tmp <- Sys.getenv("M31_R_TMP", file.path(runtime_root, "tmp"))
dir.create(runtime_tmp, recursive = TRUE, showWarnings = FALSE)
Sys.setenv(TMPDIR = runtime_tmp, TEMP = runtime_tmp, TMP = runtime_tmp)

if (!file.exists(stan_file))   stop("Stan file not found: ", stan_file)
if (!dir.exists(cmdstan_path)) stop("CmdStan not found: ", cmdstan_path)
if (grepl("[^ -~]", runtime_root)) {
  stop("runtime_root must be ASCII-only (CmdStan CSV metadata cannot hold CJK paths): ",
       runtime_root)
}

rds_file <- file.path(data_dir, paste0("formal_osla__h16_b200__", group, ".rds"))
if (!file.exists(rds_file)) stop("rds not found: ", rds_file)
data_list <- readRDS(rds_file)
if (model_id != "M3") data_list$k <- NULL   # only M3 takes the gating gain k

model_runtime_dir <- Sys.getenv("M31_MODEL_RUNTIME_ROOT", file.path(runtime_root, "model"))
dir.create(model_runtime_dir, recursive = TRUE, showWarnings = FALSE)
staged_stan <- file.path(model_runtime_dir, MODEL_STAN[[model_id]])
if (!file.exists(staged_stan) ||
    !identical(unname(tools::md5sum(stan_file)), unname(tools::md5sum(staged_stan)))) {
  file.copy(stan_file, staged_stan, overwrite = TRUE)
}
staged_exe <- sub("\\.stan$", ".exe", staged_stan)
cmdstanr::set_cmdstan_path(cmdstan_path)
if (force_compile || !file.exists(staged_exe)) {
  try(cmdstanr::check_cmdstan_toolchain(fix = TRUE), silent = TRUE)
}

settings <- if (mode == "smoke") {
  list(chains = 2L, parallel_chains = 2L, threads_per_chain = 1L,
       warmup = 250L, sampling = 250L, adapt_delta = 0.90, max_treedepth = 12L)
} else {
  list(chains = 4L, parallel_chains = 4L, threads_per_chain = threads,
       warmup = 2000L, sampling = 2000L, adapt_delta = 0.95, max_treedepth = 14L)
}
if (!is.null(adapt_delta_arg))   settings$adapt_delta   <- as.numeric(adapt_delta_arg)
if (!is.null(max_treedepth_arg)) settings$max_treedepth <- as.integer(max_treedepth_arg)

if (file.exists(staged_exe) && !force_compile && !allow_stale_exe) {
  exe_info <- file.info(staged_exe)
  stan_info <- file.info(staged_stan)
  if (!is.na(exe_info$mtime) && !is.na(stan_info$mtime) && exe_info$mtime < stan_info$mtime) {
    cat("Existing executable is older than staged Stan; recompiling.\n")
    unlink(staged_exe)
  }
} else if (file.exists(staged_exe) && allow_stale_exe) {
  cat("M31_ALLOW_STALE_EXE=1; using existing executable without mtime invalidation.\n")
}

# A staged executable cannot be declared thread-capable after the fact.  Loading it with
# compile = FALSE loses the stan_threads flag, so cmdstanr stops passing num_threads and
# every chain silently runs on ONE core -- while the log still reports the requested
# "threads/chain: 4".  Measured on this box: two identical fits of M3 differ by 5.4x in
# wall time purely from this (1.02 cores per chain instead of ~4), and the only visible
# difference is cmdstan's own banner ("with 4 thread(s) per chain" vs without).  So when
# threads are requested, compile rather than reuse; the ~3 min compile buys back hours.
reuse_exe <- file.exists(staged_exe) && !force_compile && threads <= 1
if (reuse_exe) {
  cat("Loading existing executable", staged_exe, "\n")
  model <- cmdstanr::cmdstan_model(staged_stan, exe_file = staged_exe, compile = FALSE)
} else {
  if (file.exists(staged_exe) && force_compile) {
    cat("M31_FORCE_COMPILE=1; removing existing executable before compile.\n")
    unlink(staged_exe)
  }
  cat("Compiling", MODEL_STAN[[model_id]], "(threading)...\n")
  model <- cmdstanr::cmdstan_model(staged_stan, cpp_options = list(stan_threads = TRUE))
}

tag <- paste0(model_id, "_condA_", group, "_", mode)
cmdstan_output_dir <- file.path(runtime_root, tag, "cmdstan_csv")
dir.create(cmdstan_output_dir, recursive = TRUE, showWarnings = FALSE)

# adapt_delta / max_treedepth are printed because the output tag does NOT encode them --
# an "M3_condA_mac2B_final" directory can now come from either 0.95 or 0.99 and nothing
# else in the artifacts would say which.
cat("Sampling:", tag, "; nSubjects:", data_list$nSubjects,
    "; nTrials:", data_list$nTrials,
    "; chains:", settings$chains, "; threads/chain:", settings$threads_per_chain,
    "; adapt_delta:", settings$adapt_delta, "; max_treedepth:", settings$max_treedepth, "\n")

fit <- model$sample(
  data = data_list,
  seed = seed,
  chains = settings$chains,
  parallel_chains = settings$parallel_chains,
  threads_per_chain = settings$threads_per_chain,
  iter_warmup = settings$warmup,
  iter_sampling = settings$sampling,
  adapt_delta = settings$adapt_delta,
  max_treedepth = settings$max_treedepth,
  refresh = if (mode == "smoke") 100L else 400L,
  output_dir = cmdstan_output_dir,
  output_basename = paste0(MODEL_STAN[[model_id]], "_", tag),
  save_cmdstan_config = TRUE
)

fit_dir_grp <- file.path(fit_dir, tag)
dir.create(fit_dir_grp, recursive = TRUE, showWarnings = FALSE)
fit$save_object(file.path(fit_dir_grp, "fit.rds"))

write.csv(fit$summary(), file.path(fit_dir_grp, "parameter_summary.csv"), row.names = FALSE)
write.csv(fit$diagnostic_summary(), file.path(fit_dir_grp, "diagnostic_summary.csv"), row.names = FALSE)

cat("\nFit saved:", file.path(fit_dir_grp, "fit.rds"), "\n")
cat("--- diagnostics ---\n")
print(fit$diagnostic_summary(), row.names = FALSE)
