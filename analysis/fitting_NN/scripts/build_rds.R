# build_rds.R — convert the new batch's m3_subjects CSVs to Stan data-list .rds
#
# Ported from formal_batch/analysis/fitting_NN/scripts/build_rds.R.  The pivot logic is
# unchanged; only the input/output directories are re-pointed at this codebase.
#
# Reads the long tables written by extract_m3_subjects.py
#   (subject_index, trial_in_block, choice, reward, p0)
# and pivots each into the [trial, subject] arrays M1-M3 expect:
#
#   list(nSubjects, nTrials, choice[nTrials,nSubjects], reward[nTrials,nSubjects],
#        mean_p[1,nSubjects], k)
#
# 'reward' here is the machine OUTCOME (0/1), as the Stan likelihood requires; see the
# header of extract_m3_subjects.py for why that column name is a trap.
#
# Self-locating: resolves its own path so it can be run from anywhere without passing the
# CJK project path on the command line (CmdStan CSV metadata chokes on CJK paths, which is
# why fit_model.R also stages its runtime under an ASCII directory).

script_args <- commandArgs(trailingOnly = FALSE)
file_arg <- script_args[grep("--file=", script_args)]
script_path <- if (length(file_arg)) sub("--file=", "", file_arg[1]) else "."
script_dir <- dirname(normalizePath(script_path, mustWork = FALSE))
fitting_dir <- dirname(script_dir)              # analysis/fitting_NN
# Where metrics/ and data/ live.  Defaults to the fitting dir itself, so the primary
# analysis is untouched; the re-draw experiment points M31_DATA_ROOT at its own tree so it
# cannot overwrite the archived CSVs and rds files.
data_root <- Sys.getenv("M31_DATA_ROOT", fitting_dir)
csv_dir <- file.path(data_root, "metrics")
out_dir <- file.path(data_root, "data")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

K <- 10  # M3 gating gain (fixed; identical to the old pipeline so results stay comparable)

CSV_PREFIX <- "formal_osla__h16_b200__"

build_one <- function(csv_name, rds_name) {
  csv_file <- file.path(csv_dir, csv_name)
  if (!file.exists(csv_file)) stop("CSV not found: ", csv_file)

  df <- read.csv(csv_file, stringsAsFactors = FALSE)
  required <- c("subject_index", "trial_in_block", "choice", "reward", "p0")
  if (!all(required %in% names(df))) stop("missing columns in ", csv_name)

  df <- df[order(df$subject_index, df$trial_in_block), ]
  n_subjects <- length(unique(df$subject_index))
  n_trials <- length(unique(df$trial_in_block))
  if (nrow(df) != n_subjects * n_trials) stop("row mismatch in ", csv_name)

  choice <- matrix(as.integer(df$choice), nrow = n_trials, ncol = n_subjects)
  reward <- matrix(as.integer(df$reward), nrow = n_trials, ncol = n_subjects)
  p0 <- sapply(sort(unique(df$subject_index)), function(s) df$p0[df$subject_index == s][1])

  # Fail loudly rather than handing Stan a mis-shaped or out-of-range design.
  if (!all(choice %in% c(0L, 1L))) stop("choice outside {0,1} in ", csv_name)
  if (!all(reward %in% c(0L, 1L))) stop("reward outside {0,1} in ", csv_name)
  if (length(unique(p0)) != 1L) stop("p0 is not constant within ", csv_name)

  data_list <- list(
    nSubjects = n_subjects,
    nTrials   = n_trials,
    choice    = choice,
    reward    = reward,
    mean_p    = matrix(as.numeric(p0), nrow = 1L),
    k         = K
  )

  out <- file.path(out_dir, rds_name)
  saveRDS(data_list, out)
  cat(sprintf("%s -> %s | nSubjects=%d nTrials=%d p0=%.2f choice_rate=%.3f reward_rate=%.3f\n",
              csv_name, rds_name, n_subjects, n_trials, p0[1], mean(choice), mean(reward)))
}

for (group in c("mac1A", "mac1B", "mac2A", "mac2B")) {
  build_one(paste0(CSV_PREFIX, group, "_m3_subjects.csv"),
            paste0(CSV_PREFIX, group, ".rds"))
}
cat("done\n")
