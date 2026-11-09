"""Train the formal batch grid: h=16, 200 updates, 30 seeds (20260804-20260833).

This records the invocation that produced this batch's `runs/arms/h16_b200/`.
It adds no new training logic -- it only runs the two generic entry points in
this portable package with the exact arguments this batch used:

    1. scripts/testing/selfcheck.py       guardrail, must pass first
    2. scripts/training/prerun_batch.py   the 30-seed training grid

Both are executed as subprocesses on purpose.  They pin the BLAS/OMP thread
count to 1 *before* importing numpy or torch, and assert
`torch.get_num_threads() == 1`; importing them from here would import torch
first and break that guarantee.  Multi-threaded matmul changes the
floating-point reduction order and the same seed diverges (measured 2026-09-20
on h=32/seed20260804).

OUT_ROOT is anchored on this file's location so the batch folder can be moved
as a unit, per the CLAUDE.md batch convention.

Usage (from anywhere; paths resolve from the script):
    python train_batch.py --dry-run          # print the commands only
    python train_batch.py                    # selfcheck + 30 trainings
    python train_batch.py --skip-existing    # regenerate manifest, no retrain
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

OUT_ROOT = Path(__file__).resolve().parent.parent / "runs"
CODE_DIR = Path(__file__).resolve().parents[3]  # formal_run_001
REPO_ROOT = CODE_DIR.parent                     # belief_update_rnn
SELF_CHECK = CODE_DIR / "scripts" / "testing" / "selfcheck.py"
TRAIN_BATCH = CODE_DIR / "scripts" / "training" / "prerun_batch.py"
# Reuse the interpreter that launched this driver.  This keeps the package
# movable and avoids assuming that a sibling repository-level .venv exists.
PYTHON = Path(sys.executable).resolve()

SEEDS = tuple(range(20260804, 20260834))  # 30 seeds
HIDDEN_SIZES = (16,)
BUDGETS = (200,)
ARM = "h16_b200"

# Must match the pinned-thread contract of the scripts being called.
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _check_paths() -> None:
    for path, what in (
        (PYTHON, "venv python"),
        (SELF_CHECK, "portable self-check entry point"),
        (TRAIN_BATCH, "portable batch-training entry point"),
    ):
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}\n(repo root resolved to {REPO_ROOT})")


def _run(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    if dry_run:
        return
    env = {**os.environ, **THREAD_ENV}
    result = subprocess.run(cmd, cwd=CODE_DIR, env=env)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {printable}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="pass through to prerun_batch.py")
    parser.add_argument("--skip-selfcheck", action="store_true",
                        help="skip the guardrail (not recommended)")
    args = parser.parse_args()

    _check_paths()
    py = str(PYTHON)

    print(f"OUT_ROOT  : {OUT_ROOT}")
    print(f"CODE_DIR  : {CODE_DIR}")
    print(f"grid      : {len(SEEDS)} seeds x hidden{HIDDEN_SIZES} x budgets{BUDGETS} -> arm {ARM}")

    if not args.skip_selfcheck:
        _run([py, str(SELF_CHECK)], dry_run=args.dry_run)

    cmd = [
        py, str(TRAIN_BATCH),
        "--output-root", str(OUT_ROOT),
        "--hidden-sizes", *[str(h) for h in HIDDEN_SIZES],
        "--budgets", *[str(b) for b in BUDGETS],
        "--seeds", *[str(s) for s in SEEDS],
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    _run(cmd, dry_run=args.dry_run)

    print("\nnext: run_tests.py" if not args.dry_run else "\n(dry run: nothing executed)")


if __name__ == "__main__":
    sys.exit(main())
