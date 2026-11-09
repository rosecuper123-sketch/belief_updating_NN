"""Run the frozen tests for the formal batch: arm h16_b200, conflict_only.

This records the invocation that produced this batch's
`runs/frozen_tests/h16_b200/` (120 cells) plus its structural-invariant report
(`frozen_test_checks.json`) and checkpoint-sha256 ledger
(`frozen_tests/h16_b200/manifest_provenance.json`).

It adds no testing logic -- it only runs this portable package's generic entry
point `scripts/testing/run_prerun_tests.py` with the arguments this batch used.
It is executed as a subprocess on purpose: that script pins the BLAS/OMP thread
count to 1 before importing torch and asserts `torch.get_num_threads() == 1`.
See train_batch.py for the same note.

OUT_ROOT is anchored on this file's location (CLAUDE.md batch convention).

Usage:
    python run_tests.py --dry-run       # print the command only
    python run_tests.py                 # run frozen tests + checks + ledger
    python run_tests.py --check-only    # re-run invariant checks on existing outputs
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
TEST_BATCH = CODE_DIR / "scripts" / "testing" / "run_prerun_tests.py"
# Reuse the interpreter that launched this driver.  This keeps the package
# movable and avoids assuming that a sibling repository-level .venv exists.
PYTHON = Path(sys.executable).resolve()

ARM = "h16_b200"
PROTOCOLS = ("conflict_only",)

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _check_paths() -> None:
    for path, what in (
        (PYTHON, "venv python"),
        (TEST_BATCH, "portable frozen-test entry point"),
    ):
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}\n(repo root resolved to {REPO_ROOT})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print command, run nothing")
    parser.add_argument("--check-only", action="store_true",
                        help="only run structural invariant checks (no re-testing)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="pass through to run_prerun_tests.py")
    parser.add_argument("--arms", nargs="+", default=[ARM])
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    args = parser.parse_args()

    _check_paths()
    cmd = [
        str(PYTHON), str(TEST_BATCH),
        "--output-root", str(OUT_ROOT),
        "--arms", *args.arms,
        "--protocols", *args.protocols,
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.check_only:
        cmd.append("--check-only")

    printable = " ".join(cmd)
    print(f"\nOUT_ROOT : {OUT_ROOT}")
    print(f"\n$ {printable}", flush=True)
    if args.dry_run:
        print("\n(dry run: nothing executed)")
        return

    result = subprocess.run(cmd, cwd=CODE_DIR, env={**os.environ, **THREAD_ENV})
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {printable}")
    print("\ndone: see runs/frozen_test_checks.json and "
          "runs/frozen_tests/<arm>/manifest_provenance.json")


if __name__ == "__main__":
    sys.exit(main())
