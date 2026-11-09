"""Pre-run trainer for the formal one-shot LossA conditionA networks.

Grid: seeds x hidden_size x update budget.  Each (hidden_size, budget) arm gets
its own output root, because `formal_osla.train.train_formal` names its run
directory `runs/formal_osla_seed<seed>` and `run_frozen_tests` derives
`network_id` from the checkpoint's parent directory name -- neither encodes
hidden_size or the budget, so two arms sharing an output root would silently
overwrite each other.

Thread count is pinned to 1 *before* numpy/torch are imported, following the
hard constraint recorded in `formal_batch/hidden_size_sweep/scripts/hsspec.py`
(2026-09-20): multi-threaded matmul changes the floating-point reduction order
and the same seed diverges from the single-thread reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Must run before numpy/torch are imported anywhere.
# --------------------------------------------------------------------------- #
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from formal_osla.config import ExperimentConfig  # noqa: E402
from formal_osla.train import train_formal  # noqa: E402

DEFAULT_SEEDS = (20260928, 20260929, 20260930)
DEFAULT_HIDDEN_SIZES = (16, 32)
DEFAULT_BUDGETS = (200, 600)


def require_single_thread() -> None:
    n = int(torch.get_num_threads())
    if n != 1:
        raise RuntimeError(
            f"torch.get_num_threads() == {n}, expected 1. Pin OMP_NUM_THREADS "
            f"before importing numpy/torch or the results will not reproduce."
        )


def code_fingerprint() -> dict:
    """sha256 of every formal_osla source file, so a run can be tied to code."""
    out: dict[str, str] = {}
    for path in sorted((ROOT / "formal_osla").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        key = str(path.relative_to(ROOT))
        out[key] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return out


def arm_dir(output_root: Path, hidden_size: int, budget: int) -> Path:
    return output_root / "arms" / f"h{hidden_size:02d}_b{budget}"


def run_cell(
    *,
    output_root: Path,
    seed: int,
    hidden_size: int,
    budget: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    skip_existing: bool,
) -> dict:
    arm = arm_dir(output_root, hidden_size, budget)
    run_dir = arm / "runs" / f"formal_osla_seed{seed}"
    metadata_path = run_dir / "run_metadata.json"

    if skip_existing and metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "seed": seed,
            "hidden_size": hidden_size,
            "budget": budget,
            "arm": str(arm),
            "run_dir": str(run_dir),
            "status": "skipped_existing",
            "best_update": meta.get("best_update"),
            "best_score": meta.get("best_score"),
        }

    config = ExperimentConfig(output_root=arm)
    config.training.seed = seed
    config.training.updates = budget
    config.training.batch_size = batch_size
    config.training.learning_rate = learning_rate
    config.model.hidden_size = hidden_size
    config.validation.patience = patience

    started = time.perf_counter()
    _model, history = train_formal(config)
    elapsed = time.perf_counter() - started

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "seed": seed,
        "hidden_size": hidden_size,
        "budget": budget,
        "arm": str(arm),
        "run_dir": str(run_dir),
        "status": "trained",
        "updates_recorded": len(history),
        "best_update": meta.get("best_update"),
        "best_score": meta.get("best_score"),
        "early_stopping_enabled": meta.get("early_stopping_enabled"),
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pre-run network grid.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs_prerun")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=list(DEFAULT_HIDDEN_SIZES))
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    require_single_thread()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cells = [
        (seed, hidden, budget)
        for hidden in args.hidden_sizes
        for budget in args.budgets
        for seed in args.seeds
    ]
    print(
        f"grid: {len(cells)} runs | seeds={list(args.seeds)} "
        f"hidden={list(args.hidden_sizes)} budgets={list(args.budgets)}"
    )
    print(f"output_root: {output_root}")
    print(f"torch threads: {torch.get_num_threads()}  torch {torch.__version__}")

    results: list[dict] = []
    for index, (seed, hidden, budget) in enumerate(cells, start=1):
        label = f"[{index}/{len(cells)}] seed={seed} hidden={hidden:>2} budget={budget:>3}"
        print(f"{label} ...", flush=True)
        record = run_cell(
            output_root=output_root,
            seed=seed,
            hidden_size=hidden,
            budget=budget,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            skip_existing=args.skip_existing,
        )
        results.append(record)
        print(
            f"{label} {record['status']} | best_update={record['best_update']} "
            f"best_score={record['best_score']} | {record.get('elapsed_seconds', '-')}s",
            flush=True,
        )

    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_fingerprint": code_fingerprint(),
        "torch_version": torch.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "numpy_version": np.__version__,
        "grid": {
            "seeds": list(args.seeds),
            "hidden_sizes": list(args.hidden_sizes),
            "budgets": list(args.budgets),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
        },
        "entries": results,
    }
    manifest_path = output_root / "prerun_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
