"""Frozen tests for the pre-run grid, plus structural invariant checks.

Each (hidden_size, budget) arm is trained under its own output root by
`prerun_batch.py`; this script mirrors that layout for frozen tests, because
`run_frozen_tests` derives `network_id` from the checkpoint's parent directory
name only -- it does not encode hidden_size or the budget, so arms sharing an
out-root would collide.

After collecting, every output directory is checked against the structural
invariants the plan requires (boundary/trial separation, per-round resets, array
shapes).  Checks are printed and returned; nothing is silently tolerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Pin threads before numpy/torch are imported (see prerun_batch.py).
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from formal_osla.protocols import FORMAL_GROUPS  # noqa: E402
from formal_osla.test_runner import run_frozen_tests  # noqa: E402

PROTOCOL_ROUNDS = {"human_default": 2, "legacy_3plus1": 4, "conflict_only": 1}
TRIALS_PER_ROUND = 120


def require_single_thread() -> None:
    n = int(torch.get_num_threads())
    if n != 1:
        raise RuntimeError(f"torch.get_num_threads() == {n}, expected 1")


def discover_arms(output_root: Path) -> list[tuple[str, Path]]:
    """Return (arm_name, arm_dir) for every arm that has checkpoints."""
    arms_root = output_root / "arms"
    if not arms_root.is_dir():
        raise FileNotFoundError(f"no arms directory under {output_root}")
    found: list[tuple[str, Path]] = []
    for arm in sorted(arms_root.iterdir()):
        if arm.is_dir() and (arm / "runs").is_dir():
            found.append((arm.name, arm))
    return found


def record_provenance(
    out_root: Path, *, arm_name: str, network_id: str, protocol: str, checkpoint: Path
) -> dict:
    """Capture which checkpoint this call actually loaded, before the next call overwrites it.

    `run_frozen_tests` writes a single `out_root/manifest.json` on every invocation, and this
    runner invokes it once per (network, protocol) -- so with 30 networks x 3 protocols only the
    last one survives.  The data are unaffected; what is lost is the ledger of which checkpoint
    produced which outputs.  Reading it back here and aggregating at the end preserves it.

    `checkpoint_sha256` is added because the manifest records only the *path*: without a hash we
    can say which file was named, but not that the bytes on disk are still the ones that ran.
    """
    recorded = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    return {
        "arm": arm_name,
        "network_id": network_id,
        "protocol": protocol,
        "checkpoint": recorded.get("checkpoint"),
        "checkpoint_best_update": recorded.get("checkpoint_best_update"),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "test_seed": recorded.get("seed"),
        "groups": [entry["group"] for entry in recorded.get("entries", [])],
    }


def check_outputs(out_dir: Path, *, episodes: int, hidden_size: int, rounds: int) -> dict:
    """Structural invariant checks for one frozen-test output directory."""
    checks: dict[str, object] = {}
    trials = pd.read_csv(out_dir / "trials.csv")
    boundary = pd.read_csv(out_dir / "boundary_events.csv")

    expected_trials = episodes * rounds * TRIALS_PER_ROUND
    checks["trials_rows"] = (len(trials) == expected_trials, f"{len(trials)} vs {expected_trials}")
    checks["no_boundary_col"] = ("boundary_flag" not in trials.columns, "boundary_flag column absent")
    checks["boundary_rows"] = (len(boundary) == episodes * rounds, f"{len(boundary)} vs {episodes * rounds}")

    # Every round's first behavioral trial must start from a clean slate.
    first = trials[trials["trial_id"] == 1]
    checks["first_trials_present"] = (len(first) == episodes * rounds, f"{len(first)} vs {episodes * rounds}")
    clean = (
        (first["prev_action"] == 0).all()
        and (first["prev_outcome"] == 0).all()
        and (first["prev_token_delta"] == 0).all()
        and (first["current_total_tokens"] == 100).all()
    )
    checks["first_trial_reset"] = (bool(clean), "prev_*==0 and tokens==100")

    with np.load(out_dir / "hidden.npz") as hidden:
        hidden_shape = hidden["hidden"].shape
    with np.load(out_dir / "boundary_hidden.npz") as boundary_hidden:
        boundary_shape = boundary_hidden["h_pre_boundary"].shape
    checks["hidden_shape"] = (
        tuple(hidden_shape) == (episodes, rounds * TRIALS_PER_ROUND, hidden_size),
        f"{tuple(hidden_shape)} vs {(episodes, rounds * TRIALS_PER_ROUND, hidden_size)}",
    )
    checks["boundary_hidden_shape"] = (
        tuple(boundary_shape) == (episodes, rounds, hidden_size),
        f"{tuple(boundary_shape)} vs {(episodes, rounds, hidden_size)}",
    )

    checks["pass"] = all(bool(v[0]) for v in checks.values() if isinstance(v, tuple))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect frozen tests for the pre-run grid.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs_prerun")
    parser.add_argument("--arms", nargs="+", default=None, help="arm names, e.g. h32_b200")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--protocols", nargs="+", default=["human_default", "legacy_3plus1", "conflict_only"])
    parser.add_argument("--groups", nargs="+", default=list(FORMAL_GROUPS))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="only run invariant checks")
    args = parser.parse_args()

    require_single_thread()
    output_root = args.output_root.resolve()
    arms = discover_arms(output_root)
    if args.arms:
        wanted = set(args.arms)
        arms = [(name, path) for name, path in arms if name in wanted]
    if not arms:
        raise SystemExit("no arms selected")

    for protocol in args.protocols:
        if protocol not in PROTOCOL_ROUNDS:
            raise SystemExit(f"unknown protocol {protocol!r}")

    results: list[dict] = []
    provenance: list[dict] = []
    tested: set[tuple[str, str]] = set()
    for arm_name, arm_dir in arms:
        hidden_size = int(arm_name.split("_")[0].lstrip("h"))
        checkpoint_dirs = sorted((arm_dir / "runs").glob("formal_osla_seed*"))
        for checkpoint_dir in checkpoint_dirs:
            seed = int(checkpoint_dir.name.replace("formal_osla_seed", ""))
            if args.seeds and seed not in args.seeds:
                continue
            checkpoint = checkpoint_dir / "best.pt"
            if not checkpoint.exists():
                print(f"SKIP {arm_name}/{checkpoint_dir.name}: no best.pt", flush=True)
                continue
            out_root = output_root / "frozen_tests" / arm_name
            for protocol in args.protocols:
                target = out_root / protocol / checkpoint_dir.name / args.groups[0]
                if args.skip_existing and target.exists():
                    print(f"SKIP {arm_name}/{protocol}/{checkpoint_dir.name}: exists", flush=True)
                elif not args.check_only:
                    print(f"TEST {arm_name}/{protocol}/{checkpoint_dir.name} ...", flush=True)
                    run_frozen_tests(
                        checkpoint,
                        out_root=out_root,
                        protocol=protocol,
                        groups=tuple(args.groups),
                        episodes=args.episodes,
                        seed=args.seed,
                        deterministic=args.deterministic,
                    )
                    provenance.append(
                        record_provenance(
                            out_root,
                            arm_name=arm_name,
                            network_id=checkpoint_dir.name,
                            protocol=protocol,
                            checkpoint=checkpoint,
                        )
                    )
                    tested.add((arm_name, checkpoint_dir.name))
                for group in args.groups:
                    out_dir = out_root / protocol / checkpoint_dir.name / group
                    checks = check_outputs(
                        out_dir,
                        episodes=args.episodes,
                        hidden_size=hidden_size,
                        rounds=PROTOCOL_ROUNDS[protocol],
                    )
                    failed = [k for k, v in checks.items() if isinstance(v, tuple) and not v[0]]
                    results.append(
                        {
                            "arm": arm_name,
                            "seed": seed,
                            "protocol": protocol,
                            "group": group,
                            "hidden_size": hidden_size,
                            "pass": checks["pass"],
                            "failed": failed,
                            "details": {k: v[1] for k, v in checks.items() if isinstance(v, tuple)},
                        }
                    )
                    status = "ok" if checks["pass"] else f"FAIL {failed}"
                    print(f"  {arm_name}/{protocol}/{checkpoint_dir.name}/{group}: {status}", flush=True)

    out = {
        "output_root": str(output_root),
        "episodes": args.episodes,
        "test_seed": args.seed,
        "deterministic": args.deterministic,
        "torch_num_threads": int(torch.get_num_threads()),
        "n_checks": len(results),
        "n_passed": sum(1 for r in results if r["pass"]),
        "entries": results,
    }
    checks_path = output_root / "frozen_test_checks.json"
    checks_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nchecks: {out['n_passed']}/{out['n_checks']} passed -> {checks_path}")
    if out["n_passed"] != out["n_checks"]:
        raise SystemExit("structural invariant checks failed")

    # --- checkpoint provenance ledger (see record_provenance) -----------------
    if provenance:
        for arm_name, _ in sorted(tested):
            arm_entries = [p for p in provenance if p["arm"] == arm_name]
            arm_dirs = {p["network_id"] for p in arm_entries}
            arm_protocols = {p["protocol"] for p in arm_entries}
            expected = len(arm_dirs) * len(arm_protocols)
            if len(arm_entries) != expected:
                raise SystemExit(
                    f"provenance incomplete for {arm_name}: {len(arm_entries)} records, "
                    f"expected {len(arm_dirs)} networks x {len(arm_protocols)} protocols = {expected}"
                )
            (output_root / "frozen_tests" / arm_name / "manifest_provenance.json").write_text(
                json.dumps({"entries": arm_entries}, indent=2), encoding="utf-8"
            )
        distinct = {}
        for record in provenance:
            distinct.setdefault(record["checkpoint_sha256"], record)
        print(
            f"provenance: {len(provenance)} records over {len(tested)} networks and "
            f"{len(distinct)} distinct checkpoints -> frozen_tests/<arm>/manifest_provenance.json"
        )
        for sha, record in sorted(distinct.items()):
            print(f"  {sha[:12]}  best_update={record['checkpoint_best_update']}  {record['checkpoint']}")


if __name__ == "__main__":
    main()
