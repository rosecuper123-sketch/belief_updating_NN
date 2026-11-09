from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from formal_osla.protocols import FORMAL_GROUPS  # noqa: E402
from formal_osla.test_runner import run_frozen_tests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen formal test protocols.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs_runs" / "frozen_tests")
    parser.add_argument("--protocol", choices=["human_default", "legacy_3plus1", "conflict_only"], default="human_default")
    parser.add_argument("--groups", nargs="+", default=list(FORMAL_GROUPS))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--custom-rounds-json",
        type=Path,
        default=None,
        help='JSON list like [["consistent", 0.7, 0.7], ["conflict", 0.3, 0.6]].',
    )
    args = parser.parse_args()

    custom = None
    protocol = args.protocol
    if args.custom_rounds_json is not None:
        custom = json.loads(args.custom_rounds_json.read_text(encoding="utf-8"))
        custom = [(str(k), float(c), float(t)) for k, c, t in custom]
        protocol = "custom"

    manifest = run_frozen_tests(
        args.checkpoint,
        out_root=args.out_root,
        protocol=protocol,
        groups=tuple(args.groups),
        episodes=args.episodes,
        seed=args.seed,
        deterministic=args.deterministic,
        custom_rounds=custom,
    )
    print(json.dumps({"entries": manifest}, indent=2))


if __name__ == "__main__":
    main()
