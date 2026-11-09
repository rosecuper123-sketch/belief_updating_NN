from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from formal_osla.config import ExperimentConfig  # noqa: E402
from formal_osla.train import train_formal  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train formal one-shot LossA conditionA networks.")
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs_runs")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--entropy-bonus", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lambda-anchor", type=float, default=0.0)
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()

    if args.config_json is not None:
        config = ExperimentConfig.from_dict(json.loads(args.config_json.read_text(encoding="utf-8")))
    else:
        config = ExperimentConfig(output_root=args.output_root)
    config.training.seed = args.seed
    config.training.updates = args.updates
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.weight_decay = args.weight_decay
    config.training.entropy_bonus = args.entropy_bonus
    config.model.hidden_size = args.hidden_size
    config.model.dropout = args.dropout
    config.validation.enabled = not args.no_validation
    config.validation.patience = args.patience
    config.validation.lambda_anchor = args.lambda_anchor

    _model, history = train_formal(config)
    final = history[-1] if history else {}
    print(json.dumps({"updates_run": len(history), "final": final}, indent=2))


if __name__ == "__main__":
    main()
