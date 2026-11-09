from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    input_size: int = 7
    hidden_size: int = 32
    dropout: float = 0.0


@dataclass(slots=True)
class EnvironmentConfig:
    round_feedback_trials: int = 120
    block_size: int = 20
    train_rounds_per_episode: int = 3
    probability_grid: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)
    start_tokens: int = 100


@dataclass(slots=True)
class TrainingConfig:
    seed: int = 20260928
    updates: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    gradient_clip: float = 1.0
    entropy_bonus: float = 0.0
    weight_decay: float = 0.0
    log_interval: int = 10


@dataclass(slots=True)
class ValidationConfig:
    enabled: bool = True
    seed: int = 20270928
    batch_size: int = 128
    rounds_per_episode: int = 3
    interval: int = 10
    patience: int = 20
    min_delta: float = 0.0
    lambda_anchor: float = 0.0


@dataclass(slots=True)
class ExperimentConfig:
    output_root: Path = Path("outputs")
    model: ModelConfig = field(default_factory=ModelConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            output_root=Path(data.get("output_root", "outputs")),
            model=ModelConfig(**data.get("model", {})),
            environment=EnvironmentConfig(**data.get("environment", {})),
            training=TrainingConfig(**data.get("training", {})),
            validation=ValidationConfig(**data.get("validation", {})),
        )
