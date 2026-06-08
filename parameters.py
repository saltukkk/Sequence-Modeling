"""Configuration dataclasses and argparse helpers"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent

FeatureMode = Literal["returns", "rolling_avg", "turning_point"]
StockModelName = Literal["lstm", "gru", "bilstm", "bigru"]
Part1Experiment = Literal["all", "returns", "rolling_avg", "turning_point", "stability"]
TaskType = Literal["regression", "classification"]


@dataclass
class PathConfig:
    """Filesystem locations"""

    root: Path = field(default_factory=lambda: ROOT)
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    results_dir: Path = field(default_factory=lambda: ROOT / "results")

    def part_results_dir(self, part: int) -> Path:
        """Return the results directory for a homework part."""
        return self.results_dir / f"part{part}"


@dataclass
class StockDataParams:
    """Dataset and feature-engineering settings for Part 1."""

    tickers: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "GOOGL"])
    start_date: str = "2020-01-01"
    train_end: str = "2024-07-31"
    val_end: str = "2024-12-31"
    test_end: str = "2025-12-31"
    lookback: int = 20
    horizons: int = 5
    rolling_window: int = 3
    turning_point_gamma: float = 0.011


@dataclass
class StockModelParams:
    """hyper-parameters for Part 1 sequence models"""

    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    output_size: int = 5


@dataclass
class CommunicationDataParams:
    """Dataset settings for Part 2"""

    alphabet_size: int = 8
    message_length: int = 4
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    seed: int = 42

    @property
    def num_messages(self) -> int:
        """Total number of messages in the alphabet"""
        return self.alphabet_size ** self.message_length


@dataclass
class CommunicationModelParams:
    """Neural-network and channel settings for Part 2."""

    d_model: int = 64
    d_channel: int = 8
    num_heads: int = 4
    num_layers: int = 2
    ffn_dim: int = 128
    dropout: float = 0.1
    num_rounds: int = 4
    noise_variance: float = 0.25
    power_limit: float = 1.0

    @property
    def noise_std(self) -> float:
        """Standard deviation of AWGN noise"""
        return self.noise_variance ** 0.5


@dataclass
class TrainParams:
    """Shared optimizer and training-loop settings"""

    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42


@dataclass
class Part1TrainConfig:
    """Full configuration for a Part 1 training"""

    paths: PathConfig
    data: StockDataParams
    model: StockModelParams
    train: TrainParams
    experiment: Part1Experiment = "all"
    model_name: StockModelName = "lstm"
    feature_mode: FeatureMode = "returns"


@dataclass
class Part2TrainConfig:
    """Full configuration for a Part 2 training"""

    paths: PathConfig
    data: CommunicationDataParams
    model: CommunicationModelParams
    train: TrainParams = field(
        default_factory=lambda: TrainParams(
            batch_size=256,
            epochs=200,
            patience=15,
        )
    )


@dataclass
class Part1TestConfig:
    """Configuration for evaluating a trained Part 1 model"""

    paths: PathConfig
    data: StockDataParams
    model: StockModelParams
    train: TrainParams
    feature_mode: FeatureMode
    model_name: StockModelName
    checkpoint_path: Path | None = None
    experiment_name: str = ""


@dataclass
class Part2TestConfig:
    """Configuration for evaluating a trained Part 2 model"""

    paths: PathConfig
    data: CommunicationDataParams
    model: CommunicationModelParams
    train: TrainParams
    checkpoint_path: Path | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the top-level CLI parser with ``train`` and ``test`` sub-commands"""
    parser = argparse.ArgumentParser(
        description="Sequence Modeling (Part 1) and Communication Protocol (Part 2)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train models.")
    _add_train_arguments(train_parser)

    test_parser = subparsers.add_parser("test", help="Evaluate trained models.")
    _add_test_arguments(test_parser)

    return parser


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the training sub-command."""
    parser.add_argument("--part", type=int, choices=[1, 2], required=True, help="Homework part.")
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "returns", "rolling_avg", "turning_point"])
    parser.add_argument("--model", type=str, default="all", choices=["all", "lstm", "gru", "bilstm", "bigru"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=None)


def _add_test_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the test sub-command."""
    parser.add_argument("--part", type=int, choices=[1, 2], required=True, help="Homework part.")
    parser.add_argument("--experiment", type=str, default="returns", choices=["returns", "rolling_avg", "turning_point"])
    parser.add_argument("--model", type=str, required=True, choices=["lstm", "gru", "bilstm", "bigru", "communication"])
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a saved model checkpoint.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=None)


def _resolve_results_dir(args: argparse.Namespace) -> PathConfig:
    """Build path configuration from parsed CLI arguments."""
    paths = PathConfig()
    if getattr(args, "results_dir", None) is not None:
        paths.results_dir = args.results_dir
    return paths


def build_part1_train_config(args: argparse.Namespace) -> Part1TrainConfig:
    """Construct a Part 1 training configuration from CLI arguments."""
    paths = _resolve_results_dir(args)
    train = TrainParams(seed=args.seed)
    if args.batch_size is not None:
        train.batch_size = args.batch_size
    if args.epochs is not None:
        train.epochs = args.epochs
    if args.learning_rate is not None:
        train.learning_rate = args.learning_rate

    model_name: StockModelName = "lstm"
    if args.model != "all":
        model_name = args.model  

    return Part1TrainConfig(
        paths=paths,
        data=StockDataParams(),
        model=StockModelParams(),
        train=train,
        experiment=args.experiment, 
        model_name=model_name,
    )


def build_part2_train_config(args: argparse.Namespace) -> Part2TrainConfig:
    """Construct a Part 2 training configuration from CLI arguments."""
    paths = _resolve_results_dir(args)
    train = TrainParams(
        batch_size=256,
        epochs=200,
        patience=15,
        seed=args.seed,
    )
    if args.batch_size is not None:
        train.batch_size = args.batch_size
    if args.epochs is not None:
        train.epochs = args.epochs
    if args.learning_rate is not None:
        train.learning_rate = args.learning_rate

    return Part2TrainConfig(
        paths=paths,
        data=CommunicationDataParams(seed=args.seed),
        model=CommunicationModelParams(),
        train=train,
    )


def build_part1_test_config(args: argparse.Namespace) -> Part1TestConfig:
    """Construct a Part 1 evaluation configuration from CLI arguments."""
    paths = _resolve_results_dir(args)
    train = TrainParams(seed=args.seed)
    if args.batch_size is not None:
        train.batch_size = args.batch_size

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = paths.results_dir / f"part1_{args.experiment}_{args.model}.pt"

    return Part1TestConfig(
        paths=paths,
        data=StockDataParams(),
        model=StockModelParams(),
        train=train,
        feature_mode=args.experiment,  # type: ignore[arg-type]
        model_name=args.model,  # type: ignore[arg-type]
        checkpoint_path=checkpoint,
        experiment_name=f"part1_{args.experiment}_{args.model}",
    )


def build_part2_test_config(args: argparse.Namespace) -> Part2TestConfig:
    """Construct a Part 2 evaluation configuration from CLI arguments."""
    paths = _resolve_results_dir(args)
    train = TrainParams(batch_size=256, seed=args.seed)
    if args.batch_size is not None:
        train.batch_size = args.batch_size

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = paths.part_results_dir(2) / "best_model.pt"

    return Part2TestConfig(
        paths=paths,
        data=CommunicationDataParams(seed=args.seed),
        model=CommunicationModelParams(),
        train=train,
        checkpoint_path=checkpoint,
    )
