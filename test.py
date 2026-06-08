"""Evaluation routines"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.messages import make_message_loaders
from datasets.stock import build_stock_sequences, download_stock_data, make_stock_loaders
from models.communication_models import CommunicationSystem
from models.stock_models import build_part1_model
from parameters import (
    FeatureMode,
    Part1TestConfig,
    Part2TestConfig,
    TaskType,
)


def _resolve_device() -> torch.device:
    """Select CUDA when available"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        predictions: Model predictions.
        targets: Ground-truth targets.

    Returns:
        Dictionary with MSE and MAE.
    """
    mse = float(np.mean((predictions - targets) ** 2))
    mae = float(np.mean(np.abs(predictions - targets)))
    return {"mse": mse, "mae": mae}


def classification_metrics(logits: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute binary classification metrics.

    Args:
        logits: Raw classifier logits.
        targets: Ground-truth binary labels.

    Returns:
        Dictionary with accuracy, precision, recall, and F1 score.
    """
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(np.float32)
    accuracy = float(np.mean(predictions == targets))
    true_positive = float(np.sum((predictions == 1) & (targets == 1)))
    false_positive = float(np.sum((predictions == 1) & (targets == 0)))
    false_negative = float(np.sum((predictions == 0) & (targets == 1)))
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "buy_rate": float(np.mean(predictions)),
        "positive_rate": float(np.mean(targets)),
    }


@torch.no_grad()
def evaluate_stock_model(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    task: TaskType,
    device: torch.device,
    desc: str,
) -> dict[str, float]:
    """Evaluate Part 1 model 

    Args:
        model: Trained Part 1 network
        loaders: Dictionary that must contain a ``test`` loader
        task: ``regression`` or ``classification``.
        device: Target device
        desc: Description 

    Returns:
        Dictionary with loss and task-specific metrics
    """
    model.eval()
    loader = loaders["test"]

    if task == "classification":
        criterion: nn.Module = nn.BCEWithLogitsLoss()
        all_logits: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
    else:
        criterion = nn.MSELoss()
        all_predictions: list[np.ndarray] = []
        all_targets = []

    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for features, targets in progress:
        features = features.to(device)
        targets = targets.to(device)

        if task == "classification":
            target_values = targets.squeeze(-1)
            logits = model(features)
            loss = criterion(logits, target_values)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(target_values.cpu().numpy())
        else:
            predictions = model(features)
            loss = criterion(predictions, targets)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.6f}")

    if task == "classification":
        metrics = classification_metrics(
            np.concatenate(all_logits),
            np.concatenate(all_targets),
        )
    else:
        metrics = regression_metrics(
            np.concatenate(all_predictions),
            np.concatenate(all_targets),
        )

    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def evaluate_communication_model(
    model: CommunicationSystem,
    loaders: dict[str, DataLoader],
    device: torch.device,
    desc: str,
) -> dict[str, float]:
    """Evaluate Part 2 model.

    Args:
        model: Trained communication system.
        loaders: Dictionary that must contain a ``test`` loader.
        device: Target device.
        desc: Description 

    Returns:
        Dictionary with loss, symbol accuracy, and message accuracy.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = loaders["test"]

    totals = {"loss": 0.0, "symbol_acc": 0.0, "message_acc": 0.0}
    total_samples = 0
    progress = tqdm(loader, desc=desc, leave=False)

    for messages in progress:
        messages = messages.to(device)
        logits, _ = model(messages)

        loss = 0.0
        for position in range(messages.size(1)):
            loss = loss + criterion(logits[:, position, :], messages[:, position])
        loss = loss / messages.size(1)

        batch_size = messages.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["symbol_acc"] += model.symbol_accuracy(logits, messages) * batch_size
        totals["message_acc"] += model.message_accuracy(logits, messages) * batch_size
        total_samples += batch_size
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            msg_acc=f"{model.message_accuracy(logits, messages):.3f}",
        )

    return {key: totals[key] / max(total_samples, 1) for key in totals}


def _load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model weights from disk.

    Args:
        model: Model instance to populate.
        checkpoint_path: Saved state-dict path.
        device: Target device.

    Returns:
        The model moved to ``device`` with loaded weights.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)


def test_part1(config: Part1TestConfig) -> dict[str, float]:
    """Evaluate a saved Part 1 checkpoint

    Args:
        config: Part 1 evaluation configuration bundle.

    Returns:
        Test-set metrics.
    """
    if config.checkpoint_path is None or not config.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")

    device = _resolve_device()
    stock_data = download_stock_data(config.data, config.paths.data_dir)
    datasets = build_stock_sequences(stock_data, config.data, mode=config.feature_mode)
    loaders = make_stock_loaders(datasets, config.train)
    input_size = datasets["train"]["X"].shape[-1]

    task: TaskType = "classification" if config.feature_mode == "turning_point" else "regression"
    model = build_part1_model(config.model_name, input_size, config.model, task)
    model = _load_checkpoint(model, config.checkpoint_path, device)

    return evaluate_stock_model(
        model=model,
        loaders=loaders,
        task=task,
        device=device,
        desc=config.experiment_name or "Part 1 test",
    )


def test_part2(config: Part2TestConfig) -> dict[str, float]:
    """Evaluate a saved Part 2 checkpoint 

    Args:
        config: Part 2 evaluation configuration bundle.

    Returns:
        Test-set metrics.
    """
    if config.checkpoint_path is None or not config.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")

    device = _resolve_device()
    loaders = make_message_loaders(config.data, config.train)
    model = CommunicationSystem(
        config.model,
        alphabet_size=config.data.alphabet_size,
        message_length=config.data.message_length,
    )
    model = _load_checkpoint(model, config.checkpoint_path, device)

    return evaluate_communication_model(
        model=model,
        loaders=loaders,
        device=device,
        desc="Part 2 test",
    )
