"""Training routines"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.messages import make_message_loaders, split_message_sizes
from datasets.stock import (
    build_stock_sequences,
    download_stock_data,
    make_stock_loaders,
)
from models.communication_models import CommunicationSystem
from models.stock_models import build_part1_model
from parameters import (
    FeatureMode,
    Part1TrainConfig,
    Part2TrainConfig,
    StockModelName,
    TaskType,
    TrainParams,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed shared across NumPy and PyTorch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device() -> torch.device:
    """Select CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _regression_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute one Part 1 regression mini-batch."""
    features, targets = batch
    features = features.to(device)
    targets = targets.to(device)
    predictions = model(features)
    loss = criterion(predictions, targets)
    return loss, predictions, targets


def _classification_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute one Part 1 classification mini-batch."""
    features, targets = batch
    features = features.to(device)
    targets = targets.to(device).squeeze(-1)
    logits = model(features)
    loss = criterion(logits, targets)
    return loss, logits, targets


def _run_stock_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    step_fn: Callable[
        [nn.Module, tuple[torch.Tensor, torch.Tensor], nn.Module, torch.device],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> float:
    """Run one epoch for a Part 1 model."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        if is_train:
            optimizer.zero_grad()

        loss, _, targets = step_fn(model, batch, criterion, device)
        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(total_samples, 1)


def _plot_stock_loss(history: dict[str, list[float]], title: str, path: Path) -> None:
    """Save a train/validation loss curve for Part 1."""
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train_stock_experiment(
    config: Part1TrainConfig,
    feature_mode: FeatureMode,
    model_name: StockModelName,
    experiment_name: str,
) -> dict[str, object]:
    """Train one Part 1 model on a specific feature mode.

    Args:
        config: Part 1 training configuration bundle.
        feature_mode: Target type for the experiment.
        model_name: Model architecture to train.
        experiment_name: Name used for saved artifacts.

    Returns:
        Dictionary with metrics, history, and checkpoint path.
    """
    set_seed(config.train.seed)
    device = _resolve_device()
    results_dir = config.paths.part_results_dir(1)
    results_dir.mkdir(parents=True, exist_ok=True)

    stock_data = download_stock_data(config.data, config.paths.data_dir)
    datasets = build_stock_sequences(stock_data, config.data, mode=feature_mode)
    loaders = make_stock_loaders(datasets, config.train)
    input_size = datasets["train"]["X"].shape[-1]

    task: TaskType = "classification" if feature_mode == "turning_point" else "regression"
    model = build_part1_model(model_name, input_size, config.model, task).to(device)

    if task == "classification":
        criterion: nn.Module = nn.BCEWithLogitsLoss()
        step_fn = _classification_step
    else:
        criterion = nn.MSELoss()
        step_fn = _regression_step

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    stale_epochs = 0
    history = {"train_loss": [], "val_loss": []}

    epoch_bar = tqdm(range(1, config.train.epochs + 1), desc=experiment_name, unit="epoch")
    for epoch in epoch_bar:
        train_loss = _run_stock_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            step_fn,
            optimizer,
            desc=f"{experiment_name} train",
        )
        val_loss = _run_stock_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            step_fn,
            None,
            desc=f"{experiment_name} val",
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        epoch_bar.set_postfix(train_loss=f"{train_loss:.6f}", val_loss=f"{val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.train.patience:
            tqdm.write(f"[{experiment_name}] Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    checkpoint_path = results_dir / f"{experiment_name}.pt"
    torch.save(best_state, checkpoint_path)
    _plot_stock_loss(history, experiment_name, results_dir / f"{experiment_name}_loss.png")

    from test import evaluate_stock_model

    test_metrics = evaluate_stock_model(
        model=model,
        loaders=loaders,
        task=task,
        device=device,
        desc=f"{experiment_name} test",
    )

    summary: dict[str, object] = {
        "experiment": experiment_name,
        "feature_mode": feature_mode,
        "model_name": model_name,
        "task": task,
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "history": history,
        "epochs_trained": len(history["train_loss"]),
        "train_loss_std": float(np.std(history["train_loss"][-5:])),
        "val_loss_std": float(np.std(history["val_loss"][-5:])),
        "checkpoint_path": str(checkpoint_path),
    }

    with open(results_dir / f"{experiment_name}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def train_part1(config: Part1TrainConfig) -> list[dict[str, object]]:
    """Run the requested Part 1 training experiments.

    Args:
        config: Part 1 training configuration bundle.

    Returns:
        A list of per-experiment summary dictionaries.
    """
    summaries: list[dict[str, object]] = []
    experiments: list[tuple[FeatureMode, str, list[StockModelName]]] = []

    if config.experiment in ("all", "returns"):
        experiments.append(("returns", "part1b", ["lstm", "gru"]))
    if config.experiment in ("all", "rolling_avg"):
        experiments.append(("rolling_avg", "part1c", ["lstm", "gru"]))
    if config.experiment in ("all", "turning_point"):
        experiments.append(("turning_point", "part1d", ["bilstm", "bigru"]))

    for feature_mode, prefix, model_names in experiments:
        selected_models = model_names
        if config.model_name in model_names:
            selected_models = [config.model_name]

        for model_name in selected_models:
            experiment_name = f"{prefix}_{model_name}"
            summary = train_stock_experiment(config, feature_mode, model_name, experiment_name)
            summaries.append(summary)
            tqdm.write(
                f"[{experiment_name}] test metrics: {summary['test_metrics']}"
            )

    if config.experiment == "all" and len(summaries) >= 4:
        _print_stability_comparison(summaries)

    summary_path = config.paths.part_results_dir(1) / "part1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)

    return summaries


def _print_stability_comparison(summaries: list[dict[str, object]]) -> None:
    """Print the Part 1(c) stability comparison across exact and rolling targets."""
    for model_name in ("lstm", "gru"):
        exact = next(item for item in summaries if item["experiment"] == f"part1b_{model_name}")
        rolling = next(item for item in summaries if item["experiment"] == f"part1c_{model_name}")
        exact_std = float(exact["train_loss_std"])
        rolling_std = float(rolling["train_loss_std"])
        more_stable = "exact" if exact_std < rolling_std else "rolling"
        tqdm.write(
            f"Stability {model_name.upper()}: exact_std={exact_std:.6f}, "
            f"rolling_std={rolling_std:.6f}, more stable={more_stable}"
        )


def _communication_batch(
    model: CommunicationSystem,
    messages: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """Execute one Part 2 mini-batch."""
    messages = messages.to(device)
    logits, _ = model(messages)

    loss = 0.0
    for position in range(messages.size(1)):
        loss = loss + criterion(logits[:, position, :], messages[:, position])
    loss = loss / messages.size(1)

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        symbol_acc = model.symbol_accuracy(logits, messages)
        message_acc = model.message_accuracy(logits, messages)

    return {
        "loss": loss.item(),
        "symbol_acc": symbol_acc,
        "message_acc": message_acc,
    }


def _run_communication_epoch(
    model: CommunicationSystem,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> dict[str, float]:
    """Run one epoch for the Part 2 communication system."""
    is_train = optimizer is not None
    model.train(is_train)

    totals = {"loss": 0.0, "symbol_acc": 0.0, "message_acc": 0.0}
    total_samples = 0
    progress = tqdm(loader, desc=desc, leave=False)

    for messages in progress:
        metrics = _communication_batch(model, messages, criterion, device, optimizer)
        batch_size = messages.size(0)
        for key in totals:
            totals[key] += metrics[key] * batch_size
        total_samples += batch_size
        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            msg_acc=f"{metrics['message_acc']:.3f}",
        )

    return {key: totals[key] / max(total_samples, 1) for key in totals}


def _plot_communication_history(history: dict[str, list[float]], path: Path) -> None:
    """Save Part 2 loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_message_acc"], label="Train")
    axes[1].plot(history["val_message_acc"], label="Validation")
    axes[1].set_title("Message Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train_part2(config: Part2TrainConfig) -> dict[str, object]:
    """Train the Part 2 communication protocol model.

    Args:
        config: Part 2 training configuration bundle.

    Returns:
        Summary dictionary with metrics, history, and checkpoint path.
    """
    set_seed(config.train.seed)
    device = _resolve_device()
    results_dir = config.paths.part_results_dir(2)
    results_dir.mkdir(parents=True, exist_ok=True)

    sizes = split_message_sizes(config.data)
    tqdm.write(
        f"Part 2 dataset sizes: train={sizes['train']}, val={sizes['val']}, test={sizes['test']}"
    )

    loaders = make_message_loaders(config.data, config.train)
    model = CommunicationSystem(
        config.model,
        alphabet_size=config.data.alphabet_size,
        message_length=config.data.message_length,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    best_state = deepcopy(model.state_dict())
    best_val_message_acc = -1.0
    stale_epochs = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_symbol_acc": [],
        "val_symbol_acc": [],
        "train_message_acc": [],
        "val_message_acc": [],
    }

    epoch_bar = tqdm(range(1, config.train.epochs + 1), desc="Part 2", unit="epoch")
    for epoch in epoch_bar:
        train_metrics = _run_communication_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer,
            desc=f"Part 2 train {epoch:03d}",
        )
        val_metrics = _run_communication_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            None,
            desc=f"Part 2 val {epoch:03d}",
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_symbol_acc"].append(train_metrics["symbol_acc"])
        history["val_symbol_acc"].append(val_metrics["symbol_acc"])
        history["train_message_acc"].append(train_metrics["message_acc"])
        history["val_message_acc"].append(val_metrics["message_acc"])

        epoch_bar.set_postfix(
            train_loss=f"{train_metrics['loss']:.4f}",
            val_msg_acc=f"{val_metrics['message_acc']:.3f}",
        )

        if val_metrics["message_acc"] > best_val_message_acc:
            best_val_message_acc = val_metrics["message_acc"]
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.train.patience:
            tqdm.write(f"Part 2 early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    checkpoint_path = results_dir / "best_model.pt"
    torch.save(best_state, checkpoint_path)
    _plot_communication_history(history, results_dir / "training_curves.png")

    from test import evaluate_communication_model

    test_metrics = evaluate_communication_model(
        model=model,
        loaders=loaders,
        device=device,
        desc="Part 2 test",
    )

    summary: dict[str, object] = {
        "device": str(device),
        "epochs_trained": len(history["train_loss"]),
        "best_val_message_acc": best_val_message_acc,
        "test_metrics": test_metrics,
        "history": history,
        "model_params": sum(param.numel() for param in model.parameters()),
        "checkpoint_path": str(checkpoint_path),
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary
