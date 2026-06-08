"""Generate LaTeX report figures from saved experiment JSON files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "report" / "figures"


def plot_part1_loss(path: Path, output_dir: Path) -> None:
    """Save a train/validation loss curve for one Part 1 experiment."""
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data.get("history", {})
    if not history:
        return

    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(data["experiment"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{data['experiment']}_loss.png", dpi=150)
    plt.close()


def plot_part2_curves(summary_path: Path, output_dir: Path) -> None:
    """Save Part 2 loss and message-accuracy curves."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    history = data["history"]

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
    plt.savefig(output_dir / "part2_training_curves.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in sorted(RESULTS_DIR.glob("part1*.json")):
        if path.name == "part1_summary.json":
            continue
        plot_part1_loss(path, OUTPUT_DIR)

    plot_part2_curves(RESULTS_DIR / "part2" / "summary.json", OUTPUT_DIR)
    print(f"Figures saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
