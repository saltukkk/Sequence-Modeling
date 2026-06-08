"""Sequence models for Part 1 financial forecasting."""

from __future__ import annotations

import torch
import torch.nn as nn

from parameters import StockModelName, StockModelParams, TaskType


class StockLSTM(nn.Module):
    """Stacked LSTM encoder for multi-horizon return forecasting.

    The network consumes a lookback window of shape ``(batch, T, F)`` and
    predicts ``D`` future return ratios from the final hidden state.
    """

    def __init__(self, input_size: int, params: StockModelParams) -> None:
        """Initialize the LSTM forecaster.

        Args:
            input_size: Number of input features per time step.
            params: Model hyper-parameters such as hidden size and dropout.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            dropout=params.dropout if params.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(params.dropout)
        self.fc = nn.Linear(params.hidden_size, params.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, T, F)``.

        Returns:
            Predicted returns with shape ``(batch, D)``.
        """
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.fc(self.dropout(last_hidden))


class StockGRU(nn.Module):
    """Stacked GRU encoder for multi-horizon return forecasting."""

    def __init__(self, input_size: int, params: StockModelParams) -> None:
        """Initialize the GRU forecaster.

        Args:
            input_size: Number of input features per time step.
            params: Model hyper-parameters such as hidden size and dropout.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            dropout=params.dropout if params.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(params.dropout)
        self.fc = nn.Linear(params.hidden_size, params.output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, T, F)``.

        Returns:
            Predicted returns with shape ``(batch, D)``.
        """
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        return self.fc(self.dropout(last_hidden))


class StockBiLSTMClassifier(nn.Module):
    """Bidirectional LSTM classifier for buy/pass turning-point detection."""

    def __init__(self, input_size: int, params: StockModelParams) -> None:
        """Initialize the bidirectional LSTM classifier.

        Args:
            input_size: Number of input features per time step.
            params: Model hyper-parameters such as hidden size and dropout.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            dropout=params.dropout if params.num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(params.dropout)
        self.fc = nn.Linear(params.hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, T, F)``.

        Returns:
            Logits with shape ``(batch,)``.
        """
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.fc(self.dropout(last_hidden)).squeeze(-1)


class StockBiGRUClassifier(nn.Module):
    """Bidirectional GRU classifier for buy/pass turning-point detection."""

    def __init__(self, input_size: int, params: StockModelParams) -> None:
        """Initialize the bidirectional GRU classifier.

        Args:
            input_size: Number of input features per time step.
            params: Model hyper-parameters such as hidden size and dropout.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=params.hidden_size,
            num_layers=params.num_layers,
            batch_first=True,
            dropout=params.dropout if params.num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(params.dropout)
        self.fc = nn.Linear(params.hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input tensor of shape ``(batch, T, F)``.

        Returns:
            Logits with shape ``(batch,)``.
        """
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        return self.fc(self.dropout(last_hidden)).squeeze(-1)


def build_part1_model(
    model_name: StockModelName,
    input_size: int,
    params: StockModelParams,
    task: TaskType,
) -> nn.Module:
    """Instantiate a Part 1 model by name and task type.

    Args:
        model_name: One of ``lstm``, ``gru``, ``bilstm``, or ``bigru``.
        input_size: Number of input features per time step.
        params: Shared model hyper-parameters.
        task: ``regression`` for return forecasting or ``classification`` for turning points.

    Returns:
        An uninitialized PyTorch module.

    Raises:
        ValueError: If the model name does not match the requested task.
    """
    if task == "regression":
        if model_name == "lstm":
            return StockLSTM(input_size, params)
        if model_name == "gru":
            return StockGRU(input_size, params)
        raise ValueError(f"Model '{model_name}' is not a regression model.")

    if task == "classification":
        if model_name == "bilstm":
            return StockBiLSTMClassifier(input_size, params)
        if model_name == "bigru":
            return StockBiGRUClassifier(input_size, params)
        raise ValueError(f"Model '{model_name}' is not a classification model.")

    raise ValueError(f"Unknown task: {task}")
