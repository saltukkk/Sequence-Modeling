"""Message enumeration utilities for Part 2."""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from parameters import CommunicationDataParams, TrainParams


def enumerate_messages(data_params: CommunicationDataParams) -> np.ndarray:
    """Enumerate all messages in the communication alphabet.

    Args:
        data_params: Alphabet and message-length configuration.

    Returns:
        Integer array of shape ``(num_messages, message_length)`` with zero-based symbols.
    """
    symbols = range(data_params.alphabet_size)
    messages = list(itertools.product(symbols, repeat=data_params.message_length))
    return np.asarray(messages, dtype=np.int64)


def split_messages(
    data_params: CommunicationDataParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split all messages into train, validation, and test subsets.

    Args:
        data_params: Split ratios and random seed.

    Returns:
        Train, validation, and test message arrays.
    """
    messages = enumerate_messages(data_params)
    rng = np.random.default_rng(data_params.seed)
    perm = rng.permutation(len(messages))
    messages = messages[perm]

    train_end = int(len(messages) * data_params.train_ratio)
    val_end = train_end + int(len(messages) * data_params.val_ratio)
    return messages[:train_end], messages[train_end:val_end], messages[val_end:]


class MessageDataset(Dataset):
    """PyTorch dataset for Part 2 integer messages."""

    def __init__(self, messages: np.ndarray) -> None:
        """Store the message array.

        Args:
            messages: Integer array of shape ``(N, message_length)``.
        """
        self.messages = torch.from_numpy(messages)

    def __len__(self) -> int:
        """Return the number of messages."""
        return len(self.messages)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return one message."""
        return self.messages[idx]


def make_message_loaders(
    data_params: CommunicationDataParams,
    train_params: TrainParams,
) -> dict[str, DataLoader]:
    """Create DataLoader objects for Part 2 messages.

    Args:
        data_params: Message-space configuration.
        train_params: Batch-size configuration.

    Returns:
        Data loaders for ``train``, ``val``, and ``test``.
    """
    train, val, test = split_messages(data_params)
    loaders: dict[str, DataLoader] = {}
    for name, split in (("train", train), ("val", val), ("test", test)):
        dataset = MessageDataset(split)
        loaders[name] = DataLoader(
            dataset,
            batch_size=train_params.batch_size,
            shuffle=name == "train",
            drop_last=False,
        )
    return loaders


def split_message_sizes(data_params: CommunicationDataParams) -> dict[str, int]:
    """Return the number of messages in each split.

    Args:
        data_params: Message-space configuration.

    Returns:
        Dictionary with ``train``, ``val``, and ``test`` sample counts.
    """
    train, val, test = split_messages(data_params)
    return {"train": len(train), "val": len(val), "test": len(test)}
