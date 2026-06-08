from __future__ import annotations

import argparse
import json

from parameters import (
    build_arg_parser,
    build_part1_test_config,
    build_part1_train_config,
    build_part2_test_config,
    build_part2_train_config,
)
from test import test_part1, test_part2
from train import train_part1, train_part2


def print_header(title: str) -> None:
    """Print a formatted section header

    Args:
        title: Section title shown to the user.
    """
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def run_train(args: argparse.Namespace) -> None:
    """Execute the training sub-command

    Args:
        args: Parsed command-line arguments.
    """
    if args.part == 1:
        config = build_part1_train_config(args)
        print_header("Training Part 1")
        summaries = train_part1(config)
        print_header("Part 1 training finished")
        print(json.dumps(summaries, indent=2))
        print(f"Results saved to: {config.paths.part_results_dir(1).resolve()}")
        return

    config = build_part2_train_config(args)
    print_header("Training Part 2")
    summary = train_part2(config)
    print_header("Part 2 training finished")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to: {config.paths.part_results_dir(2).resolve()}")


def run_test(args: argparse.Namespace) -> None:
    """Execute the test sub-command

    Args:
        args: Parsed command-line arguments.
    """
    if args.part == 1:
        config = build_part1_test_config(args)
        print_header(f"Testing Part 1: {config.experiment_name}")
        metrics = test_part1(config)
        print_header("Part 1 test finished")
        print(json.dumps(metrics, indent=2))
        return

    config = build_part2_test_config(args)
    print_header("Testing Part 2")
    metrics = test_part2(config)
    print_header("Part 2 test finished")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "test":
        run_test(args)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
