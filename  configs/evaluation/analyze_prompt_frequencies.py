from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from training.dataset import load_metadata_rows
from training.prompts import prompt_from_scop_depth_row


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count prompt frequencies in a SCOP-style metadata folder."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--prompt-prefix", type=str, default="a photo of")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main() -> int:
    args = make_parser().parse_args()
    rows = load_metadata_rows(args.dataset_dir)
    prompt_counts = Counter(
        prompt_from_scop_depth_row(row, prefix=args.prompt_prefix) for row in rows
    )
    frequency_counts = Counter(prompt_counts.values())
    singleton_prompts = sum(1 for count in prompt_counts.values() if count == 1)
    singleton_rows = singleton_prompts
    augmentable_prompts = sum(1 for count in prompt_counts.values() if count >= 2)
    augmentable_rows = sum(count for count in prompt_counts.values() if count >= 2)

    print(f"Dataset: {args.dataset_dir}")
    print(f"Total rows: {len(rows)}")
    print(f"Unique prompts: {len(prompt_counts)}")
    print(f"Prompts with one occurrence: {singleton_prompts}")
    print(f"Rows belonging to one-occurrence prompts: {singleton_rows}")
    print(f"Prompts with at least two occurrences: {augmentable_prompts}")
    print(f"Rows belonging to prompts with at least two occurrences: {augmentable_rows}")
    print("")
    print("Prompt frequency histogram:")
    for frequency, prompt_count in sorted(frequency_counts.items()):
        print(f"  {frequency}: {prompt_count} prompts")

    if args.top_k > 0:
        print("")
        print(f"Top {args.top_k} prompts:")
        for prompt, count in prompt_counts.most_common(args.top_k):
            print(f"  {count:5d}  {prompt}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["prompt", "count", "augmentable"])
            for prompt, count in sorted(prompt_counts.items(), key=lambda item: (-item[1], item[0])):
                writer.writerow([prompt, count, int(count >= 2)])
        print("")
        print(f"Wrote prompt counts to {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
