from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CATEGORIES = ["color", "shape", "texture", "spatial", "non_spatial", "complex"]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate per-category T2I-CompBench results into an overall score."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per T2I-CompBench category result.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Category names to include in the overall aggregate.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit path for the aggregate summary JSON.",
    )
    return parser


def result_file_name(category: str) -> str:
    return f"t2i_compbench_{category}_eval.json"


def main() -> int:
    args = make_parser().parse_args()
    root_dir = args.root_dir.resolve()
    category_scores: dict[str, float] = {}
    missing: list[str] = []

    for category in args.categories:
        summary_path = root_dir / category / result_file_name(category)
        if not summary_path.exists():
            missing.append(category)
            continue

        summary = json.loads(summary_path.read_text())
        average_score = summary.get("average_score")
        if average_score is None:
            missing.append(category)
            continue
        category_scores[category] = float(average_score)

    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing category summaries or average scores for: {missing_str}"
        )

    overall_score = sum(category_scores.values()) / len(category_scores)
    payload = {
        "root_dir": str(root_dir),
        "categories": args.categories,
        "category_scores": category_scores,
        "overall_score": overall_score,
    }

    output_file = args.output_file
    if output_file is None:
        output_file = root_dir / "t2i_compbench_overall_eval.json"
    output_file.write_text(json.dumps(payload, indent=2))

    print(f"T2I-CompBench overall score: {overall_score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
