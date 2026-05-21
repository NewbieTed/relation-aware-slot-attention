from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluation.t2i_compbench import BENCHMARK_SPECS


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated T2I-CompBench runs into mean/std statistics."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Directory containing repeated run subdirectories such as run_000, run_001, ...",
    )
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARK_SPECS.keys()),
        required=True,
        help="Benchmark category to aggregate.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="run_",
        help="Prefix used for repeated run subdirectories (default: run_).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional explicit output JSON path.",
    )
    return parser


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def main() -> int:
    args = make_parser().parse_args()
    root_dir = args.root_dir.resolve()
    result_file = BENCHMARK_SPECS[args.benchmark]["result_file"]

    runs: list[dict[str, object]] = []
    for run_dir in sorted(path for path in root_dir.iterdir() if path.is_dir() and path.name.startswith(args.run_prefix)):
        summary_path = run_dir / result_file
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        average_score = summary.get("average_score")
        if average_score is None:
            continue

        run_config_path = run_dir / "run_config.json"
        run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
        runs.append(
            {
                "run_dir": str(run_dir),
                "average_score": float(average_score),
                "seed": run_config.get("seed"),
                "num_images_per_prompt": run_config.get("num_images_per_prompt"),
            }
        )

    if not runs:
        raise FileNotFoundError(
            f"No completed {args.benchmark} run summaries found under {root_dir} with prefix {args.run_prefix!r}"
        )

    scores = [float(run["average_score"]) for run in runs]
    mean_score = _mean(scores)
    std_score = _std(scores, mean_score)

    payload = {
        "root_dir": str(root_dir),
        "benchmark": args.benchmark,
        "num_runs": len(runs),
        "mean_score": mean_score,
        "std_score": std_score,
        "min_score": min(scores),
        "max_score": max(scores),
        "runs": runs,
    }

    output_file = args.output_file
    if output_file is None:
        output_file = root_dir / f"t2i_compbench_{args.benchmark}_repeats_summary.json"
    output_file.write_text(json.dumps(payload, indent=2))

    print(
        f"T2I-CompBench {args.benchmark} repeats: mean={mean_score:.6f}, std={std_score:.6f}, runs={len(runs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
