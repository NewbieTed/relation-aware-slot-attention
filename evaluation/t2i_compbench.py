from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_ROOT = REPO_ROOT / "evaluation" / "vendor" / "t2i_compbench_spatial" / "unidet"

BENCHMARK_SPECS = {
    "spatial": {
        "script": "2D_spatial_eval.py",
        "result_file": "t2i_compbench_spatial_eval.json",
        "error_file": "t2i_compbench_spatial_eval_error.json",
        "label_output": "../examples/labels/annotation_obj_detection_2d/vqa_result.json",
    },
    "3d_spatial": {
        "script": "3D_spatial_eval.py",
        "result_file": "t2i_compbench_3d_spatial_eval.json",
        "error_file": "t2i_compbench_3d_spatial_eval_error.json",
        "label_output": "../examples/labels/annotation_obj_detection_3d/vqa_result.json",
    },
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run vendored T2I-CompBench spatial evaluation on an existing generated samples directory."
    )
    parser.add_argument(
        "--t2i-compbench-root",
        type=Path,
        default=DEFAULT_VENDOR_ROOT,
        help=(
            "Path to the vendored T2I-CompBench UniDet_eval root. "
            "Defaults to evaluation/vendor/t2i_compbench_spatial/unidet."
        ),
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="Our generated output directory containing samples/ and run_config.json.",
    )
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARK_SPECS.keys()),
        default="spatial",
        help="Benchmark subset to score (default: spatial).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Prompt file used to generate the samples.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable to use for the evaluator.",
    )
    parser.add_argument(
        "--copy-instead-of-symlink",
        action="store_true",
        help="Copy generated samples into the vendored examples/samples instead of symlinking.",
    )
    return parser


def prepare_examples_dir(
    benchmark_root: Path,
    generated_samples_dir: Path,
    prompt_file: Path,
    *,
    copy_instead_of_symlink: bool,
) -> None:
    examples_dir = benchmark_root.parent / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    target_prompt_file = examples_dir / prompt_file.name
    shutil.copy2(prompt_file, target_prompt_file)

    target_samples_dir = examples_dir / "samples"
    if target_samples_dir.exists() or target_samples_dir.is_symlink():
        if target_samples_dir.is_symlink() or target_samples_dir.is_file():
            target_samples_dir.unlink()
        else:
            shutil.rmtree(target_samples_dir)

    if copy_instead_of_symlink:
        shutil.copytree(generated_samples_dir, target_samples_dir)
    else:
        os.symlink(generated_samples_dir.resolve(), target_samples_dir)


def read_label_output(benchmark_root: Path, relative_path: str) -> list[dict] | None:
    label_file = benchmark_root / relative_path
    if not label_file.exists():
        return None
    try:
        return json.loads(label_file.read_text())
    except json.JSONDecodeError:
        return None


def run_benchmark(
    benchmark_root: Path,
    benchmark: str,
    python_bin: str,
) -> subprocess.CompletedProcess[str]:
    spec = BENCHMARK_SPECS[benchmark]
    eval_script = benchmark_root / spec["script"]
    if not eval_script.exists():
        raise FileNotFoundError(f"Missing vendored evaluator script: {eval_script}")

    return subprocess.run(
        [python_bin, eval_script.name],
        cwd=benchmark_root,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> int:
    args = make_parser().parse_args()
    benchmark_root = args.t2i_compbench_root.resolve()
    generated_dir = args.generated_dir.resolve()
    prompt_file = args.prompt_file.resolve()
    generated_samples_dir = generated_dir / "samples"
    if not generated_samples_dir.exists():
        raise FileNotFoundError(f"Missing generated samples dir: {generated_samples_dir}")

    prepare_examples_dir(
        benchmark_root,
        generated_samples_dir,
        prompt_file,
        copy_instead_of_symlink=args.copy_instead_of_symlink,
    )

    completed = run_benchmark(benchmark_root, args.benchmark, args.python_bin)
    spec = BENCHMARK_SPECS[args.benchmark]
    label_results = read_label_output(benchmark_root, spec["label_output"])

    if completed.returncode != 0:
        failure_summary = {
            "benchmark": args.benchmark,
            "generated_dir": str(generated_dir),
            "prompt_file": str(prompt_file),
            "benchmark_root": str(benchmark_root),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        (generated_dir / spec["error_file"]).write_text(json.dumps(failure_summary, indent=2))
        raise RuntimeError(
            f"T2I-CompBench {args.benchmark} evaluation failed. "
            f"See {spec['error_file']} for the captured stdout/stderr."
        )

    score_summary = {
        "benchmark": args.benchmark,
        "generated_dir": str(generated_dir),
        "prompt_file": str(prompt_file),
        "benchmark_root": str(benchmark_root),
        "label_output": spec["label_output"],
        "label_results_count": len(label_results) if label_results is not None else None,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    (generated_dir / spec["result_file"]).write_text(json.dumps(score_summary, indent=2))

    print(f"T2I-CompBench {args.benchmark} evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
