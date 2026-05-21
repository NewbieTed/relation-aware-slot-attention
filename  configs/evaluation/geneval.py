from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GENEVAL_ROOT = REPO_ROOT / "external" / "geneval"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GenEval on an existing generated sample directory."
    )
    parser.add_argument(
        "--geneval-root",
        type=Path,
        default=DEFAULT_GENEVAL_ROOT,
        help="Path to the external GenEval checkout.",
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        required=True,
        help="Generated output directory containing samples/ and run_config.json.",
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=None,
        help="GenEval metadata JSONL file. Defaults to prompts/evaluation_metadata.jsonl inside the checkout.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Directory containing the downloaded GenEval detector checkpoint(s).",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python executable to use for GenEval.",
    )
    parser.add_argument(
        "--copy-instead-of-symlink",
        action="store_true",
        help="Copy images into the prepared GenEval folder instead of symlinking.",
    )
    return parser


def resolve_geneval_root(path: Path) -> Path:
    root = path.resolve()
    if not (root / "evaluation" / "evaluate_images.py").exists():
        raise FileNotFoundError(f"Could not locate a GenEval checkout under {root}")
    return root


def load_run_config(generated_dir: Path) -> dict:
    run_config_path = generated_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run_config.json in {generated_dir}")
    return json.loads(run_config_path.read_text())


def sorted_sample_paths(samples_dir: Path) -> list[Path]:
    sample_paths = [path for path in samples_dir.iterdir() if path.is_file() and not path.name.startswith(".")]
    sample_paths.sort(key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    return sample_paths


def prepare_geneval_layout(
    generated_dir: Path,
    metadata_file: Path,
    *,
    copy_instead_of_symlink: bool,
) -> Path:
    run_config = load_run_config(generated_dir)
    num_images_per_prompt = int(run_config["num_images_per_prompt"])
    samples_dir = generated_dir / "samples"
    sample_paths = sorted_sample_paths(samples_dir)
    if len(sample_paths) % num_images_per_prompt != 0:
        raise ValueError(
            f"Sample count {len(sample_paths)} is not divisible by num_images_per_prompt={num_images_per_prompt}"
        )

    prompt_count = len(sample_paths) // num_images_per_prompt
    metadata_lines = [line for line in metadata_file.read_text().splitlines() if line.strip()]
    if prompt_count > len(metadata_lines):
        raise ValueError(
            f"Need {prompt_count} metadata entries but only found {len(metadata_lines)} in {metadata_file}"
        )

    prepared_dir = generated_dir / "_geneval_eval"
    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    for prompt_index in range(prompt_count):
        prompt_dir = prepared_dir / f"{prompt_index:05d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "metadata.jsonl").write_text(metadata_lines[prompt_index] + "\n")

        prompt_samples_dir = prompt_dir / "samples"
        prompt_samples_dir.mkdir(parents=True, exist_ok=True)
        group = sample_paths[
            prompt_index * num_images_per_prompt : (prompt_index + 1) * num_images_per_prompt
        ]
        for image_index, sample_path in enumerate(group):
            target = prompt_samples_dir / f"{image_index:04d}{sample_path.suffix}"
            if copy_instead_of_symlink:
                shutil.copy2(sample_path, target)
            else:
                os.symlink(sample_path.resolve(), target)

    return prepared_dir


def summarize_results(results_path: Path) -> dict:
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    total_images = len(rows)
    by_prompt: dict[str, list[bool]] = {}
    by_tag: dict[str, list[bool]] = {}

    for row in rows:
        by_prompt.setdefault(row["metadata"], []).append(bool(row["correct"]))
        by_tag.setdefault(row["tag"], []).append(bool(row["correct"]))

    task_scores = {tag: sum(values) / len(values) for tag, values in by_tag.items()}
    return {
        "total_images": total_images,
        "total_prompts": len(by_prompt),
        "correct_image_rate": sum(bool(row["correct"]) for row in rows) / total_images if total_images else None,
        "correct_prompt_rate": (
            sum(any(values) for values in by_prompt.values()) / len(by_prompt) if by_prompt else None
        ),
        "task_scores": task_scores,
        "overall_score": (
            sum(task_scores.values()) / len(task_scores) if task_scores else None
        ),
    }


def main() -> int:
    args = make_parser().parse_args()
    geneval_root = resolve_geneval_root(args.geneval_root)
    generated_dir = args.generated_dir.resolve()
    metadata_file = (
        args.metadata_file.resolve()
        if args.metadata_file is not None
        else (geneval_root / "prompts" / "evaluation_metadata.jsonl")
    )
    model_path = (
        args.model_path.resolve()
        if args.model_path is not None
        else (geneval_root / "models")
    )

    prepared_dir = prepare_geneval_layout(
        generated_dir,
        metadata_file,
        copy_instead_of_symlink=args.copy_instead_of_symlink,
    )
    results_path = generated_dir / "geneval_results.jsonl"

    completed = subprocess.run(
        [
            args.python_bin,
            "evaluation/evaluate_images.py",
            str(prepared_dir),
            "--outfile",
            str(results_path),
            "--model-path",
            str(model_path),
        ],
        cwd=geneval_root,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        payload = {
            "generated_dir": str(generated_dir),
            "geneval_root": str(geneval_root),
            "metadata_file": str(metadata_file),
            "model_path": str(model_path),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        (generated_dir / "geneval_eval_error.json").write_text(json.dumps(payload, indent=2))
        raise RuntimeError(
            "GenEval failed. See geneval_eval_error.json for the captured stdout/stderr."
        )

    summary = summarize_results(results_path)
    payload = {
        "generated_dir": str(generated_dir),
        "geneval_root": str(geneval_root),
        "metadata_file": str(metadata_file),
        "model_path": str(model_path),
        "results_file": str(results_path),
        "summary": summary,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    (generated_dir / "geneval_eval.json").write_text(json.dumps(payload, indent=2))
    overall = summary.get("overall_score")
    if overall is not None:
        print(f"GenEval overall score: {overall:.6f}")
    else:
        print("GenEval evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
