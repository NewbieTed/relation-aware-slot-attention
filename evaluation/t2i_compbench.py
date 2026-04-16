from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_T2I_ROOT = REPO_ROOT / "external" / "T2I-CompBench"

BENCHMARK_SPECS = {
    "color": {
        "result_file": "t2i_compbench_color_eval.json",
        "error_file": "t2i_compbench_color_eval_error.json",
        "label_output": "examples/annotation_blip/vqa_result.json",
        "included_in_overall": True,
    },
    "shape": {
        "result_file": "t2i_compbench_shape_eval.json",
        "error_file": "t2i_compbench_shape_eval_error.json",
        "label_output": "examples/annotation_blip/vqa_result.json",
        "included_in_overall": True,
    },
    "texture": {
        "result_file": "t2i_compbench_texture_eval.json",
        "error_file": "t2i_compbench_texture_eval_error.json",
        "label_output": "examples/annotation_blip/vqa_result.json",
        "included_in_overall": True,
    },
    "spatial": {
        "result_file": "t2i_compbench_spatial_eval.json",
        "error_file": "t2i_compbench_spatial_eval_error.json",
        "label_output": "examples/labels/annotation_obj_detection_2d/vqa_result.json",
        "included_in_overall": True,
    },
    "3d_spatial": {
        "result_file": "t2i_compbench_3d_spatial_eval.json",
        "error_file": "t2i_compbench_3d_spatial_eval_error.json",
        "label_output": "examples/labels/annotation_obj_detection_3d/vqa_result.json",
        "included_in_overall": False,
    },
    "numeracy": {
        "result_file": "t2i_compbench_numeracy_eval.json",
        "error_file": "t2i_compbench_numeracy_eval_error.json",
        "label_output": "examples/annotation_num/vqa_result.json",
        "included_in_overall": False,
    },
    "non_spatial": {
        "result_file": "t2i_compbench_non_spatial_eval.json",
        "error_file": "t2i_compbench_non_spatial_eval_error.json",
        "label_output": "examples/annotation_clip/vqa_result.json",
        "included_in_overall": True,
    },
    "complex": {
        "result_file": "t2i_compbench_complex_eval.json",
        "error_file": "t2i_compbench_complex_eval_error.json",
        "label_output": "examples/annotation_3_in_1/vqa_result.json",
        "included_in_overall": True,
    },
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run T2I-CompBench evaluation on an existing generated samples directory."
    )
    parser.add_argument(
        "--t2i-compbench-root",
        type=Path,
        default=DEFAULT_T2I_ROOT,
        help="Path to a T2I-CompBench checkout. Defaults to external/T2I-CompBench.",
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
        help="T2I-CompBench subset to score.",
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
        help="Copy generated samples into examples/samples instead of symlinking.",
    )
    return parser


def resolve_t2i_root(t2i_compbench_root: Path) -> Path:
    root = t2i_compbench_root.resolve()
    if (root / "Readme.md").exists():
        return root
    if (root / "UniDet_eval").exists():
        return root
    raise FileNotFoundError(f"Could not locate a T2I-CompBench checkout under {root}")


def prepare_examples_dir(
    t2i_root: Path,
    generated_samples_dir: Path,
    prompt_file: Path,
    *,
    copy_instead_of_symlink: bool,
) -> None:
    examples_dir = t2i_root / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    copied_prompt_file = examples_dir / prompt_file.name
    shutil.copy2(prompt_file, copied_prompt_file)

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


def read_label_output(t2i_root: Path, relative_path: str) -> list[dict] | None:
    label_file = t2i_root / relative_path
    if not label_file.exists():
        return None
    try:
        return json.loads(label_file.read_text())
    except json.JSONDecodeError:
        return None


def compute_average_score(label_results: list[dict] | None) -> float | None:
    if not label_results:
        return None
    values: list[float] = []
    for record in label_results:
        answer = record.get("answer")
        try:
            values.append(float(answer))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def run_script(
    cwd: Path,
    python_bin: str,
    script: str,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [python_bin, script, *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def run_blip_vqa(t2i_root: Path, python_bin: str) -> list[subprocess.CompletedProcess[str]]:
    return [
        run_script(
            t2i_root / "BLIPvqa_eval",
            python_bin,
            "BLIP_vqa.py",
            "--out_dir",
            "../examples/",
        )
    ]


def run_spatial(
    t2i_root: Path,
    python_bin: str,
    *,
    complex_mode: bool,
) -> list[subprocess.CompletedProcess[str]]:
    args: list[str] = []
    if complex_mode:
        args.extend(["--complex", "True"])
    return [run_script(t2i_root / "UniDet_eval", python_bin, "2D_spatial_eval.py", *args)]


def run_3d_spatial(t2i_root: Path, python_bin: str) -> list[subprocess.CompletedProcess[str]]:
    return [run_script(t2i_root / "UniDet_eval", python_bin, "3D_spatial_eval.py")]


def run_numeracy(t2i_root: Path, python_bin: str) -> list[subprocess.CompletedProcess[str]]:
    return [run_script(t2i_root / "UniDet_eval", python_bin, "numeracy_eval.py")]


def run_clipscore(
    t2i_root: Path,
    python_bin: str,
    *,
    complex_mode: bool,
) -> list[subprocess.CompletedProcess[str]]:
    args = ["--outpath", "examples/"]
    if complex_mode:
        args.extend(["--complex", "True"])
    return [run_script(t2i_root, python_bin, "CLIPScore_eval/CLIP_similarity.py", *args)]


def run_complex(t2i_root: Path, python_bin: str) -> list[subprocess.CompletedProcess[str]]:
    completions: list[subprocess.CompletedProcess[str]] = []
    completions.extend(run_blip_vqa(t2i_root, python_bin))
    completions.extend(run_spatial(t2i_root, python_bin, complex_mode=True))
    completions.extend(run_clipscore(t2i_root, python_bin, complex_mode=True))
    completions.append(
        run_script(
            t2i_root / "3_in_1_eval",
            python_bin,
            "3_in_1.py",
            "--outpath",
            "../examples/",
            "--data_path",
            "../examples/dataset",
        )
    )
    return completions


def run_benchmark(
    t2i_root: Path,
    benchmark: str,
    python_bin: str,
) -> list[subprocess.CompletedProcess[str]]:
    if benchmark in {"color", "shape", "texture"}:
        return run_blip_vqa(t2i_root, python_bin)
    if benchmark == "spatial":
        return run_spatial(t2i_root, python_bin, complex_mode=False)
    if benchmark == "3d_spatial":
        return run_3d_spatial(t2i_root, python_bin)
    if benchmark == "numeracy":
        return run_numeracy(t2i_root, python_bin)
    if benchmark == "non_spatial":
        return run_clipscore(t2i_root, python_bin, complex_mode=False)
    if benchmark == "complex":
        return run_complex(t2i_root, python_bin)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def summarize_completed_processes(
    completions: list[subprocess.CompletedProcess[str]],
) -> tuple[int, str, str]:
    returncode = 0
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    for index, completed in enumerate(completions, start=1):
        stdout_parts.append(f"[step {index} stdout]\n{completed.stdout}")
        stderr_parts.append(f"[step {index} stderr]\n{completed.stderr}")
        if completed.returncode != 0 and returncode == 0:
            returncode = completed.returncode
    return returncode, "\n".join(stdout_parts), "\n".join(stderr_parts)


def main() -> int:
    args = make_parser().parse_args()
    t2i_root = resolve_t2i_root(args.t2i_compbench_root)
    generated_dir = args.generated_dir.resolve()
    prompt_file = args.prompt_file.resolve()
    generated_samples_dir = generated_dir / "samples"
    if not generated_samples_dir.exists():
        raise FileNotFoundError(f"Missing generated samples dir: {generated_samples_dir}")

    prepare_examples_dir(
        t2i_root,
        generated_samples_dir,
        prompt_file,
        copy_instead_of_symlink=args.copy_instead_of_symlink,
    )

    completions = run_benchmark(t2i_root, args.benchmark, args.python_bin)
    spec = BENCHMARK_SPECS[args.benchmark]
    label_results = read_label_output(t2i_root, spec["label_output"])
    average_score = compute_average_score(label_results)
    returncode, stdout, stderr = summarize_completed_processes(completions)

    if returncode != 0:
        failure_summary = {
            "benchmark": args.benchmark,
            "generated_dir": str(generated_dir),
            "prompt_file": str(prompt_file),
            "t2i_compbench_root": str(t2i_root),
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
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
        "t2i_compbench_root": str(t2i_root),
        "label_output": spec["label_output"],
        "label_results_count": len(label_results) if label_results is not None else None,
        "average_score": average_score,
        "stdout": stdout,
        "stderr": stderr,
    }
    (generated_dir / spec["result_file"]).write_text(json.dumps(score_summary, indent=2))

    if average_score is not None:
        print(f"T2I-CompBench {args.benchmark} average score: {average_score:.6f}")
    else:
        print(f"T2I-CompBench {args.benchmark} evaluation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
