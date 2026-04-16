from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VISOR_ROOT = REPO_ROOT / "external" / "VISOR"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score VISOR metrics from an object-detection results JSON."
    )
    parser.add_argument(
        "--visor-root",
        type=Path,
        default=DEFAULT_VISOR_ROOT,
        help="Path to the external VISOR checkout.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        required=True,
        help="Detection results JSON in VISOR's expected format.",
    )
    parser.add_argument(
        "--text-json",
        type=Path,
        default=None,
        help="VISOR text_spatial_rel_phrases.json file. Defaults to the one in the checkout.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional path for the VISOR summary JSON.",
    )
    return parser


def increment_dict(values: dict[int, list[float]], key: int, value: float) -> None:
    values.setdefault(key, []).append(value)


def get_visor_n(visor_by_uniq_id: dict[int, list[float]]) -> dict[str, float]:
    visor_1 = visor_2 = visor_3 = visor_4 = 0
    for scores in visor_by_uniq_id.values():
        score_sum = sum(scores)
        if score_sum >= 4:
            visor_4 += 1
        if score_sum >= 3:
            visor_3 += 1
        if score_sum >= 2:
            visor_2 += 1
        if score_sum >= 1:
            visor_1 += 1

    total = len(visor_by_uniq_id) or 1
    return {
        "VISOR_1": 100 * visor_1 / total,
        "VISOR_2": 100 * visor_2 / total,
        "VISOR_3": 100 * visor_3 / total,
        "VISOR_4": 100 * visor_4 / total,
    }


def score_visor(results: dict, text_data: list[dict]) -> dict:
    objacc_both = 0
    visor_cond_total = 0.0
    both_count = 0
    count = 0
    visor_by_uniq_id: dict[int, list[float]] = {}

    for image_id, record in results.items():
        uniq_id = int(image_id.split("_")[0])
        ann = text_data[uniq_id]
        relation = ann["rel_type"]
        num_objects = ann["num_objects"]
        if relation == "and" or num_objects != 2:
            continue

        obj1 = ann["obj_1_attributes"][0]
        obj2 = ann["obj_2_attributes"][0]
        detected = record["classes"]
        det_both = int(obj1 in detected and obj2 in detected)
        sra = float(record["sra"])

        objacc_both += det_both
        visor_cond_total += det_both * sra
        both_count += det_both
        count += 1
        increment_dict(visor_by_uniq_id, uniq_id, det_both * sra)

    oa = 100 * objacc_both / count if count else 0.0
    visor_cond = 100 * visor_cond_total / both_count if both_count else 0.0
    visor_uncond = visor_cond * oa / 100.0
    visor_n = get_visor_n(visor_by_uniq_id)

    return {
        "num_images": len(results),
        "num_evaluated_pairs": count,
        "OA": oa,
        "VISOR_cond": visor_cond,
        "VISOR_uncond": visor_uncond,
        **visor_n,
    }


def main() -> int:
    args = make_parser().parse_args()
    visor_root = args.visor_root.resolve()
    text_json = (
        args.text_json.resolve()
        if args.text_json is not None
        else (visor_root / "text_spatial_rel_phrases.json")
    )
    results_json = args.results_json.resolve()

    text_data = json.loads(text_json.read_text())
    results = json.loads(results_json.read_text())
    summary = score_visor(results, text_data)

    output_file = args.output_file
    if output_file is None:
        output_file = results_json.with_name(results_json.stem + "_visor_eval.json")
    payload = {
        "visor_root": str(visor_root),
        "text_json": str(text_json),
        "results_json": str(results_json),
        "summary": summary,
    }
    output_file.write_text(json.dumps(payload, indent=2))

    print(f"VISOR_uncond: {summary['VISOR_uncond']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
