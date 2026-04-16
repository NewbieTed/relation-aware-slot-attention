from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the image subset referenced by a SCOP-Depth metadata.jsonl file."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--coco-root",
        type=Path,
        required=True,
        help="COCO root that contains train2017/ and annotations/",
    )
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Whether to symlink or copy the needed images into dataset_dir/images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing broken links or files in dataset_dir/images.",
    )
    return parser.parse_args()


def _load_needed_filenames(metadata_path: Path) -> list[str]:
    filenames: list[str] = []
    seen: set[str] = set()
    for line in metadata_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        filename = Path(row["file_name"]).name
        if filename not in seen:
            seen.add(filename)
            filenames.append(filename)
    return filenames


def _remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        raise IsADirectoryError(f"Refusing to remove directory path: {path}")


def main() -> int:
    args = parse_args()
    metadata_path = args.dataset_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    train2017_dir = args.coco_root / "train2017"
    if not train2017_dir.exists():
        raise FileNotFoundError(f"Missing COCO image directory: {train2017_dir}")

    images_dir = args.dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    filenames = _load_needed_filenames(metadata_path)
    if not filenames:
        raise ValueError(f"No image references found in {metadata_path}")

    created = 0
    skipped = 0
    for filename in tqdm(filenames, desc="Materializing SCOP-Depth images"):
        source = train2017_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Missing COCO source image: {source}")

        target = images_dir / filename
        if target.exists() or target.is_symlink():
            if args.force:
                _remove_existing(target)
            else:
                skipped += 1
                continue

        if args.mode == "symlink":
            rel_source = os.path.relpath(source, images_dir)
            target.symlink_to(rel_source)
        else:
            target.write_bytes(source.read_bytes())
        created += 1

    print(
        f"Prepared {created} images in {images_dir} "
        f"(skipped {skipped} existing files, mode={args.mode})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
