# SCOP-Depth Data Engine

This directory contains the implementation of `SCOP-Depth`, a depth-augmented extension of the SCOP data engine.

The Python module path is `scop_depth`, so commands use `python3 -m scop_depth ...`.

## Usage

### Obtaining Images

Go to the repository root (i.e. parent directory of this directory) and run:

```bash
# make sure CWD is the repository root
python3 -m scop_depth --coco-root /path/to/coco2017 --output-dir ./scop-coco2017
```

The `--coco-root` directory should contain `annotations/` and `train2017/`.

On successful processing of COCO2017, the output directory (`scop-coco2017` in this example) should
have a `metadata.jsonl` file (28,028 lines) and an `images/` folder (15,426 images).

### Optional: Depth Anything V2 Enrichment

This branch also supports optional per-pair depth enrichment with the Hugging Face
`depth-anything/Depth-Anything-V2-Base-hf` checkpoint.

```bash
python3 -m scop_depth \
  --coco-root /path/to/coco2017 \
  --output-dir ./scop-coco2017-depth \
  --limit-images 50 \
  --create-samples \
  --use-depth-anything \
  --depth-device auto
```

When depth is enabled, each exported relationship entry may include a `depth` field in
`metadata.jsonl` with bbox-level statistics and a conservative depth ordering. Sample
visualizations also become side-by-side RGB and depth previews.

If you additionally pass `--include-depth-order-labels`, the pipeline appends
`in front of` / `behind` to `oros` only when the normalized median-depth gap is above
`--depth-min-separation` (default: `0.2`). Depth summaries are computed from the center crop of each
bounding box, controlled by `--depth-center-crop-ratio` (default: `0.6`), to reduce
background contamination. Overlapping pairs can also emit `hidden by` when the overlap
ratio exceeds `--hidden-overlap-threshold` (default: `0.4`) and the depth ordering is
confident; in that case the export includes both `behind` and `hidden by`. The default
behavior stays unchanged unless depth is enabled.

Backend behavior:

- `--depth-device auto` prefers `cuda`, then `mps`, then `cpu`.
- `--depth-device mps` explicitly targets Apple Silicon GPU when available.
- On Apple Silicon, the code enables `PYTORCH_ENABLE_MPS_FALLBACK=1` so unsupported
  ops can fall back to CPU while keeping the rest of the model on `mps`.
- If MPS still cannot execute the model reliably, the depth module falls back to full
  CPU inference instead of crashing the run.
- `--depth-device cuda` is supported for Linux/NVIDIA environments and falls back to
  CPU if CUDA is unavailable.

### Obtaining Object Masks (Optional for FLUX.1, required for Stable Diffusion 1.4/1.5/2.1)

Install [SAM2] per the [official instructions][SAM2] (if you ran the [../setup_env.sh](../setup_env.sh)
script to set up your environment, SAM2 should have already been installed), then run:

```bash
# make sure CWD is the repository root
python3 -m scop_depth.process_masks ./scop-coco2017
```

[SAM2]: <https://github.com/facebookresearch/sam2>

<!-- vim: set ts=2 sts=2 sw=2 et: -->
