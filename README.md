# relation-aware-slot-attention

This workspace currently contains `SCOP-Depth`, a depth-augmented variant of the SCOP data pipeline used to extract 2D, depth-order, and occlusion-style relations from COCO.

The implementation lives in [scop_depth](/Users/newbieted/workspace/relation-aware-slot-attention/scop_depth), and the module path is `scop_depth`.

For evaluation, this repo now keeps only our wrapper scripts and a lightweight
override layer. External benchmark code such as T2I-CompBench should be prepared
outside the tracked source tree with:

```bash
./scripts/setup/bootstrap_all.sh
```

This creates a local `.venv`, installs the repo and evaluation dependencies,
prepares the external T2I-CompBench checkout, and downloads the benchmark
weights.

If you previously used an older bootstrap version that created
`scripts/.venv`, remove that stale environment first.

## Training Baseline

The repo now also contains a first training scaffold for a vanilla Stable
Diffusion 1.5 LoRA baseline on SCOP-Depth.

Key pieces:

- [`training/dataset.py`](/Users/newbieted/workspace/relation-aware-slot-attention/training/dataset.py):
  loads `metadata.jsonl`, images, prompts, and serialized scene-graph metadata
- [`training/prompts.py`](/Users/newbieted/workspace/relation-aware-slot-attention/training/prompts.py):
  builds concise baseline prompts from SCOP-Depth relations
- [`training/train_sd15_lora.py`](/Users/newbieted/workspace/relation-aware-slot-attention/training/train_sd15_lora.py):
  trains SD1.5 LoRA attention processors on the exported dataset
- [`scripts/train/run_sd15_lora_baseline.sh`](/Users/newbieted/workspace/relation-aware-slot-attention/scripts/train/run_sd15_lora_baseline.sh):
  thin shell wrapper for launching the baseline trainer

Example:

```bash
./.venv/bin/python -m pip install -e ".[train]"

DATASET_DIR=/path/to/scop_depth_full \
OUTPUT_DIR=outputs/train/sd15_scopdepth_lora \
DEVICE=cuda \
MAX_TRAIN_STEPS=1000 \
./scripts/train/run_sd15_lora_baseline.sh
```

This baseline intentionally keeps the model text-only at conditioning time while
still carrying scene-graph metadata through the dataloader so the next stage can
replace or augment text conditioning with graph-aware slots.

New SCOP-Depth exports create CoMPaSS-style pair crops by default, so each
training row points to a crop around the selected object pair. If you explicitly
used `--no-crop-pairs` and `metadata.jsonl` exists but
`data/scop_depth_full/images/` is missing because the original COCO tree was
moved or deleted, you can rebuild just the referenced full-image subset without
rerunning SCOP-Depth:

```bash
./.venv/bin/python -m training.materialize_images \
  --dataset-dir /path/to/scop_depth_full \
  --coco-root /path/to/coco2017 \
  --mode symlink
```
