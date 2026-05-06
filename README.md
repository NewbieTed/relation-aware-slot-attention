# relation-aware-slot-attention

This branch is the FLUX.1-dev version of the project. It keeps the SCOP-Depth
data pipeline and relation-aware GNN, then uses the GNN to predict 3D object
centers and box sizes. Those predictions are rendered into SeeThrough3D-style
OSCR condition images, VAE-encoded as FLUX condition latents, and consumed by
rank-128 LoRA processors in FLUX self-attention.

## Setup

Run the full bootstrap:

```bash
./scripts/setup/bootstrap_all.sh
```

This creates `.venv`, installs the repo with FLUX/evaluation dependencies, clones
SeeThrough3D into `external/seethrough3d`, and prepares optional benchmark
checkouts. The FLUX training and generation code directly imports
SeeThrough3D's FLUX transformer fork and custom LoRA attention processors.

If you only need the SeeThrough3D checkout:

```bash
./scripts/setup/setup_seethrough3d_checkout.sh
```

## Current Workflow

1. Build SCOP-Depth cropped/depth data.
2. Pretrain the GNN with position, relation, and 3D box-size losses.
3. Freeze the GNN and FLUX base model.
4. Render GNN-predicted 3D boxes into OSCR condition images.
5. Train FLUX.1-dev condition-stream LoRA processors.
6. Generate/debug images with `evaluation.generate_flux_relation`.

Example GNN training:

```bash
python3 -m training.pretrain_graph_encoder \
  --dataset-dir data/scop_depth_crops_depth \
  --output-dir outputs/train/graph_pretrain_flux_3dbox \
  --position-loss-weight 1.0 \
  --relation-loss-weight 8.0 \
  --box3d-loss-weight 1.0 \
  --embedding-loss-weight 0.0 \
  --inverse-relation-loss-weight 0.0
```

Example FLUX LoRA training:

```bash
python3 -m training.train_relation_flux_lora \
  --dataset-dir data/scop_depth_crops_depth \
  --output-dir outputs/train/flux1dev_oscr_lora128 \
  --init-graph-encoder outputs/train/graph_pretrain_flux_3dbox/final/graph_encoder.pt \
  --model-id black-forest-labs/FLUX.1-dev \
  --lora-rank 128 \
  --lora-alpha 128
```

Example generation/debug:

```bash
python3 -m evaluation.generate_flux_relation \
  --prompt "a cat to the left of a dog" \
  --checkpoint-dir outputs/train/flux1dev_oscr_lora128/final \
  --output-dir outputs/debug/flux_cat_left_dog
```

The generation utility saves the generated image, the rendered OSCR condition,
and a JSON file containing predicted centers and 3D sizes.

## Official SeeThrough3D LoRA Baseline

The T2I-CompBench FLUX relation generator can also download and use the released
SeeThrough3D LoRA from Hugging Face. This keeps our GNN-predicted boxes but
replaces our trained LoRA with the official SeeThrough3D condition adapter:

```bash
python3 -m evaluation.generate_flux_relation_t2i \
  --prompt-file external/T2I-CompBench/examples/dataset/spatial_val.txt \
  --graph-encoder-path outputs/train/graph_pretrain_flux_3dbox_gpu7/final/graph_encoder.pt \
  --output-dir outputs/eval/flux_official_seethrough3d_quick \
  --use-official-seethrough3d-lora \
  --flux-quantization 4bit \
  --low-vram \
  --lora-rank 128 \
  --lora-alpha 128 \
  --image-size 384 \
  --oscr-size 256 \
  --oscr-render-size 512 \
  --condition-renderer blender
```

The official LoRA checkpoint is downloaded from
`va1bhavagrawa1/seethrough3d-flux.1-weights` using the model-card path
`checkpoints/seethrough3d_release/lora.safetensors`. Set `HF_HOME` or
`--official-lora-cache-dir` to keep the download out of home-directory quota.
