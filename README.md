# relation-aware-slot-attention

This branch is the FLUX.1-dev version of the project. It keeps the SCOP-Depth
data pipeline and relation-aware GNN, then uses the GNN to predict 3D object
centers and box sizes. Those predictions are rendered into SeeThrough3D-style
OSCR condition images, VAE-encoded as FLUX condition latents, and consumed by
the released SeeThrough3D FLUX LoRA during inference.

The current research focus is no longer FLUX fine-tuning. We train and debug the
GNN/layout side, then evaluate whether GNN-generated OSCR conditions improve a
strong pretrained FLUX+SeeThrough3D generator. This direction produced the first
clear benchmark gain: T2I-CompBench spatial improved from the FLUX baseline
around `0.24` to the GNN-generated OSCR + SeeThrough3D path around `0.37`.

## Setup

Run the full bootstrap:

```bash
./scripts/setup/bootstrap_all.sh
```

This creates `.venv`, installs the repo with FLUX/evaluation dependencies, clones
SeeThrough3D into `external/seethrough3d`, and prepares optional benchmark
checkouts. The FLUX generation code directly imports SeeThrough3D's FLUX
transformer fork and custom LoRA attention processors.

If you only need the SeeThrough3D checkout:

```bash
./scripts/setup/setup_seethrough3d_checkout.sh
```

## Current Workflow

1. Build SCOP-Depth cropped/depth data.
2. Pretrain the GNN with the active 3D_SLN-aligned triple-CVAE config.
3. Use the frozen GNN to predict 3D boxes for new prompts.
4. Render GNN-predicted 3D boxes into OSCR condition images.
5. Run FLUX.1-dev inference with the official SeeThrough3D LoRA.
6. Benchmark/debug generated images and OSCR conditions.

Example GNN training:

```bash
python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_triple_cvae_3dsln.yaml
```

This config uses FLUX T5 object-label embeddings, 5 triple-GNN layers,
scene-level and object-level CVAE latents, normalized min/max 3D box L1 loss,
and 3D_SLN-style KL regularization.

Example generation/debug:

```bash
python3 -m evaluation.generate_flux_relation_t2i \
  --prompt-file external/T2I-CompBench/examples/dataset/spatial_val.txt \
  --graph-encoder-path outputs/train/graph_pretrain_flux_3dbox_gpu7/final/graph_encoder.pt \
  --output-dir outputs/eval/flux_official_seethrough3d_quick \
  --use-official-seethrough3d-lora \
  --flux-quantization 8bit \
  --low-vram \
  --image-size 512 \
  --oscr-size 512 \
  --oscr-render-size 512 \
  --condition-renderer blender \
  --oscr-face-alpha 0.0025
```

The generation utility saves generated images, rendered OSCR visualizations,
binding prompts, and JSON records containing predicted centers and 3D sizes.

## Official SeeThrough3D LoRA Baseline

The T2I-CompBench FLUX relation generator can also download and use the released
SeeThrough3D LoRA from Hugging Face. This keeps our GNN-predicted boxes but
uses the official SeeThrough3D condition adapter:

```bash
python3 -m evaluation.generate_flux_relation_t2i \
  --prompt-file external/T2I-CompBench/examples/dataset/spatial_val.txt \
  --graph-encoder-path outputs/train/graph_pretrain_flux_3dbox_gpu7/final/graph_encoder.pt \
  --output-dir outputs/eval/flux_official_seethrough3d_quick \
  --use-official-seethrough3d-lora \
  --flux-quantization 8bit \
  --low-vram \
  --image-size 512 \
  --oscr-size 512 \
  --oscr-render-size 512 \
  --condition-renderer blender \
  --oscr-face-alpha 0.0025
```

The official LoRA checkpoint is downloaded from
`va1bhavagrawa1/seethrough3d-flux.1-weights` using the model-card path
`checkpoints/seethrough3d_release/lora.safetensors`. Set `HF_HOME` or
`--official-lora-cache-dir` to keep the download out of home-directory quota.
