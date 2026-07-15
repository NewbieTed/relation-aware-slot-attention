# RELAY-3D: Scene-Graph 3D Layout Prediction 

This branch is the FLUX.1-dev version of the project. It keeps the SCOP-Depth
data pipeline and relation-aware GNN, then uses the GNN to predict normalized
3D object cuboids. Those predictions can be rendered into SeeThrough3D-style
OSCR condition images, VAE-encoded as FLUX condition latents, and consumed by
the released SeeThrough3D FLUX LoRA during inference.

The current research focus is no longer FLUX fine-tuning. We train and evaluate
the GNN/layout side directly, then use GNN-generated OSCR conditions with the
official SeeThrough3D FLUX LoRA for qualitative/downstream image generation.
The image-generation direction produced the first clear benchmark gain:
T2I-CompBench spatial improved from the FLUX baseline around `0.24` to the
GNN-generated OSCR + SeeThrough3D path around `0.37`.

The latest quantitative focus is layout-module evaluation rather than
T2I-CompBench. See `docs/paper_table.md` for the current paper table. The main
takeaways as of 2026-05-25 are:

- Hand-coded relation heuristics can trivially reach `1.0` relation accuracy,
  so relation accuracy alone is not sufficient.
- Deterministic GNNs trained on prompt-balanced augmented data improve relation
  validity over the original deterministic GNN, especially 3D relation validity,
  but still have zero stochastic diversity.
- CVAE models with freebits avoid collapse and produce meaningful valid
  diversity; no-freebits CVAEs have much lower spread.
- Prompt-balanced augmented CVAEs produce the largest valid diversity, but trade
  off reference-box L1/IoU and some relation accuracy. This is the main current
  analysis point for the paper/future-work section.

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
2. Optionally build prompt-balanced augmented layout metadata.
3. Train deterministic GNN and triple-CVAE GNN layout models.
4. Evaluate predicted 3D boxes directly with relation, L1/IoU, validity, and
   diversity metrics.
5. Use the frozen GNN to predict 3D boxes for new prompts.
6. Render GNN-predicted 3D boxes into OSCR condition images.
7. Run FLUX.1-dev inference with the official SeeThrough3D LoRA for downstream
   qualitative checks.

Example GNN training:

```bash
python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_orig_cvae_nofilm_freebits1_bs128_3200.yaml
```

The active CVAE configs use FLUX T5 object-label embeddings, 5 triple-GNN
layers, 3D_SLN-style per-object CVAE latents, normalized min/max 3D box L1 loss,
and KL regularization with optional freebits. Rotation/yaw is not modeled yet.

Example layout evaluation:

```bash
python3 -m evaluation.evaluate_layout_models \
  --config configs/flux/eval/layout/layout_eval_ready_now.yaml
```

The evaluator writes `paper_table.md`, `paper_table.csv`,
`metrics_summary.csv`, `per_sample_metrics.csv`, and `per_relation_metrics.csv`
under `outputs/eval/...`. The current downloaded paper table is checked in as
`docs/paper_table.md`.

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
