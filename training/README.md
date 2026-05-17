# Training Workflow

This FLUX branch now trains only the relation-aware GNN/layout module. FLUX
fine-tuning experiments were removed from the active workflow after the official
SeeThrough3D LoRA plus our GNN-generated OSCR conditions gave the first clear
spatial benchmark gain.

`training.pretrain_graph_encoder` pretrains the graph encoder on SCOP-Depth
normalized 3D min/max box targets. The resulting checkpoint is used at inference
time to predict object cuboids, which are rendered into OSCR conditions for the
released SeeThrough3D FLUX LoRA.

The active graph encoder uses `layout_mode: triple_cvae`, a 3D_SLN-style variant
that updates contextual subject-relation-object edge states, samples both
scene-level and object-level latents, and runs a decoder triple-GNN before
predicting normalized 3D min/max boxes.

The graph trainer creates deterministic train/eval/test splits, saves
`data_split.json`, writes losses to `metrics.jsonl` and `metrics.csv`, and saves
reusable checkpoints under the run output directory.

Before training, `training.precompute_graph_label_cache` can fill the
`label_embedding_cache` path from the same YAML config. This keeps the frozen
T5/CLIP object-label embeddings out of the hot training loop; training will
still lazily encode and append any unseen labels if the cache is incomplete.

`training.augment_scop_layouts` creates a derived SCOP-Depth dataset for layout
diversity experiments. It reuses the original cropped images by symlinking them
into a new dataset folder, writes new relation-preserving 3D box/depth values
to `metadata.jsonl`, saves an `augmentation_report.json`, and renders sampled
box overlays under `samples/`. The sampler mixes category-level empirical
size/position priors, relation-level empirical offsets, jitter around the
original row, and broad synthetic relation-valid layouts. The report includes
relation checks, so a build fails loudly if any sampled layout violates the
requested relation.

Example graph pretraining:

```bash
python3 -m training.precompute_graph_label_cache \
  --config configs/flux/gnn_pretrain_3dbox_triple_cvae_3dsln.yaml

python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_triple_cvae_3dsln.yaml
```

Example augmented-layout dataset build:

```bash
python3 -m training.augment_scop_layouts \
  --input-dir /local1/cse_481_m_l/relation-aware-slot-attention/data/scop_depth_crops_depth \
  --output-dir /local1/cse_481_m_l/relation-aware-slot-attention/data/scop_depth_crops_depth_aug_rel4 \
  --variants-per-row 4 \
  --limit-rows 100 \
  --num-samples 48 \
  --seed 42
```

For graph pretraining on a large augmented dataset, the trainer uses metadata
only and skips image loading. This keeps the symlinked crops available for
visual inspection without spending training time decoding images that the GNN
does not consume.

For one-prompt distribution tests, set `prompt_filter` in the config. The filter
matches the exact prompt produced by `prompt_prefix` plus the SCOP relation, so
all train/eval/test rows share the same text graph while retaining different
augmented target boxes.

`training.augment_scop_layouts` can also build a one-prompt augmented dataset by
combining `--prompt-filter` with `--target-augmented-rows`. In this mode it
samples matching source rows with replacement until the requested number of
valid bbox pairs has been written.
