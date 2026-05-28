# Evaluation And Debugging

This FLUX branch keeps evaluation code that is still useful for the current
architecture:

- `evaluation.generate_flux_relation`: generate one FLUX image from a prompt,
  using the frozen GNN to render an OSCR condition image.
- `evaluation.generate_flux_relation_t2i`: generate T2I-CompBench prompt-file
  samples with GNN-predicted OSCR conditions and the released SeeThrough3D LoRA.
- `evaluation.debug_gnn_prompt`: print detailed GNN/CVAE message-passing and
  layout-head traces for a prompt.
- `evaluation.debug_layout_spread`: print compact per-object center/size spread
  summaries across many stochastic layout samples.
- `evaluation.debug_dataset_layout_spread`: print the same compact spread
  summary for target boxes stored in a SCOP-style `metadata.jsonl` folder.
- `evaluation.evaluate_layout_models`: compare layout predictors directly on
  held-out SCOP-Depth metadata with relation accuracy, box L1, 2D/3D IoU,
  out-of-bounds/overlap rates, best-of-K reference metrics, and valid diversity.
- `evaluation.visualize_gnn_layout`: visualize predicted GNN centers and box regions.
- `evaluation.t2i_compbench`: score an existing generated samples directory with
  a prepared T2I-CompBench checkout.
- `evaluation.geneval` and `evaluation.visor`: wrappers for existing generated
  outputs when those benchmarks are needed.

The old Stable Diffusion generator and slot-attention inspection tools have been
removed from this branch. If we need those experiments again, switch back to the
SD branch rather than reintroducing them here.

Example FLUX generation:

```bash
python3 -m evaluation.generate_flux_relation_t2i \
  --config configs/flux/eval_official_seethrough3d_spatial_20p_bf16_512.yaml
```

Generation-only prompt add-ons can be supplied without changing the relation
graph used by the GNN:

```yaml
generate:
  background_prompt: in a bright kitchen
  style_prompt: natural photo
  quality_prompt: high detail
```

The parser also keeps comma-style prompt-file add-ons out of object labels, so
`a dog to the left of a chair, in a bright kitchen` still parses as `dog` and
`chair` for layout.

Example GNN/CVAE trace:

```bash
python3 -m evaluation.debug_gnn_prompt \
  --prompt "a suitcase in front of an apple" \
  --graph-encoder-path outputs/train/graph_pretrain_flux_3dbox_cvae/final/graph_encoder.pt \
  --output-dir outputs/debug/gnn_cvae_trace \
  --layout-sample-mode prior_sample \
  --seed 42
```

Example layout-module evaluation:

```bash
python3 -m evaluation.evaluate_layout_models \
  --config configs/flux/eval/layout/layout_eval_ready_now.yaml
```

The layout evaluator intentionally treats GT box L1/IoU as reference metrics,
not as the whole definition of success. For stochastic CVAE methods, it also
reports best-of-K and nearest same-prompt reference scores, plus diversity only
among samples that still satisfy the requested relation.

For depth/occlusion relations (`in_front_of`, `behind`, `hidden_by`), `rel_acc`
requires both the correct z-order and projected 2D box overlap. The legacy
order-only check is still written as `rel_order_acc`, which helps diagnose
models that learn depth ordering without learning visual occlusion.
The compact paper table also splits relation accuracy into `2D Rel Acc` for
left/right/above/below/on/next-to and `3D Rel Acc` for front/behind/hidden-by.

The current paper-facing result table is checked in at `docs/paper_table.md`.
That table compares class/relation priors, hand-coded relation heuristics,
original deterministic GNN, deterministic GNNs trained on prompt-balanced
augmentation, base-data CVAE ablations, prompt-balanced augmented CVAEs, and a
random-jitter baseline.

The most useful columns for the current paper story are:

- `Rel Acc`: all relation types, with 3D relations requiring both z-order and
  projected overlap.
- `2D Rel Acc`: left/right/above/below/on/next-to only.
- `3D Rel Acc`: front/behind/hidden-by with z-order and projected overlap.
- `3D Order Acc`: front/behind/hidden-by depth order only.
- `Occ. Overlap`: projected overlap for front/behind/hidden-by.
- `Box L1`, `2D IoU`, `3D IoU`: reference-box fidelity diagnostics.
- `Center STD`, `Size STD`, `Valid Diversity`: stochastic spread diagnostics.
- `Valid Rate`, `OOB`, `Overlap`: validity/collision diagnostics.

Read the table with one caveat: `relation_heuristic` is designed to satisfy
relations, so its `Rel Acc = 1.0` is a sanity baseline rather than a learned
result. The learned models should be interpreted through the combined tradeoff
between relation validity, reference-box quality, and valid diversity.

Example scoring of an already generated T2I-CompBench directory:

```bash
python3 -m evaluation.t2i_compbench \
  --benchmark spatial \
  --t2i-compbench-root external/T2I-CompBench \
  --generated-dir outputs/eval/my_flux_spatial_run \
  --prompt-file external/T2I-CompBench/examples/dataset/spatial_val.txt \
  --python-bin python3
```
